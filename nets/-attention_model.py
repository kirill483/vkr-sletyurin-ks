import math
from typing import NamedTuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nets.graph_encoder import GraphAttentionEncoder


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, activation=nn.ReLU):
        super().__init__()

        if n_layers <= 1:
            self.net = nn.Linear(input_dim, output_dim)
            return

        layers = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointerFixed(NamedTuple):
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor




class AttentionModel(nn.Module):
    """
    Hierarchical Attention Model for GTSP-like coverage routing.

    Public interface is intentionally close to the original AM code:
        - class name: AttentionModel
        - problem.NAME can stay "tsp"
        - forward(input, return_pi=False) returns cost, ll, optionally pi
        - pi contains flat candidate actions in 1..N*K

    Internal factorization:
        P(solution) = Π_t P(field_t | state_t) * P(template_t | field_t, state_t)
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalization="batch",
        n_heads=8,
        checkpoint_encoder=False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder

        self.n_templates = 8
        self.template_feature_dim = 4

        assert embedding_dim % n_heads == 0

        # Depot embedding: [B, 2] -> [B, D]
        self.init_embed_depot = nn.Linear(2, embedding_dim)

        # Template embedding inside every field: [entry_x, entry_y, exit_x, exit_y] -> D
        self.init_embed_template = MLP(
            input_dim=self.template_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=2,
        )

        # Local encoder over the 8 templates of each field.
        # Input shape is flattened from [B, N, K, D] to [B*N, K, D].
        self.template_set_encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=1,
            normalization=normalization,
        )

        # Learned pooling K templates -> one field embedding.
        self.template_pool_score = nn.Linear(embedding_dim, 1)

        # Global encoder over [depot, field_1, ..., field_N].
        self.field_encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
        )

        # Build final per-template candidate embeddings after global field encoding.
        # Combines:
        #   encoded field embedding
        #   locally encoded template embedding
        #   raw template coordinates
        self.template_candidate_mlp = MLP(
            input_dim=2 * embedding_dim + self.template_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=2,
        )

        # Previous action embedding is depot embedding on step 0,
        # otherwise embedding of the previously selected concrete template.
        # Area decoder context:
        #   remaining context + current field + previous action + current coord + depot
        self.project_area_context = nn.Linear(
            4 * embedding_dim + 2,
            embedding_dim,
            bias=False,
        )

        # Template decoder context:
        #   future context + selected field + previous action + current coord + depot
        self.project_template_context = nn.Linear(
            4 * embedding_dim + 2,
            embedding_dim,
            bias=False,
        )

        # Dynamic template scorer enrichment.
        # Per selected template candidate combines:
        #   base template candidate embedding
        #   future context
        #   entry - current_exit
        #   ||entry - current_exit||
        #   depot - exit
        #   ||depot - exit||
        self.template_dynamic_mlp = MLP(
            input_dim=2 * embedding_dim + 2 + 1 + 2 + 1,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=2,
        )

        # Separate pointer projections for area-level and template-level decisions.
        self.project_area_node_embeddings = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=False,
        )
        self.project_template_node_embeddings = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=False,
        )

        self.project_area_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_template_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        input:
            {
                'depot':     [B, 2],
                'templates': [B, N, 8, 4]
            }

        return:
            cost: [B]
            ll:   [B]
            pi:   [B, N], optional, values in 1..N*8
        """

        if self.checkpoint_encoder and self.training:
            field_embeddings, template_candidate_embeddings = checkpoint(
                self._encode_input,
                input["depot"],
                input["templates"],
            )
        else:
            field_embeddings, template_candidate_embeddings = self._encode_input(
                input["depot"],
                input["templates"],
            )

        node_in, node_out = self._make_node_in_out(input)

        log_likelihood, pi = self._inner(
            input=input,
            field_embeddings=field_embeddings,
            template_candidate_embeddings=template_candidate_embeddings,
            node_in=node_in,
            node_out=node_out,
        )

        cost, mask = self.problem.get_costs(input, pi)

        # mask is kept for compatibility with original AM problems.
        # For this problem get_costs returns None.
        if mask is not None:
            log_likelihood = log_likelihood.masked_fill(mask.any(dim=1), 0)

        if return_pi:
            return cost, log_likelihood, pi

        return cost, log_likelihood

    def _encode_input(self, depot, templates):
        """
        depot:
            [B, 2]
        templates:
            [B, N, K, 4]

        returns:
            field_embeddings:
                [B, N + 1, D]
                0    = depot
                1..N = encoded fields

            template_candidate_embeddings:
                [B, N, K, D]
        """

        B, N, K, F = templates.size()
        assert K == self.n_templates
        assert F == self.template_feature_dim

        # 1) Encode templates inside each field.
        template_init = self.init_embed_template(templates)  # [B, N, K, D]

        template_init_flat = template_init.reshape(B * N, K, self.embedding_dim)
        template_encoded_flat, _ = self.template_set_encoder(template_init_flat)
        template_encoded = template_encoded_flat.reshape(B, N, K, self.embedding_dim)

        # 2) Aggregate 8 templates -> field embedding.
        pool_logits = self.template_pool_score(template_encoded).squeeze(-1)  # [B, N, K]
        pool_weights = torch.softmax(pool_logits, dim=-1)
        field_init = torch.sum(template_encoded * pool_weights[:, :, :, None], dim=2)

        depot_embedding = self.init_embed_depot(depot)[:, None, :]  # [B, 1, D]

        # 3) Global field encoder over depot + fields.
        encoder_input = torch.cat((depot_embedding, field_init), dim=1)
        field_embeddings, _ = self.field_encoder(encoder_input)

        encoded_fields = field_embeddings[:, 1:, :]  # [B, N, D]

        # 4) Build concrete template candidate embeddings.
        encoded_fields_expanded = encoded_fields[:, :, None, :].expand(B, N, K, self.embedding_dim)
        candidate_input = torch.cat(
            (
                encoded_fields_expanded,
                template_encoded,
                templates,
            ),
            dim=-1,
        )
        template_candidate_embeddings = self.template_candidate_mlp(candidate_input)

        return field_embeddings, template_candidate_embeddings

    def _make_node_in_out(self, input):
        templates = input["templates"]
        depot = input["depot"]

        B, N, K, F = templates.size()
        assert K == self.n_templates
        assert F == self.template_feature_dim

        candidate_in = templates[..., 0:2].reshape(B, N * K, 2)
        candidate_out = templates[..., 2:4].reshape(B, N * K, 2)

        depot_coord = depot[:, None, :]

        node_in = torch.cat((depot_coord, candidate_in), dim=1)
        node_out = torch.cat((depot_coord, candidate_out), dim=1)

        return node_in, node_out

    def _inner(self, input, field_embeddings, template_candidate_embeddings, node_in, node_out):
        outputs = []
        sequences = []

        templates = input["templates"]
        depot = input["depot"]

        B, N, K, _ = templates.size()

        state = self.problem.make_state(
            input,
            node_in=node_in,
            node_out=node_out,
        )

        encoded_fields = field_embeddings[:, 1:, :]      # [B, N, D]
        depot_embedding = field_embeddings[:, 0, :]      # [B, D]

        area_fixed = self._precompute_pointer(
            encoded_fields,
            self.project_area_node_embeddings,
        )

        log_likelihood = torch.zeros(B, device=templates.device)

        while not state.all_finished():
            # -------- Level 1: choose field / area --------
            remaining_context = self._masked_mean_or_depot(
                encoded_fields=encoded_fields,
                visited_fields=state.visited_fields,
                depot_embedding=depot_embedding,
            )

            area_log_p, area_mask = self._get_area_log_p(
                field_embeddings=field_embeddings,
                template_candidate_embeddings=template_candidate_embeddings,
                area_fixed=area_fixed,
                state=state,
                remaining_context=remaining_context,
                depot_embedding=depot_embedding,
            )

            selected_field = self._select_node(
                area_log_p.exp()[:, 0, :],
                area_mask[:, 0, :],
            )  # [B], 0..N-1

            selected_area_log_p = area_log_p[:, 0, :].gather(
                1,
                selected_field[:, None],
            ).squeeze(1)

            # Future context should exclude the selected field.
            future_visited = state.visited_fields.scatter(
                -1,
                selected_field[:, None, None],
                True,
            )

            future_context = self._masked_mean_or_depot(
                encoded_fields=encoded_fields,
                visited_fields=future_visited,
                depot_embedding=depot_embedding,
            )

            # -------- Level 2: choose template inside selected field --------
            template_log_p, template_mask = self._get_template_log_p(
                input=input,
                field_embeddings=field_embeddings,
                template_candidate_embeddings=template_candidate_embeddings,
                state=state,
                selected_field=selected_field,
                future_context=future_context,
                depot_embedding=depot_embedding,
            )

            selected_template = self._select_node(
                template_log_p.exp()[:, 0, :],
                template_mask[:, 0, :],
            )  # [B], 0..K-1

            selected_template_log_p = template_log_p[:, 0, :].gather(
                1,
                selected_template[:, None],
            ).squeeze(1)

            selected_action = 1 + selected_field * K + selected_template  # [B], 1..N*K

            log_likelihood = log_likelihood + selected_area_log_p + selected_template_log_p

            state = state.update(selected_action)

            outputs.append(selected_action)
            sequences.append(selected_action)

        pi = torch.stack(sequences, dim=1)  # [B, N]

        return log_likelihood, pi

    def _get_area_log_p(
        self,
        field_embeddings,
        template_candidate_embeddings,
        area_fixed,
        state,
        remaining_context,
        depot_embedding,
        normalize=True,
    ):
        prev_field_embedding = self._get_prev_field_embedding(field_embeddings, state)
        prev_action_embedding = self._get_prev_action_embedding(
            template_candidate_embeddings=template_candidate_embeddings,
            depot_embedding=depot_embedding,
            state=state,
        )

        context = torch.cat(
            (
                remaining_context[:, None, :],
                prev_field_embedding,
                prev_action_embedding,
                state.cur_coord,
                depot_embedding[:, None, :],
            ),
            dim=-1,
        )

        query = self.project_area_context(context)
        mask = state.get_area_mask()

        logits, _ = self._one_to_many_logits(
            query=query,
            glimpse_K=area_fixed.glimpse_key,
            glimpse_V=area_fixed.glimpse_val,
            logit_K=area_fixed.logit_key,
            mask=mask,
            project_out=self.project_area_out,
        )

        if normalize:
            log_p = torch.log_softmax(logits / self.temp, dim=-1)
        else:
            log_p = logits

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _get_template_log_p(
        self,
        input,
        field_embeddings,
        template_candidate_embeddings,
        state,
        selected_field,
        future_context,
        depot_embedding,
        normalize=True,
    ):
        templates = input["templates"]
        depot = input["depot"]

        B, N, K, _ = templates.size()

        selected_field_embedding = field_embeddings[:, 1:, :].gather(
            1,
            selected_field[:, None, None].expand(B, 1, self.embedding_dim),
        )  # [B, 1, D]

        prev_action_embedding = self._get_prev_action_embedding(
            template_candidate_embeddings=template_candidate_embeddings,
            depot_embedding=depot_embedding,
            state=state,
        )

        selected_template_embeddings = template_candidate_embeddings.gather(
            1,
            selected_field[:, None, None, None].expand(B, 1, K, self.embedding_dim),
        ).squeeze(1)  # [B, K, D]

        selected_templates = templates.gather(
            1,
            selected_field[:, None, None, None].expand(B, 1, K, self.template_feature_dim),
        ).squeeze(1)  # [B, K, 4]

        entry = selected_templates[..., 0:2]
        exit_ = selected_templates[..., 2:4]

        rel_entry = entry - state.cur_coord.expand(B, K, 2)
        rel_entry_dist = rel_entry.norm(p=2, dim=-1, keepdim=True)

        depot_to_exit = depot[:, None, :] - exit_
        depot_to_exit_dist = depot_to_exit.norm(p=2, dim=-1, keepdim=True)

        dynamic_template_input = torch.cat(
            (
                selected_template_embeddings,
                future_context[:, None, :].expand(B, K, self.embedding_dim),
                rel_entry,
                rel_entry_dist,
                depot_to_exit,
                depot_to_exit_dist,
            ),
            dim=-1,
        )

        dynamic_template_embeddings = self.template_dynamic_mlp(dynamic_template_input)

        template_fixed = self._precompute_pointer(
            dynamic_template_embeddings,
            self.project_template_node_embeddings,
        )

        context = torch.cat(
            (
                future_context[:, None, :],
                selected_field_embedding,
                prev_action_embedding,
                state.cur_coord,
                depot_embedding[:, None, :],
            ),
            dim=-1,
        )

        query = self.project_template_context(context)

        # Inside a selected unvisited field all K templates are allowed.
        template_mask = torch.zeros(
            B, 1, K,
            dtype=torch.bool,
            device=templates.device,
        )

        logits, _ = self._one_to_many_logits(
            query=query,
            glimpse_K=template_fixed.glimpse_key,
            glimpse_V=template_fixed.glimpse_val,
            logit_K=template_fixed.logit_key,
            mask=template_mask,
            project_out=self.project_template_out,
        )

        if normalize:
            log_p = torch.log_softmax(logits / self.temp, dim=-1)
        else:
            log_p = logits

        assert not torch.isnan(log_p).any()
        return log_p, template_mask

    def _get_prev_field_embedding(self, field_embeddings, state):
        B = field_embeddings.size(0)
        return field_embeddings.gather(
            1,
            state.prev_field[:, :, None].expand(B, 1, self.embedding_dim),
        )

    def _get_prev_action_embedding(self, template_candidate_embeddings, depot_embedding, state):
        """
        Returns:
            [B, 1, D]

        If prev_a == 0, returns depot embedding.
        Otherwise gathers the concrete previous template embedding.
        """
        B, N, K, D = template_candidate_embeddings.size()
        flat_templates = template_candidate_embeddings.reshape(B, N * K, D)

        prev_a = state.prev_a  # [B, 1], 0 or 1..N*K
        gather_idx = (prev_a - 1).clamp(min=0)

        prev_template_embedding = flat_templates.gather(
            1,
            gather_idx[:, :, None].expand(B, 1, D),
        )

        depot_embedding = depot_embedding[:, None, :]
        is_depot = prev_a.eq(0)[:, :, None]

        return torch.where(is_depot, depot_embedding, prev_template_embedding)

    def _masked_mean_or_depot(self, encoded_fields, visited_fields, depot_embedding):
        """
        encoded_fields:
            [B, N, D]
        visited_fields:
            [B, 1, N]
        depot_embedding:
            [B, D]

        Returns masked mean of unvisited fields.
        If no unvisited field remains, returns depot embedding.
        """
        valid = (~visited_fields.squeeze(1)).float()  # [B, N]
        count = valid.sum(dim=1, keepdim=True)        # [B, 1]

        summed = torch.sum(encoded_fields * valid[:, :, None], dim=1)
        mean = summed / count.clamp_min(1.0)

        return torch.where(count.gt(0), mean, depot_embedding)

    def _precompute_pointer(self, embeddings, projection_layer):
        """
        embeddings:
            [B, M, D]
        """
        glimpse_key, glimpse_val, logit_key = projection_layer(
            embeddings[:, None, :, :]
        ).chunk(3, dim=-1)

        return PointerFixed(
            glimpse_key=self._make_heads(glimpse_key),
            glimpse_val=self._make_heads(glimpse_val),
            logit_key=logit_key.contiguous(),
        )

    def _make_heads(self, v):
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any NaNs"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"

        return selected

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, project_out):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(
            batch_size,
            num_steps,
            self.n_heads,
            1,
            key_size,
        ).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(
            glimpse_Q,
            glimpse_K.transpose(-2, -1),
        ) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = project_out(
            heads.permute(1, 2, 3, 0, 4)
            .contiguous()
            .view(-1, num_steps, 1, self.n_heads * val_size)
        )

        final_Q = glimpse

        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2)
        logits = logits / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)


