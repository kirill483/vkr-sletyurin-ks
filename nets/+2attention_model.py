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
    Order-only Attention Model + exact DP template optimizer.

    Public interface stays close to the original AM code:
        - class name: AttentionModel
        - problem.NAME can stay "tsp"
        - forward(input, return_pi=False) returns cost, ll, optionally pi
        - pi contains flat candidate actions in 1..N*K

    Internal factorization:
        1) neural decoder samples/greedily selects only field order:
               P(order) = Π_t P(field_t | state_t)
        2) for the selected field order, templates are optimized exactly by DP:
               min over template sequence of transition + coverage cost

    Input templates must have shape:
        [B, N, K, 5]
    where each template is:
        [x_in, y_in, x_out, y_out, coverage_cost]
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
        self.n_encode_layers = n_encode_layers + 2
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder

        self.n_templates = 8
        self.template_feature_dim = 5

        assert embedding_dim % n_heads == 0

        # Depot embedding: [B, 2] -> [B, D]
        self.init_embed_depot = nn.Linear(2, embedding_dim)

        # Template embedding: [entry_x, entry_y, exit_x, exit_y, coverage_cost] -> [D]
        self.init_embed_template = MLP(
            input_dim=self.template_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=2,
        )

        # Local encoder over templates inside every field.
        # [B, N, K, D] -> [B*N, K, D] -> [B, N, K, D]
        self.template_set_encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=1,
            normalization=normalization,
        )

        # Learned pooling K templates -> one field embedding.
        self.template_pool_score = nn.Linear(embedding_dim, 1)

        # Extra field-level geometric statistics from raw templates.
        # mean_entry(2), mean_exit(2), min_entry(2), max_entry(2),
        # min_exit(2), max_exit(2), min_cov, mean_cov, max_cov = 15 dims.
        self.field_stats_dim = 15
        self.field_stats_embed = MLP(
            input_dim=self.field_stats_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=2,
        )

        self.field_fusion = MLP(
            input_dim=2 * embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=2,
        )

        # Global encoder over [depot, field_1, ..., field_N].
        self.field_encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
        )

        # Area decoder context:
        #   remaining_context
        #   visited_context
        #   previous field embedding
        #   first field embedding
        #   depot embedding
        #   step_fraction
        self.project_area_context = nn.Linear(
            5 * embedding_dim + 1,
            embedding_dim,
            bias=False,
        )

        # Pointer projections for area-level decisions.
        self.project_area_node_embeddings = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=False,
        )
        self.project_area_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Candidate-specific pair bias. This compensates for the lack of concrete current_exit
        # before DP selects templates.
        # Features per candidate field j:
        #   min_prev_to_j
        #   mean_prev_to_j
        #   min_j_to_depot
        #   mean_j_to_depot
        #   center_distance_prev_j
        #   min_coverage_j
        #   mean_coverage_j
        #   max_coverage_j
        #   remaining_fraction
        self.pair_feature_dim = 9
        self.pair_bias_mlp = MLP(
            input_dim=self.pair_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=2,
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        input:
            {
                'depot':     [B, 2],
                'templates': [B, N, 8, 5]
            }

        return:
            cost: [B]
            ll:   [B]
            pi:   [B, N], optional, values in 1..N*8
        """

        if self.checkpoint_encoder and self.training:
            field_embeddings = checkpoint(
                self._encode_input,
                input["depot"],
                input["templates"],
            )
        else:
            field_embeddings = self._encode_input(input["depot"], input["templates"])

        node_in, node_out = self._make_node_in_out(input)

        log_likelihood, pi = self._inner(
            input=input,
            field_embeddings=field_embeddings,
            node_in=node_in,
            node_out=node_out,
        )

        cost, mask = self.problem.get_costs(input, pi)

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
            [B, N, K, 5]

        returns:
            field_embeddings:
                [B, N + 1, D]
                0    = depot
                1..N = encoded fields
        """
        B, N, K, F = templates.size()
        assert K == self.n_templates
        assert F == self.template_feature_dim

        # 1) Encode templates inside every field.
        template_init = self.init_embed_template(templates)  # [B, N, K, D]

        template_init_flat = template_init.reshape(B * N, K, self.embedding_dim)
        template_encoded_flat, _ = self.template_set_encoder(template_init_flat)
        template_encoded = template_encoded_flat.reshape(B, N, K, self.embedding_dim)

        # 2) Learned attention pooling: K templates -> one field embedding.
        pool_logits = self.template_pool_score(template_encoded).squeeze(-1)  # [B, N, K]
        pool_weights = torch.softmax(pool_logits, dim=-1)
        pooled_template_field = torch.sum(template_encoded * pool_weights[:, :, :, None], dim=2)

        # 3) Add raw geometric statistics of the field.
        field_stats = self._make_field_stats(templates)  # [B, N, 15]
        field_stats_embedding = self.field_stats_embed(field_stats)

        field_init = self.field_fusion(torch.cat((pooled_template_field, field_stats_embedding), dim=-1))

        depot_embedding = self.init_embed_depot(depot)[:, None, :]  # [B, 1, D]

        # 4) Global encoder over depot + fields.
        encoder_input = torch.cat((depot_embedding, field_init), dim=1)
        field_embeddings, _ = self.field_encoder(encoder_input)

        return field_embeddings

    def _make_field_stats(self, templates):
        entry = templates[..., 0:2]      # [B, N, K, 2]
        exit_ = templates[..., 2:4]      # [B, N, K, 2]
        coverage = templates[..., 4]     # [B, N, K]

        mean_entry = entry.mean(dim=2)
        mean_exit = exit_.mean(dim=2)
        min_entry = entry.min(dim=2).values
        max_entry = entry.max(dim=2).values
        min_exit = exit_.min(dim=2).values
        max_exit = exit_.max(dim=2).values

        min_cov = coverage.min(dim=2).values[..., None]
        mean_cov = coverage.mean(dim=2)[..., None]
        max_cov = coverage.max(dim=2).values[..., None]

        return torch.cat(
            (
                mean_entry,
                mean_exit,
                min_entry,
                max_entry,
                min_exit,
                max_exit,
                min_cov,
                mean_cov,
                max_cov,
            ),
            dim=-1,
        )

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

    def _inner(self, input, field_embeddings, node_in, node_out):
        templates = input["templates"]
        B, N, K, _ = templates.size()

        state = self.problem.make_state(input, node_in=node_in, node_out=node_out)

        encoded_fields = field_embeddings[:, 1:, :]  # [B, N, D]
        depot_embedding = field_embeddings[:, 0, :]  # [B, D]

        area_fixed = self._precompute_pointer(
            encoded_fields,
            self.project_area_node_embeddings,
        )

        log_likelihood = torch.zeros(B, device=templates.device)
        field_sequence = []

        while not state.all_finished():
            area_log_p, area_mask = self._get_area_log_p(
                input=input,
                field_embeddings=field_embeddings,
                area_fixed=area_fixed,
                state=state,
                encoded_fields=encoded_fields,
                depot_embedding=depot_embedding,
            )

            selected_field = self._select_node(
                area_log_p.exp()[:, 0, :],
                area_mask[:, 0, :],
            )  # [B], values 0..N-1

            selected_log_p = area_log_p[:, 0, :].gather(1, selected_field[:, None]).squeeze(1)
            log_likelihood = log_likelihood + selected_log_p

            field_sequence.append(selected_field)
            state = state.update_field(selected_field)

        field_order = torch.stack(field_sequence, dim=1)  # [B, N], 0..N-1

        # Exact DP chooses the best template for each selected field in this order.
        _, pi_actions = self._solve_templates_dp(input, field_order)

        return log_likelihood, pi_actions

    def _get_area_log_p(
        self,
        input,
        field_embeddings,
        area_fixed,
        state,
        encoded_fields,
        depot_embedding,
        normalize=True,
    ):
        remaining_context = self._masked_mean_or_depot(
            encoded_fields=encoded_fields,
            mask=~state.visited_fields.squeeze(1),
            depot_embedding=depot_embedding,
        )

        visited_context = self._masked_mean_or_depot(
            encoded_fields=encoded_fields,
            mask=state.visited_fields.squeeze(1),
            depot_embedding=depot_embedding,
        )

        prev_field_embedding = self._gather_field_embedding(field_embeddings, state.prev_field)
        first_field_embedding = self._gather_field_embedding(field_embeddings, state.first_field)

        B = encoded_fields.size(0)
        N = encoded_fields.size(1)
        step_fraction = (state.i.float() / float(N)).view(1, 1, 1).expand(B, 1, 1)

        context = torch.cat(
            (
                remaining_context[:, None, :],
                visited_context[:, None, :],
                prev_field_embedding,
                first_field_embedding,
                depot_embedding[:, None, :],
                step_fraction,
            ),
            dim=-1,
        )

        query = self.project_area_context(context)

        if hasattr(state, "get_area_mask"):
            mask = state.get_area_mask()
        else:
            mask = state.visited_fields

        logits, _ = self._one_to_many_logits(
            query=query,
            glimpse_K=area_fixed.glimpse_key,
            glimpse_V=area_fixed.glimpse_val,
            logit_K=area_fixed.logit_key,
            mask=mask,
            project_out=self.project_area_out,
        )

        # Add candidate-specific geometric bias based on raw template sets.
        pair_bias = self._compute_pair_bias(input, state)  # [B, 1, N]
        logits = logits + pair_bias

        # Re-apply mask after adding bias for safety.
        if self.mask_logits:
            logits[mask] = -math.inf

        if normalize:
            log_p = torch.log_softmax(logits / self.temp, dim=-1)
        else:
            log_p = logits

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _compute_pair_bias(self, input, state):
        """
        Computes learned geometric bias for selecting the next field.

        Returns:
            [B, 1, N]
        """
        templates = input["templates"]
        depot = input["depot"]

        B, N, K, _ = templates.size()
        entry = templates[..., 0:2]       # [B, N, K, 2]
        exit_ = templates[..., 2:4]       # [B, N, K, 2]
        coverage = templates[..., 4]      # [B, N, K]

        # Candidate -> depot stats.
        cand_to_depot = (exit_ - depot[:, None, None, :]).norm(p=2, dim=-1)  # [B, N, K]
        min_j_to_depot = cand_to_depot.min(dim=2).values
        mean_j_to_depot = cand_to_depot.mean(dim=2)

        # Coverage stats.
        min_cov = coverage.min(dim=2).values
        mean_cov = coverage.mean(dim=2)
        max_cov = coverage.max(dim=2).values

        # Field centers for center-distance feature.
        field_center = torch.cat((entry, exit_), dim=2).mean(dim=2)  # [B, N, 2]

        prev_field = state.prev_field  # [B, 1], 0 = depot, 1..N
        prev_is_depot = prev_field.eq(0)  # [B, 1]
        prev_idx = (prev_field - 1).clamp(min=0)  # [B, 1]

        # Depot -> candidate stats for the first step.
        depot_to_entry = (entry - depot[:, None, None, :]).norm(p=2, dim=-1)  # [B, N, K]
        depot_to_j_min = depot_to_entry.min(dim=2).values
        depot_to_j_mean = depot_to_entry.mean(dim=2)

        # Previous field exits -> candidate entries for normal steps.
        prev_exit = exit_.gather(
            1,
            prev_idx[:, :, None, None].expand(B, 1, K, 2),
        ).squeeze(1)  # [B, K, 2]

        pair_dist = (
            prev_exit[:, None, :, None, :] - entry[:, :, None, :, :]
        ).norm(p=2, dim=-1)  # [B, N, K_prev, K_cur]

        prev_to_j_min = pair_dist.amin(dim=(2, 3))
        prev_to_j_mean = pair_dist.mean(dim=(2, 3))

        min_prev_to_j = torch.where(prev_is_depot, depot_to_j_min, prev_to_j_min)
        mean_prev_to_j = torch.where(prev_is_depot, depot_to_j_mean, prev_to_j_mean)

        # Center distance from previous field/depot to candidate field.
        prev_center_field = field_center.gather(
            1,
            prev_idx[:, :, None].expand(B, 1, 2),
        ).squeeze(1)  # [B, 2]
        prev_center = torch.where(prev_is_depot[:, :, None], depot[:, None, :], prev_center_field[:, None, :]).squeeze(1)
        center_distance_prev_j = (field_center - prev_center[:, None, :]).norm(p=2, dim=-1)

        remaining_fraction = (~state.visited_fields.squeeze(1)).float().sum(dim=1, keepdim=True) / float(N)
        remaining_fraction = remaining_fraction.expand(B, N)

        features = torch.stack(
            (
                min_prev_to_j,
                mean_prev_to_j,
                min_j_to_depot,
                mean_j_to_depot,
                center_distance_prev_j,
                min_cov,
                mean_cov,
                max_cov,
                remaining_fraction,
            ),
            dim=-1,
        )  # [B, N, 9]

        return self.pair_bias_mlp(features).squeeze(-1)[:, None, :]

    def _solve_templates_dp(self, input, field_order):
        """
        Exact batched DP over templates for a fixed field order.

        input['templates']:
            [B, N, K, 5]
        field_order:
            [B, N], values 0..N-1

        returns:
            best_cost:  [B]
            pi_actions: [B, N], values 1..N*K
        """
        templates = input["templates"]
        depot = input["depot"]

        B, N, K, F = templates.size()
        assert F == self.template_feature_dim

        ordered_templates = templates.gather(
            1,
            field_order[:, :, None, None].expand(B, N, K, F),
        )  # [B, N, K, 5]

        entry = ordered_templates[..., 0:2]      # [B, N, K, 2]
        exit_ = ordered_templates[..., 2:4]      # [B, N, K, 2]
        coverage = ordered_templates[..., 4]     # [B, N, K]

        # Start: depot -> first field template entry + coverage.
        dp = (entry[:, 0] - depot[:, None, :]).norm(p=2, dim=-1) + coverage[:, 0]  # [B, K]
        parents = []

        for t in range(1, N):
            transition = (
                exit_[:, t - 1, :, None, :] - entry[:, t, None, :, :]
            ).norm(p=2, dim=-1)  # [B, K_prev, K_cur]

            candidate_cost = dp[:, :, None] + transition  # [B, K_prev, K_cur]
            best_prev_cost, best_prev_template = candidate_cost.min(dim=1)  # [B, K_cur]

            dp = best_prev_cost + coverage[:, t]
            parents.append(best_prev_template)

        final_cost_by_template = dp + (exit_[:, -1] - depot[:, None, :]).norm(p=2, dim=-1)
        best_cost, last_template = final_cost_by_template.min(dim=1)  # [B]

        template_order = torch.zeros(B, N, dtype=torch.long, device=templates.device)
        template_order[:, -1] = last_template

        current_template = last_template
        for t in range(N - 1, 0, -1):
            parent_t = parents[t - 1]  # [B, K_cur]
            previous_template = parent_t.gather(1, current_template[:, None]).squeeze(1)
            template_order[:, t - 1] = previous_template
            current_template = previous_template

        pi_actions = 1 + field_order * K + template_order

        return best_cost, pi_actions

    def _gather_field_embedding(self, field_embeddings, field_index):
        """
        field_embeddings:
            [B, N+1, D]
        field_index:
            [B, 1], 0 = depot, 1..N = field
        returns:
            [B, 1, D]
        """
        B = field_embeddings.size(0)
        return field_embeddings.gather(
            1,
            field_index[:, :, None].expand(B, 1, self.embedding_dim),
        )

    def _masked_mean_or_depot(self, encoded_fields, mask, depot_embedding):
        """
        encoded_fields:
            [B, N, D]
        mask:
            [B, N], True = include field in mean
        depot_embedding:
            [B, D]

        Returns:
            [B, D]
        """
        valid = mask.float()
        count = valid.sum(dim=1, keepdim=True)
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


