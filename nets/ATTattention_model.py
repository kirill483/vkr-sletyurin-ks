import math
from typing import NamedTuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nets.graph_encoder import GraphAttentionEncoder


class PointerFixed(NamedTuple):
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor





class AttentionModel(nn.Module):
    """
    Hierarchical Attention Model for GTSP-like coverage routing.

    New architecture:
        1. Template -> Linear embedding.
        2. Field embedding = mean over 8 template embeddings.
        3. Standard encoder over depot + fields.
        4. Area decoder selects next field.
        5. Selected field is expanded into K templates.
        6. Every selected template attends to future unvisited fields;
           if no future fields remain, it attends to depot.
        7. Template decoder selects one of K enriched templates.

    Public interface is kept compatible with the previous implementation:
        forward(input, return_pi=False) -> cost, ll, optionally pi

    input:
        {
            'depot':     [B, 2],
            'templates': [B, N, 8, 4] or [B, N, 8, 5]
        }

    If templates have 4 features, coverage length is treated as zero.
    If templates have 5 features, feature 4 is coverage length.
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
        self.template_feature_dim = 5

        assert embedding_dim % n_heads == 0

        # Depot: [x, y] -> D
        self.init_embed_depot = nn.Linear(2, embedding_dim)

        # Template: [entry_x, entry_y, exit_x, exit_y, coverage_len] -> D
        # This is intentionally simple: Linear, no local self-attention and no learned pooling.
        self.init_embed_template = nn.Linear(self.template_feature_dim, embedding_dim)

        # Standard encoder over [depot, field_1, ..., field_N].
        self.field_encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
        )

        # ---------------- Area choice ----------------
        # Context: mean/unvisited graph context + Linear(current field embedding, current point).
        self.project_area_current = nn.Linear(embedding_dim + 2, embedding_dim, bias=False)
        self.project_area_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_area_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # ---------------- Template enrichment ----------------
        # Seed per selected template: Linear(selected field embedding, candidate exit point) + future graph context.
        self.project_template_seed = nn.Linear(embedding_dim + 2, embedding_dim, bias=False)

        # Separate K/V set for template-to-future-fields attention.
        self.project_template_enrich_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_template_enrich_key_value = nn.Linear(embedding_dim, 2 * embedding_dim, bias=False)
        self.project_template_enrich_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Final template embedding before template pointer:
        # Linear(enriched template embedding, entry point, coverage length).
        self.project_template_final = nn.Linear(embedding_dim + 2 + 1, embedding_dim, bias=False)

        # ---------------- Template choice ----------------
        # Context: mean enriched templates + Linear(current field embedding, current point).
        self.project_template_current = nn.Linear(embedding_dim + 2, embedding_dim, bias=False)
        self.project_template_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_template_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        returns:
            cost: [B]
            ll:   [B]
            pi:   [B, N], optional, values in 1..N*8
        """
        templates5 = self._ensure_templates5(input["templates"])

        if self.checkpoint_encoder and self.training:
            field_embeddings, template_init_embeddings = checkpoint(
                self._encode_input,
                input["depot"],
                templates5,
                use_reentrant=False,
            )
        else:
            field_embeddings, template_init_embeddings = self._encode_input(input["depot"], templates5)

        node_in, node_out = self._make_node_in_out(input)

        log_likelihood, pi = self._inner(
            input=input,
            templates5=templates5,
            field_embeddings=field_embeddings,
            template_init_embeddings=template_init_embeddings,
            node_in=node_in,
            node_out=node_out,
        )

        cost, mask = self.problem.get_costs(input, pi)

        if mask is not None:
            log_likelihood = log_likelihood.masked_fill(mask.any(dim=1), 0)

        if return_pi:
            return cost, log_likelihood, pi
        return cost, log_likelihood

    def _ensure_templates5(self, templates):
        """
        Accepts [B, N, K, 4] or [B, N, K, 5].
        For 4 features, appends zero coverage length.
        """
        B, N, K, F = templates.size()
        assert K == self.n_templates, f"Expected K={self.n_templates}, got K={K}"
        assert F in (4, 5), f"Expected template feature dim 4 or 5, got {F}"

        if F == 5:
            return templates

        zero_len = torch.zeros(B, N, K, 1, dtype=templates.dtype, device=templates.device)
        return torch.cat((templates, zero_len), dim=-1)

    def _encode_input(self, depot, templates5):
        """
        depot:     [B, 2]
        templates: [B, N, K, 5]

        returns:
            field_embeddings:        [B, N + 1, D]
            template_init_embeddings:[B, N, K, D]
        """
        B, N, K, F = templates5.size()
        assert K == self.n_templates
        assert F == self.template_feature_dim

        # 1) Template -> Linear embedding.
        template_init_embeddings = self.init_embed_template(templates5)  # [B, N, K, D]

        # 2) Field embedding = average over K template embeddings.
        field_init = template_init_embeddings.mean(dim=2)  # [B, N, D]

        depot_embedding = self.init_embed_depot(depot)[:, None, :]  # [B, 1, D]

        # 3) Standard encoder over depot + fields.
        encoder_input = torch.cat((depot_embedding, field_init), dim=1)
        field_embeddings, _ = self.field_encoder(encoder_input)

        return field_embeddings, template_init_embeddings

    def _make_node_in_out(self, input):
        templates = input["templates"]
        depot = input["depot"]

        B, N, K, F = templates.size()
        assert K == self.n_templates
        assert F in (4, 5)

        candidate_in = templates[..., 0:2].reshape(B, N * K, 2)
        candidate_out = templates[..., 2:4].reshape(B, N * K, 2)

        depot_coord = depot[:, None, :]
        node_in = torch.cat((depot_coord, candidate_in), dim=1)
        node_out = torch.cat((depot_coord, candidate_out), dim=1)

        return node_in, node_out

    def _inner(self, input, templates5, field_embeddings, template_init_embeddings, node_in, node_out):
        sequences = []

        B, N, K, _ = templates5.size()

        state = self.problem.make_state(input, node_in=node_in, node_out=node_out)

        encoded_fields = field_embeddings[:, 1:, :]      # [B, N, D]
        depot_embedding = field_embeddings[:, 0, :]      # [B, D]

        area_fixed = self._precompute_pointer(encoded_fields, self.project_area_node_embeddings)

        log_likelihood = torch.zeros(B, device=templates5.device)

        while not state.all_finished():
            # -------- Level 1: choose field / area --------
            remaining_context = self._masked_mean_or_depot(
                encoded_fields=encoded_fields,
                visited_fields=state.visited_fields,
                depot_embedding=depot_embedding,
            )  # [B, D]

            area_log_p, area_mask = self._get_area_log_p(
                field_embeddings=field_embeddings,
                area_fixed=area_fixed,
                state=state,
                remaining_context=remaining_context,
            )

            selected_field = self._select_node(
                area_log_p.exp()[:, 0, :],
                area_mask[:, 0, :],
            )  # [B], 0..N-1

            selected_area_log_p = area_log_p[:, 0, :].gather(1, selected_field[:, None]).squeeze(1)

            # Future context excludes already visited fields and the selected field.
            future_visited = state.visited_fields.scatter(-1, selected_field[:, None, None], True)
            future_context = self._masked_mean_or_depot(
                encoded_fields=encoded_fields,
                visited_fields=future_visited,
                depot_embedding=depot_embedding,
            )  # [B, D]

            # -------- Level 2: choose template inside selected field --------
            template_log_p, template_mask = self._get_template_log_p(
                templates5=templates5,
                field_embeddings=field_embeddings,
                template_init_embeddings=template_init_embeddings,
                encoded_fields=encoded_fields,
                depot_embedding=depot_embedding,
                state=state,
                selected_field=selected_field,
                future_visited=future_visited,
                future_context=future_context,
            )

            selected_template = self._select_node(
                template_log_p.exp()[:, 0, :],
                template_mask[:, 0, :],
            )  # [B], 0..K-1

            selected_template_log_p = template_log_p[:, 0, :].gather(1, selected_template[:, None]).squeeze(1)

            selected_action = 1 + selected_field * K + selected_template  # [B], 1..N*K

            log_likelihood = log_likelihood + selected_area_log_p + selected_template_log_p
            state = state.update(selected_action)

            sequences.append(selected_action)

        pi = torch.stack(sequences, dim=1)  # [B, N]
        return log_likelihood, pi

    def _get_area_log_p(self, field_embeddings, area_fixed, state, remaining_context, normalize=True):
        prev_field_embedding = self._get_prev_field_embedding(field_embeddings, state)  # [B, 1, D]

        current_input = torch.cat((prev_field_embedding, state.cur_coord), dim=-1)  # [B, 1, D+2]
        current_context = self.project_area_current(current_input)                  # [B, 1, D]

        query = remaining_context[:, None, :] + current_context                     # [B, 1, D]

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

        if normalize:
            log_p = torch.log_softmax(logits / self.temp, dim=-1)
        else:
            log_p = logits

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _get_template_log_p(
        self,
        templates5,
        field_embeddings,
        template_init_embeddings,
        encoded_fields,
        depot_embedding,
        state,
        selected_field,
        future_visited,
        future_context,
        normalize=True,
    ):
        B, N, K, _ = templates5.size()

        selected_field_embedding = field_embeddings[:, 1:, :].gather(
            1,
            selected_field[:, None, None].expand(B, 1, self.embedding_dim),
        )  # [B, 1, D]

        selected_templates = templates5.gather(
            1,
            selected_field[:, None, None, None].expand(B, 1, K, self.template_feature_dim),
        ).squeeze(1)  # [B, K, 5]

        entry = selected_templates[..., 0:2]       # [B, K, 2]
        exit_ = selected_templates[..., 2:4]       # [B, K, 2]
        coverage_len = selected_templates[..., 4:5]  # [B, K, 1]

        # Seed: Linear(selected area embedding, candidate exit point) + future graph context.
        selected_field_expanded = selected_field_embedding.expand(B, K, self.embedding_dim)
        seed_input = torch.cat((selected_field_expanded, exit_), dim=-1)  # [B, K, D+2]
        template_seed = self.project_template_seed(seed_input) + future_context[:, None, :]  # [B, K, D]

        # Every candidate template attends to remaining unvisited areas.
        # If no unvisited area remains for a batch element, the only available memory is depot.
        future_memory, future_memory_mask = self._build_future_memory(
            encoded_fields=encoded_fields,
            depot_embedding=depot_embedding,
            future_visited=future_visited,
        )  # memory: [B, N+1, D], mask: [B, 1, N+1]

        future_attended = self._template_to_future_attention(
            template_seed=template_seed,
            future_memory=future_memory,
            future_memory_mask=future_memory_mask,
        )  # [B, K, D]

        # Residual keeps template-specific seed information while adding future-aware information.
        enriched_templates = template_seed + future_attended  # [B, K, D]

        # Final per-template embeddings used by the template pointer.
        template_final_input = torch.cat((enriched_templates, entry, coverage_len), dim=-1)
        template_embeddings = self.project_template_final(template_final_input)  # [B, K, D]

        template_fixed = self._precompute_pointer(template_embeddings, self.project_template_node_embeddings)

        # Context for choosing one of 8 templates:
        # mean enriched templates + Linear(current field embedding, current physical point).
        prev_field_embedding = self._get_prev_field_embedding(field_embeddings, state)  # [B, 1, D]
        current_input = torch.cat((prev_field_embedding, state.cur_coord), dim=-1)
        current_context = self.project_template_current(current_input)  # [B, 1, D]
        template_mean = template_embeddings.mean(dim=1, keepdim=True)   # [B, 1, D]
        query = template_mean + current_context                         # [B, 1, D]

        template_mask = torch.zeros(B, 1, K, dtype=torch.bool, device=templates5.device)

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

    def _build_future_memory(self, encoded_fields, depot_embedding, future_visited):
        """
        Build memory for template enrichment.

        Normally memory is all still-unvisited fields.
        For a batch element with no future fields left, memory is depot only.

        returns:
            future_memory:      [B, N+1, D] = fields + depot
            future_memory_mask: [B, 1, N+1], True = forbidden
        """
        B, N, D = encoded_fields.size()

        future_memory = torch.cat((encoded_fields, depot_embedding[:, None, :]), dim=1)  # [B, N+1, D]

        unvisited_count = (~future_visited.squeeze(1)).sum(dim=-1)  # [B]

        # If there are future fields, depot is masked.
        # If there are no future fields, all fields are masked and depot is unmasked.
        field_mask = future_visited  # [B, 1, N]
        depot_mask = unvisited_count.gt(0).view(B, 1, 1)  # True while fields remain, False on last step

        future_memory_mask = torch.cat((field_mask, depot_mask), dim=-1)  # [B, 1, N+1]
        return future_memory, future_memory_mask

    def _template_to_future_attention(self, template_seed, future_memory, future_memory_mask):
        """
        Multi-head attention from K selected templates to future areas/depot.

        template_seed:       [B, K, D]
        future_memory:       [B, M, D]
        future_memory_mask:  [B, 1, M], True = forbidden

        returns:
            [B, K, D]
        """
        B, K, D = template_seed.size()
        M = future_memory.size(1)
        head_dim = D // self.n_heads

        q = self.project_template_enrich_query(template_seed)  # [B, K, D]
        k, v = self.project_template_enrich_key_value(future_memory).chunk(2, dim=-1)  # [B, M, D] each

        q = q.view(B, K, self.n_heads, head_dim).permute(2, 0, 1, 3)  # [H, B, K, Hd]
        k = k.view(B, M, self.n_heads, head_dim).permute(2, 0, 1, 3)  # [H, B, M, Hd]
        v = v.view(B, M, self.n_heads, head_dim).permute(2, 0, 1, 3)  # [H, B, M, Hd]

        compatibility = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [H, B, K, M]

        mask = future_memory_mask[:, :, None, :].expand(B, 1, K, M).permute(1, 0, 2, 3)  # [1, B, K, M]
        compatibility = compatibility.masked_fill(mask, -math.inf)

        attn = torch.softmax(compatibility, dim=-1)
        heads = torch.matmul(attn, v)  # [H, B, K, Hd]

        out = heads.permute(1, 2, 0, 3).contiguous().view(B, K, D)  # [B, K, D]
        return self.project_template_enrich_out(out)

    def _get_prev_field_embedding(self, field_embeddings, state):
        B = field_embeddings.size(0)
        return field_embeddings.gather(
            1,
            state.prev_field[:, :, None].expand(B, 1, self.embedding_dim),
        )

    def _masked_mean_or_depot(self, encoded_fields, visited_fields, depot_embedding):
        """
        encoded_fields:  [B, N, D]
        visited_fields:  [B, 1, N]
        depot_embedding: [B, D]

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
        embeddings: [B, M, D]
        """
        glimpse_key, glimpse_val, logit_key = projection_layer(embeddings[:, None, :, :]).chunk(3, dim=-1)

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

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility = compatibility.masked_fill(
                mask[None, :, :, None, :].expand_as(compatibility),
                -math.inf,
            )

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
            logits = logits.masked_fill(mask, -math.inf)

        return logits, glimpse.squeeze(-2)


