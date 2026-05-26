import torch
from typing import NamedTuple


class StateTSP(NamedTuple):
    """
    Order-only decoding state.

    This state is intentionally compatible with the rest of the project:
        - visited_fields has shape [B, 1, N]
        - prev_field uses 0 for depot and 1..N for fields
        - get_mask() still returns a flat action mask for compatibility

    The model itself chooses only fields. Templates are selected afterwards by DP.
    """

    # Fixed input
    n_fields: int
    n_templates: int
    depot_coord: torch.Tensor       # [B, 1, 2]

    # Optional compatibility tensors. They are not used by the order-only decoder.
    node_in: torch.Tensor           # [B, 1 + N*K, 2]
    node_out: torch.Tensor          # [B, 1 + N*K, 2]

    # Dynamic state
    prev_a: torch.Tensor            # [B, 1], compatibility only
    prev_field: torch.Tensor        # [B, 1], 0 = depot, 1..N = previous selected field
    first_field: torch.Tensor       # [B, 1], 0 before first step, otherwise 1..N
    visited_fields: torch.Tensor    # [B, 1, N], True = field already selected
    i: torch.Tensor                 # scalar step counter

    # Compatibility fields with the older StateTSP interface.
    lengths: torch.Tensor           # [B, 1], not used for final cost in this model
    cur_coord: torch.Tensor         # [B, 1, 2], not used by order-only decoder

    @property
    def visited(self):
        return self.visited_fields

    @staticmethod
    def initialize(input, node_in=None, node_out=None):
        templates = input["templates"]
        depot = input["depot"]

        batch_size, n_fields, n_templates, _ = templates.size()
        depot_coord = depot[:, None, :]

        if node_in is None or node_out is None:
            candidate_in = templates[..., 0:2].reshape(batch_size, n_fields * n_templates, 2)
            candidate_out = templates[..., 2:4].reshape(batch_size, n_fields * n_templates, 2)
            node_in = torch.cat((depot_coord, candidate_in), dim=1)
            node_out = torch.cat((depot_coord, candidate_out), dim=1)

        return StateTSP(
            n_fields=n_fields,
            n_templates=n_templates,
            depot_coord=depot_coord,
            node_in=node_in,
            node_out=node_out,
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=templates.device),
            prev_field=torch.zeros(batch_size, 1, dtype=torch.long, device=templates.device),
            first_field=torch.zeros(batch_size, 1, dtype=torch.long, device=templates.device),
            visited_fields=torch.zeros(batch_size, 1, n_fields, dtype=torch.bool, device=templates.device),
            i=torch.zeros(1, dtype=torch.int64, device=templates.device),
            lengths=torch.zeros(batch_size, 1, device=templates.device),
            cur_coord=depot_coord,
        )

    def update_field(self, selected_field):
        """
        selected_field:
            [B], values in 0..N-1
        """
        selected_field = selected_field[:, None]  # [B, 1]
        selected_field_for_embeddings = selected_field + 1  # 1..N

        visited_fields = self.visited_fields.scatter(
            -1,
            selected_field[:, :, None],
            True,
        )

        if int(self.i.item()) == 0:
            first_field = selected_field_for_embeddings
        else:
            first_field = self.first_field

        return self._replace(
            prev_a=selected_field_for_embeddings,
            prev_field=selected_field_for_embeddings,
            first_field=first_field,
            visited_fields=visited_fields,
            i=self.i + 1,
        )

    def update(self, selected_action):
        """
        Compatibility method for code that still calls state.update(action).
        Converts flat action 1..N*K to field id and updates only field state.
        """
        candidate_id = selected_action[:, None] - 1
        field_id = (candidate_id // self.n_templates).squeeze(1)
        return self.update_field(field_id)

    def all_finished(self):
        return self.visited_fields.all()

    def get_current_node(self):
        return self.prev_a

    def get_current_field(self):
        return self.prev_field

    def get_area_mask(self):
        """
        Returns mask over fields:
            [B, 1, N]
        True means forbidden.
        """
        return self.visited_fields

    def get_mask(self):
        """
        Compatibility method with the old flat decoder.

        Returns mask over actions:
            [B, 1, 1 + N*K]
        """
        B = self.visited_fields.size(0)
        N = self.n_fields
        K = self.n_templates

        mask_depot = torch.ones(B, 1, 1, dtype=torch.bool, device=self.visited_fields.device)
        mask_candidates = self.visited_fields[:, :, :, None].expand(B, 1, N, K)
        mask_candidates = mask_candidates.reshape(B, 1, N * K)

        return torch.cat((mask_depot, mask_candidates), dim=-1)

    def get_final_cost(self):
        raise RuntimeError(
            "Order-only StateTSP does not compute final cost. "
            "Use TSP.get_costs(dataset, pi) or AttentionModel._solve_templates_dp()."
        )