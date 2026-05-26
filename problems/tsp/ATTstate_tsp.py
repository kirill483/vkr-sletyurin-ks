import torch
from typing import NamedTuple




class StateTSP(NamedTuple):
    """
    State for hierarchical GTSP-like decoding.

    Actions are flat ids:
        0       = depot, only used as initial previous action
        1..N*K  = concrete template actions

    For an action a in 1..N*K:
        field_id    = (a - 1) // K, 0..N-1
        template_id = (a - 1) % K,  0..K-1
    """

    # Fixed input
    node_in: torch.Tensor          # [B, 1 + N*K, 2]
    node_out: torch.Tensor         # [B, 1 + N*K, 2]
    n_fields: int
    n_templates: int
    depot_coord: torch.Tensor      # [B, 1, 2]

    # Dynamic state
    prev_a: torch.Tensor           # [B, 1], 0 = depot, 1..N*K = selected template action
    prev_field: torch.Tensor       # [B, 1], 0 = depot, 1..N = encoded field index
    visited_fields: torch.Tensor   # [B, 1, N], True = field already covered
    lengths: torch.Tensor          # [B, 1], transition distance only; final objective is computed in TSP.get_costs
    cur_coord: torch.Tensor        # [B, 1, 2], current physical point, initially depot, then previous template exit
    i: torch.Tensor

    @property
    def visited(self):
        return self.visited_fields

    @staticmethod
    def initialize(input, node_in, node_out):
        templates = input["templates"]
        depot = input["depot"]

        batch_size, n_fields, n_templates, _ = templates.size()
        depot_coord = depot[:, None, :]

        return StateTSP(
            node_in=node_in,
            node_out=node_out,
            n_fields=n_fields,
            n_templates=n_templates,
            depot_coord=depot_coord,
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=templates.device),
            prev_field=torch.zeros(batch_size, 1, dtype=torch.long, device=templates.device),
            visited_fields=torch.zeros(batch_size, 1, n_fields, dtype=torch.bool, device=templates.device),
            lengths=torch.zeros(batch_size, 1, device=templates.device),
            cur_coord=depot_coord,
            i=torch.zeros(1, dtype=torch.int64, device=templates.device),
        )

    def update(self, selected_action):
        """
        selected_action: [B], values in 1..N*K.
        """
        selected_action = selected_action[:, None]  # [B, 1]

        selected_in = self.node_in.gather(
            1,
            selected_action[:, :, None].expand(selected_action.size(0), 1, 2),
        )
        selected_out = self.node_out.gather(
            1,
            selected_action[:, :, None].expand(selected_action.size(0), 1, 2),
        )

        lengths = self.lengths + (selected_in - self.cur_coord).norm(p=2, dim=-1)

        candidate_id = selected_action - 1
        field_id = candidate_id // self.n_templates       # [B, 1], 0..N-1
        prev_field = field_id + 1                         # [B, 1], 1..N in encoded [depot + fields]

        visited_fields = self.visited_fields.scatter(-1, field_id[:, :, None], True)

        return self._replace(
            prev_a=selected_action,
            prev_field=prev_field,
            visited_fields=visited_fields,
            lengths=lengths,
            cur_coord=selected_out,
            i=self.i + 1,
        )

    def all_finished(self):
        return self.visited_fields.all()

    def get_current_node(self):
        return self.prev_a

    def get_current_field(self):
        return self.prev_field

    def get_area_mask(self):
        """
        Mask over fields.

        Shape: [B, 1, N]
        True  = forbidden
        False = allowed
        """
        return self.visited_fields

    def get_mask(self):
        """
        Compatibility method for flat action decoders.

        Shape: [B, 1, 1 + N*K]
        Depot and all templates from visited fields are forbidden.
        """
        B = self.visited_fields.size(0)
        N = self.n_fields
        K = self.n_templates

        mask_depot = torch.ones(B, 1, 1, dtype=torch.bool, device=self.visited_fields.device)
        mask_candidates = self.visited_fields[:, :, :, None].expand(B, 1, N, K).reshape(B, 1, N * K)
        return torch.cat((mask_depot, mask_candidates), dim=-1)

    def get_final_cost(self):
        assert self.all_finished()
        return self.lengths + (self.depot_coord - self.cur_coord).norm(p=2, dim=-1)