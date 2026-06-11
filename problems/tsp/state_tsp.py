import torch
from typing import NamedTuple


class StateTSP(NamedTuple):
    """
    Order-only decoding state.

    The model chooses only fields. Templates are selected afterwards by DP.
    """

    n_fields: int

    # Dynamic state
    prev_field: torch.Tensor        # [B, 1], 0 = depot, 1..N = previous selected field
    visited_fields: torch.Tensor    # [B, 1, N], True = field already selected
    i: torch.Tensor                 # scalar step counter

    @staticmethod
    def initialize(input):
        templates = input["templates"]

        batch_size, n_fields, _, _ = templates.size()

        return StateTSP(
            n_fields=n_fields,
            prev_field=torch.zeros(
                batch_size,
                1,
                dtype=torch.long,
                device=templates.device,
            ),
            visited_fields=torch.zeros(
                batch_size,
                1,
                n_fields,
                dtype=torch.bool,
                device=templates.device,
            ),
            i=torch.zeros(
                1,
                dtype=torch.int64,
                device=templates.device,
            ),
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

        return self._replace(
            prev_field=selected_field_for_embeddings,
            visited_fields=visited_fields,
            i=self.i + 1,
        )

    def all_finished(self):
        return self.visited_fields.all()

    def get_area_mask(self):
        """
        Returns mask over fields:
            [B, 1, N]

        True means forbidden.
        """
        return self.visited_fields