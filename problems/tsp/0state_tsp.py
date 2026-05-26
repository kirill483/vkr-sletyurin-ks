import torch
from typing import NamedTuple

class StateTSP(NamedTuple):
    # Fixed input
    node_in: torch.Tensor        # [B, 1 + N*8, 2]
    node_out: torch.Tensor       # [B, 1 + N*8, 2]
    n_fields: int
    n_templates: int
    depot_coord: torch.Tensor  # [B, 1, 2]
    # State
    prev_a: torch.Tensor         # [B, 1], последний выбранный candidate/action
    prev_field: torch.Tensor     # [B, 1], 0 = depot, 1..N = fields
    visited_fields: torch.Tensor # [B, 1, N], True = поле уже покрыто
    lengths: torch.Tensor        # [B, 1], накопленная длина
    cur_coord: torch.Tensor      # [B, 1, 2], текущая точка выхода
    i: torch.Tensor              # шаг decoding

    @property
    def visited(self):
        return self.visited_fields

    @staticmethod
    def initialize(input, node_in, node_out):
        """
        input:
            {
                'depot':     [B, 2],
                'templates': [B, N, 8, 4]
            }

        node_in / node_out:
            [B, 1 + N*8, 2]

        Индексация action:
            0       = depot
            1..N*8  = candidate-шаблоны

        Индексация field:
            0    = depot
            1..N = fields
        """

        depot = input['depot']
        templates = input['templates']

        batch_size, n_fields, n_templates, _ = templates.size()

        return StateTSP(
            node_in=node_in,
            node_out=node_out,
            n_fields=n_fields,
            n_templates=n_templates,

            # current action = depot
            prev_a=torch.zeros(
                batch_size,
                1,
                dtype=torch.long,
                device=templates.device
            ),

            # current field = depot
            prev_field=torch.zeros(
                batch_size,
                1,
                dtype=torch.long,
                device=templates.device
            ),

            visited_fields=torch.zeros(
                batch_size,
                1,
                n_fields,
                dtype=torch.bool,
                device=templates.device
            ),

            lengths=torch.zeros(
                batch_size,
                1,
                device=templates.device
            ),

            # На старте текущая физическая позиция = depot
            cur_coord=depot[:, None, :],

            i=torch.zeros(
                1,
                dtype=torch.int64,
                device=templates.device
            )
        )

    def update(self, selected):
        """
        selected:
            [B]
            0       = depot, должен быть masked
            1..N*8  = выбранный candidate-шаблон
        """

        selected = selected[:, None]  # [B, 1]

        # input-точка выбранного шаблона
        selected_in = self.node_in.gather(
            1,
            selected[:, :, None].expand(
                selected.size(0),
                1,
                2
            )
        )

        # output-точка выбранного шаблона
        selected_out = self.node_out.gather(
            1,
            selected[:, :, None].expand(
                selected.size(0),
                1,
                2
            )
        )

        # Добавляем переход:
        # previous_exit -> selected_template_input
        lengths = self.lengths + (
            selected_in - self.cur_coord
        ).norm(p=2, dim=-1)

        # selected: 1..N*8
        # candidate_id: 0..N*8-1
        candidate_id = selected - 1

        # field_id: 0..N-1
        field_id = candidate_id // self.n_templates

        # field index for field_embeddings:
        # 0    = depot
        # 1..N = fields
        prev_field = field_id + 1

        visited_fields = self.visited_fields.scatter(
            -1,
            field_id[:, :, None],
            True
        )

        return self._replace(
            prev_a=selected,
            prev_field=prev_field,
            visited_fields=visited_fields,
            lengths=lengths,
            cur_coord=selected_out,
            i=self.i + 1
        )

    def all_finished(self):
        # Все поля покрыты
        return self.visited_fields.all()

    def get_current_node(self):
        # Последний выбранный candidate/action
        return self.prev_a

    def get_current_field(self):
        # Последнее выбранное поле.
        # Именно это используется в context.
        return self.prev_field

    def get_mask(self):
        """
        Возвращает mask размера [B, 1, 1 + N*8]

        True  = action запрещён
        False = action разрешён

        Правила:
            depot всегда запрещён как action
            если поле уже посещено, запрещаем все 8 его шаблонов
        """

        B = self.visited_fields.size(0)
        N = self.n_fields
        K = self.n_templates

        # depot нельзя выбирать
        mask_depot = torch.ones(
            B,
            1,
            1,
            dtype=torch.bool,
            device=self.visited_fields.device
        )

        # [B, 1, N] -> [B, 1, N, K]
        mask_candidates = self.visited_fields[:, :, :, None].expand(
            B,
            1,
            N,
            K
        )

        # [B, 1, N, K] -> [B, 1, N*K]
        mask_candidates = mask_candidates.reshape(
            B,
            1,
            N * K
        )

        return torch.cat(
            (
                mask_depot,
                mask_candidates
            ),
            dim=-1
        )

    def get_final_cost(self):
        """
        После последнего выбранного шаблона возвращаемся из его exit-точки в depot.
        """

        assert self.all_finished()

        depot_coord = self.node_in[:, 0:1, :]  # [B, 1, 2]

        return self.lengths + (
            depot_coord - self.cur_coord
        ).norm(p=2, dim=-1)