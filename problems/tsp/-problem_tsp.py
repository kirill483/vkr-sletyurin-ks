from torch.utils.data import Dataset
import torch
import os
import pickle
from LCOPYL.problems.tsp.BESTstate_tsp import StateTSP


def rectangles_intersect(rect_a, rect_b):
    """
    rect = [x0, y0, x1, y1]
    Returns True if rectangles intersect.
    """
    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b

    no_intersection = ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1

    return not no_intersection


def point_inside_rect(point, rect):
    """
    point = [x, y]
    rect = [x0, y0, x1, y1]
    """
    x, y = point
    x0, y0, x1, y1 = rect

    return x0 <= x <= x1 and y0 <= y <= y1


def expand_rect(rect, margin):
    """
    Expands rectangle by margin in all directions.
    """
    x0, y0, x1, y1 = rect

    return torch.tensor(
        [x0 - margin, y0 - margin, x1 + margin, y1 + margin], dtype=torch.float32
    )


def make_templates_for_rect(x0, y0, w, h, nx, ny, a, length_mode="zero"):
    """
    Creates 8 templates for one rectangle.

    If length_mode == "zero":
        template = [entry_x, entry_y, exit_x, exit_y]

    If length_mode == "area":
        template = [entry_x, entry_y, exit_x, exit_y, length]
    """

    # 8 perimeter ports
    p1 = torch.tensor([x0 + a / 2, y0], dtype=torch.float32)
    p2 = torch.tensor([x0, y0 + a / 2], dtype=torch.float32)
    p3 = torch.tensor([x0, y0 + h - a / 2], dtype=torch.float32)
    p4 = torch.tensor([x0 + a / 2, y0 + h], dtype=torch.float32)

    p5 = torch.tensor([x0 + w - a / 2, y0 + h], dtype=torch.float32)
    p6 = torch.tensor([x0 + w, y0 + h - a / 2], dtype=torch.float32)
    p7 = torch.tensor([x0 + w, y0 + a / 2], dtype=torch.float32)
    p8 = torch.tensor([x0 + w - a / 2, y0], dtype=torch.float32)

    if length_mode == "zero":
        length = None
    elif length_mode == "area":
        length = float(w * h / a)
    else:
        raise ValueError(f"Unknown length_mode: {length_mode}")

    templates = []

    def add_template(entry, exit_):
        if length is None:
            template = torch.tensor(
                [
                    entry[0],
                    entry[1],
                    exit_[0],
                    exit_[1],
                ],
                dtype=torch.float32,
            )
        else:
            template = torch.tensor(
                [entry[0], entry[1], exit_[0], exit_[1], length], dtype=torch.float32
            )

        templates.append(template)

    # First 4 templates depend on nx parity
    if nx % 2 == 0:
        add_template(p1, p8)
        add_template(p4, p5)
        add_template(p5, p4)
        add_template(p8, p1)
    else:
        add_template(p1, p5)
        add_template(p4, p8)
        add_template(p5, p1)
        add_template(p8, p4)

    # Second 4 templates depend on ny parity
    if ny % 2 == 0:
        add_template(p2, p3)
        add_template(p3, p2)
        add_template(p6, p7)
        add_template(p7, p6)
    else:
        add_template(p2, p6)
        add_template(p3, p7)
        add_template(p6, p2)
        add_template(p7, p3)

    return torch.stack(templates, dim=0)  # [8, 4] or [8, 5]


def generate_depot_not_inside_rects(rects, max_attempts=10_000):
    """
    Generates depot point that is not inside any rectangle.
    """
    for _ in range(max_attempts):
        depot = torch.rand(2)

        inside_any = False
        for rect in rects:
            if point_inside_rect(depot, rect):
                inside_any = True
                break

        if not inside_any:
            return depot

    raise RuntimeError(
        f"Could not generate depot outside rectangles after {max_attempts} attempts."
    )


def generate_sample(
    n_rects,
    a=0.02,
    min_cells=3,
    max_cells=9,
    margin=None,
    max_attempts=10_000,
    length_mode="zero",
):
    """
    Generates one sample.

    Returns:
        {
            "templates": Tensor [N, 8, 4] if length_mode == "zero"
                         Tensor [N, 8, 5] otherwise,
            "depot": Tensor [2]
        }
    """

    if margin is None:
        margin = a

    rects = []
    templates = []

    attempts = 0

    while len(rects) < n_rects and attempts < max_attempts:
        attempts += 1

        # 1. Random rectangle size in cells
        nx = int(torch.randint(min_cells, max_cells + 1, (1,)).item())
        ny = int(torch.randint(min_cells, max_cells + 1, (1,)).item())

        w = nx * a
        h = ny * a

        if w >= 1.0 or h >= 1.0:
            continue

        # 2. Random bottom-left corner inside [0, 1]^2
        x0 = torch.rand(1).item() * (1.0 - w)
        y0 = torch.rand(1).item() * (1.0 - h)

        x1 = x0 + w
        y1 = y0 + h

        candidate = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)

        # 3. Check margin intersection
        expanded = expand_rect(candidate, margin)

        intersects = False
        for old_rect in rects:
            if rectangles_intersect(expanded, old_rect):
                intersects = True
                break

        if intersects:
            continue

        # 4. Accept rectangle
        rects.append(candidate)

        # 5. Create 8 templates
        rect_templates = make_templates_for_rect(
            x0=x0,
            y0=y0,
            w=w,
            h=h,
            nx=nx,
            ny=ny,
            a=a,
            length_mode=length_mode,
        )

        templates.append(rect_templates)

    if len(rects) < n_rects:
        raise RuntimeError(
            f"Could not generate {n_rects} non-overlapping rectangles "
            f"after {max_attempts} attempts. Try smaller a/margin or fewer rectangles."
        )

    depot = generate_depot_not_inside_rects(rects)

    return {
        "templates": torch.stack(templates, dim=0),
        "depot": depot,
    }





class TSP(object):
    NAME = "tsp"

    @staticmethod
    def get_costs(dataset, pi):
        """
        dataset:
            {
                'templates': [B, N, K, 4],
                'depot':     [B, 2]
            }

        templates[..., 0:2] = entry point
        templates[..., 2:4] = exit point

        pi:
            [B, N]
            values are in 1..N*K
        """
        templates = dataset["templates"]
        depot = dataset["depot"]

        batch_size, n_fields, n_templates, _ = templates.size()

        selected = pi - 1
        field_ids = selected // n_templates
        template_ids = selected % n_templates

        assert (
            torch.arange(
                n_fields,
                device=pi.device,
                dtype=field_ids.dtype,
            ).view(1, -1).expand_as(field_ids)
            == field_ids.sort(1)[0]
        ).all(), "Invalid tour: each field must be selected exactly once"

        batch_idx = torch.arange(
            batch_size,
            device=pi.device,
        )[:, None].expand(batch_size, n_fields)

        chosen_templates = templates[
            batch_idx,
            field_ids,
            template_ids,
        ]  # [B, N, 4]

        chosen_in = chosen_templates[..., 0:2]
        chosen_out = chosen_templates[..., 2:4]

        depot_to_first = (chosen_in[:, 0] - depot).norm(p=2, dim=1)
        between_templates = (chosen_in[:, 1:] - chosen_out[:, :-1]).norm(p=2, dim=2).sum(1)
        last_to_depot = (chosen_out[:, -1] - depot).norm(p=2, dim=1)

        return depot_to_first + between_templates + last_to_depot, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(input, node_in=None, node_out=None):
        return StateTSP.initialize(
            input,
            node_in=node_in,
            node_out=node_out,
        )


class TSPDataset(Dataset):

    def __init__(
        self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None
    ):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)

                self.data = [
                    {
                        "depot": torch.FloatTensor(depot),
                        "templates": torch.FloatTensor(templates),
                    }
                    for depot, templates in data[offset : offset + num_samples]
                ]

        else:
            self.data = [
                generate_sample(
                    n_rects=size,
                    a=0.02,
                    min_cells=3,
                    max_cells=9,
                    margin=0.02,
                    length_mode="zero",
                )
                for _ in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
