import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch


class Config:
    data_dir = ""
    filename = ""
    dataset_size = 1000
    graph_size = 20
    seed = 1234
    overwrite = True
    name = "val"

def rectangles_intersect_np(a, b):
    """
    a: [4]
    b: [4]
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    return not (
        ax1 <= bx0 or
        ax0 >= bx1 or
        ay1 <= by0 or
        ay0 >= by1
    )


def intersects_any_np(rect, rects):
    """
    rect: [4]
    rects: [M, 4]
    """

    if len(rects) == 0:
        return False

    rects = np.asarray(rects, dtype=np.float32)

    ax0, ay0, ax1, ay1 = rect

    bx0 = rects[:, 0]
    by0 = rects[:, 1]
    bx1 = rects[:, 2]
    by1 = rects[:, 3]

    no_intersection = (
        (ax1 <= bx0) |
        (ax0 >= bx1) |
        (ay1 <= by0) |
        (ay0 >= by1)
    )

    return not np.all(no_intersection)


def point_inside_any_rect_np(points, rects):
    """
    points: [M, 2]
    rects:  [N, 4]

    return:
        inside_any: [M]
    """

    if len(rects) == 0:
        return np.zeros(points.shape[0], dtype=bool)

    rects = np.asarray(rects, dtype=np.float32)

    x = points[:, 0:1]
    y = points[:, 1:2]

    x0 = rects[None, :, 0]
    y0 = rects[None, :, 1]
    x1 = rects[None, :, 2]
    y1 = rects[None, :, 3]

    inside = (
        (x >= x0) &
        (x <= x1) &
        (y >= y0) &
        (y <= y1)
    )

    return inside.any(axis=1)


def generate_depot_not_inside_rects_fast(rects, rng, max_attempts=10_000, batch_attempts=1024):
    

    attempts = 0

    while attempts < max_attempts:
        points = rng.random((batch_attempts, 2), dtype=np.float32)
        inside = point_inside_any_rect_np(points, rects)

        valid = np.where(~inside)[0]

        if len(valid) > 0:
            return points[valid[0]]

        attempts += batch_attempts

    raise RuntimeError(
        f"Could not generate depot outside rectangles after {max_attempts} attempts."
    )


def make_templates_for_rect_fast(
    x0,
    y0,
    w,
    h,
    nx,
    ny,
    a,
    length_mode="zero",
    turn_penalty=None,
):
   

    if turn_penalty is None:
        turn_penalty = 0.5 * a

    half_a = a / 2.0

    p1 = np.array([x0 + half_a, y0], dtype=np.float32)
    p2 = np.array([x0, y0 + half_a], dtype=np.float32)
    p3 = np.array([x0, y0 + h - half_a], dtype=np.float32)
    p4 = np.array([x0 + half_a, y0 + h], dtype=np.float32)

    p5 = np.array([x0 + w - half_a, y0 + h], dtype=np.float32)
    p6 = np.array([x0 + w, y0 + h - half_a], dtype=np.float32)
    p7 = np.array([x0 + w, y0 + half_a], dtype=np.float32)
    p8 = np.array([x0 + w - half_a, y0], dtype=np.float32)

    pairs = []
    turn_counts = []

    
    turns_x = nx - 1

    if nx % 2 == 0:
        pairs.extend([
            (p1, p8),
            (p4, p5),
            (p5, p4),
            (p8, p1),
        ])
    else:
        pairs.extend([
            (p1, p5),
            (p4, p8),
            (p5, p1),
            (p8, p4),
        ])

    turn_counts.extend([turns_x] * 4)

    
    turns_y = ny - 1

    if ny % 2 == 0:
        pairs.extend([
            (p2, p3),
            (p3, p2),
            (p6, p7),
            (p7, p6),
        ])
    else:
        pairs.extend([
            (p2, p6),
            (p3, p7),
            (p6, p2),
            (p7, p3),
        ])

    turn_counts.extend([turns_y] * 4)

    if length_mode == "zero":
        templates = np.empty((8, 4), dtype=np.float32)

        for i, (entry, exit_) in enumerate(pairs):
            templates[i, 0:2] = entry
            templates[i, 2:4] = exit_

        return templates

    elif length_mode == "area":
        templates = np.empty((8, 5), dtype=np.float32)

        base_length = float(nx * ny * a)

        for i, (entry, exit_) in enumerate(pairs):
            templates[i, 0:2] = entry
            templates[i, 2:4] = exit_
            templates[i, 4] = base_length

        return templates

    elif length_mode == "coverage":
        templates = np.empty((8, 5), dtype=np.float32)

        base_length = float(nx * ny * a)

        for i, (entry, exit_) in enumerate(pairs):
            num_turns = turn_counts[i]
            coverage_cost = base_length + float(turn_penalty * num_turns)

            templates[i, 0:2] = entry
            templates[i, 2:4] = exit_
            templates[i, 4] = coverage_cost

        return templates

    else:
        raise ValueError(f"Unknown length_mode: {length_mode}")


def generate_sample_fast(
    n_rects,
    a=0.02,
    min_cells=3,
    max_cells=9,
    margin=None,
    max_attempts=10_000,
    length_mode="coverage",
    turn_penalty=None,
    rng=None,
):
   

    if rng is None:
        rng = np.random.default_rng()

    if margin is None:
        margin = a

    template_dim = 4 if length_mode == "zero" else 5

    rects = []
    templates = np.empty((n_rects, 8, template_dim), dtype=np.float32)

    attempts = 0
    count = 0

    while count < n_rects and attempts < max_attempts:
        attempts += 1

        nx = int(rng.integers(min_cells, max_cells + 1))
        ny = int(rng.integers(min_cells, max_cells + 1))

        w = nx * a
        h = ny * a

        if w >= 1.0 or h >= 1.0:
            continue

        x0 = float(rng.random() * (1.0 - w))
        y0 = float(rng.random() * (1.0 - h))

        x1 = x0 + w
        y1 = y0 + h

        candidate = np.array([x0, y0, x1, y1], dtype=np.float32)

        expanded = np.array(
            [
                x0 - margin,
                y0 - margin,
                x1 + margin,
                y1 + margin,
            ],
            dtype=np.float32
        )

        if intersects_any_np(expanded, rects):
            continue

        rects.append(candidate)

        templates[count] = make_templates_for_rect_fast(
            x0=x0,
            y0=y0,
            w=w,
            h=h,
            nx=nx,
            ny=ny,
            a=a,
            length_mode=length_mode,
            turn_penalty=turn_penalty,
        )

        count += 1

    if count < n_rects:
        raise RuntimeError(
            f"Could not generate {n_rects} non-overlapping rectangles "
            f"after {max_attempts} attempts. Try smaller a/margin or fewer rectangles."
        )

    depot = generate_depot_not_inside_rects_fast(rects, rng)

    return {
        "templates": templates,
        "depot": depot,
    }


import time
import numpy as np


def generate_tsp_data(
    dataset_size,
    tsp_size,
    a=0.02,
    min_cells=3,
    max_cells=9,
    margin=None,
    seed=None,
    log_every=1000,
    length_mode="coverage",
    turn_penalty=0.01,
):
    

    rng = np.random.default_rng(seed)

    data = []
    start_time = time.time()

    for i in range(dataset_size):
        sample = generate_sample_fast(
            n_rects=tsp_size,
            a=a,
            min_cells=min_cells,
            max_cells=max_cells,
            margin=margin if margin is not None else a,
            length_mode=length_mode,
            turn_penalty=turn_penalty,
            rng=rng,
        )

        data.append(
            (
                sample["depot"].tolist(),
                sample["templates"].tolist(),
            )
        )

        created = i + 1

        if log_every and (created % log_every == 0 or created == dataset_size):
            elapsed = time.time() - start_time
            speed = created / elapsed if elapsed > 0 else 0.0
            remaining = dataset_size - created
            eta = remaining / speed if speed > 0 else 0.0

            print(
                f"Generated {created}/{dataset_size} samples "
                f"({created / dataset_size * 100:.1f}%) | "
                f"elapsed={elapsed:.1f}s | "
                f"speed={speed:.1f} samples/s | "
                f"eta={eta:.1f}s",
                flush=True,
            )

    return data

if __name__ == "__main__":
    opts = Config()

    os.makedirs(opts.data_dir, exist_ok=True)

    if opts.filename is None:
        filename = os.path.join(
            opts.data_dir, f"tsp{opts.graph_size}_val_seed{opts.seed}.pkl"
        )
    else:
        filename = os.path.join(opts.data_dir, check_extension(opts.filename))

    assert opts.overwrite or not os.path.isfile(
        check_extension(filename)
    ), "File already exists!"

    np.random.seed(opts.seed)
    dataset = generate_tsp_data(opts.dataset_size, opts.graph_size)

    save_dataset(dataset, filename)
