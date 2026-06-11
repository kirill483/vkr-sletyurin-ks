import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle


MODEL_RESULTS = "results/model_200_20.pkl"
EXACT_RESULTS = "results/exact_200_20.pkl"

DATASET_PATH = "data/200_20.pkl"

INSTANCE_ID = 39
N_TEMPLATES = 8

A = 0.02


def load_results(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        data = data[0]

    costs = np.array([x[0] for x in data], dtype=np.float64)
    tours = [x[1] for x in data]

    return costs, tours


def unpack_sample(sample):
   

    if isinstance(sample, dict):
        depot = sample["depot"]
        templates = sample["templates"]
    else:
        depot, templates = sample

    depot = np.asarray(depot, dtype=np.float64)
    templates = np.asarray(templates, dtype=np.float64)

    return depot, templates


def coverage_constant(templates):
    
    if templates.shape[-1] < 5:
        return 0.0

    lengths = templates[:, :, 4]          # [N, 8]
    min_per_field = lengths.min(axis=1)   # [N]

    return float(min_per_field.sum())


def decode_tour(pi, n_templates=8):
    

    decoded = []

    for action_id in pi:
        action_id = int(action_id)

        candidate_id = action_id - 1
        field_id = candidate_id // n_templates
        template_id = candidate_id % n_templates

        decoded.append((field_id, template_id, action_id))

    return decoded


def get_field_rects_from_templates(templates):
    
    rects = []

    for field_id in range(templates.shape[0]):
        field_templates = templates[field_id]

        points = np.concatenate(
            (
                field_templates[:, 0:2],  # входы
                field_templates[:, 2:4],  # выходы
            ),
            axis=0,
        )

        x_min = points[:, 0].min()
        y_min = points[:, 1].min()
        x_max = points[:, 0].max()
        y_max = points[:, 1].max()

        rects.append((x_min, y_min, x_max, y_max))

    return rects


def draw_fields(ax, templates):
    

    rects = get_field_rects_from_templates(templates)

    for x_min, y_min, x_max, y_max in rects:
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            linewidth=1.4,
            alpha=0.75,
        )
        ax.add_patch(rect)


def compute_route_points(depot, templates, pi, n_templates=8):
   

    decoded = decode_tour(pi, n_templates)

    travel_segments = []
    template_segments = []

    current_point = depot

    for field_id, template_id, action_id in decoded:
        template = templates[field_id, template_id]

        in_point = template[0:2]
        out_point = template[2:4]

        travel_segments.append((current_point, in_point))
        template_segments.append((in_point, out_point))

        current_point = out_point

    travel_segments.append((current_point, depot))

    return travel_segments, template_segments


def route_cost_no_const(depot, templates, pi, n_templates=8):
   

    decoded = decode_tour(pi, n_templates)

    total_transition = 0.0
    total_coverage = 0.0

    current_point = depot

    has_length = templates.shape[-1] >= 5

    for field_id, template_id, action_id in decoded:
        template = templates[field_id, template_id]

        in_point = template[0:2]
        out_point = template[2:4]

        total_transition += float(np.linalg.norm(in_point - current_point))

        if has_length:
            total_coverage += float(template[4])

        current_point = out_point

    total_transition += float(np.linalg.norm(depot - current_point))

    const = coverage_constant(templates)

    return total_transition + total_coverage - const


def draw_arrow(ax, a, b, linewidth=1.2, alpha=0.9, linestyle="-"):
    

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if np.linalg.norm(b - a) < 1e-12:
        return

    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
        ),
    )


def make_snake_path(rect, in_point, out_point, a=0.02):
    

    x_min, y_min, x_max, y_max = rect

    in_point = np.asarray(in_point, dtype=np.float64)
    out_point = np.asarray(out_point, dtype=np.float64)

    dx = abs(out_point[0] - in_point[0])
    dy = abs(out_point[1] - in_point[1])

    path = []

    if dx >= dy:
        x_start = in_point[0]
        x_end = out_point[0]

        direction_x = 1 if x_end >= x_start else -1

        xs = np.arange(
            x_start,
            x_end + direction_x * a * 0.5,
            direction_x * a,
        )

        if len(xs) == 0:
            xs = np.array([x_start])

        xs = np.clip(xs, x_min, x_max)

        go_up = abs(in_point[1] - y_min) <= abs(in_point[1] - y_max)

        path.append(in_point)

        for x in xs:
            if go_up:
                p1 = np.array([x, y_min])
                p2 = np.array([x, y_max])
            else:
                p1 = np.array([x, y_max])
                p2 = np.array([x, y_min])

            if np.linalg.norm(path[-1] - p1) > 1e-12:
                path.append(p1)

            path.append(p2)
            go_up = not go_up

        if np.linalg.norm(path[-1] - out_point) > 1e-12:
            path.append(out_point)

    else:
        y_start = in_point[1]
        y_end = out_point[1]

        direction_y = 1 if y_end >= y_start else -1

        ys = np.arange(
            y_start,
            y_end + direction_y * a * 0.5,
            direction_y * a,
        )

        if len(ys) == 0:
            ys = np.array([y_start])

        ys = np.clip(ys, y_min, y_max)

        go_right = abs(in_point[0] - x_min) <= abs(in_point[0] - x_max)

        path.append(in_point)

        for y in ys:
            if go_right:
                p1 = np.array([x_min, y])
                p2 = np.array([x_max, y])
            else:
                p1 = np.array([x_max, y])
                p2 = np.array([x_min, y])

            if np.linalg.norm(path[-1] - p1) > 1e-12:
                path.append(p1)

            path.append(p2)
            go_right = not go_right

        if np.linalg.norm(path[-1] - out_point) > 1e-12:
            path.append(out_point)

    return np.asarray(path, dtype=np.float64)


def plot_route(ax, depot, templates, pi, title, saved_cost_no_const=None):
    

    travel_segments, _ = compute_route_points(depot, templates, pi, N_TEMPLATES)

    # Прямоугольники
    draw_fields(ax, templates)

    # Депо
    ax.scatter(
        [depot[0]],
        [depot[1]],
        marker="s",
        s=90,
        label="Депо",
        zorder=5,
    )

    for a, b in travel_segments:
        draw_arrow(
            ax,
            a,
            b,
            linewidth=1.1,
            alpha=0.9,
        )

    if saved_cost_no_const is None:
        title_text = title
    else:
        title_text = f"{title}\nстоимость  = {saved_cost_no_const:.6f}"

    ax.set_title(title_text)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def main():
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    depot, templates = unpack_sample(dataset[INSTANCE_ID])

    model_costs, model_tours = load_results(MODEL_RESULTS)
    exact_costs, exact_tours = load_results(EXACT_RESULTS)

    model_pi = model_tours[INSTANCE_ID]
    exact_pi = exact_tours[INSTANCE_ID]

    const = coverage_constant(templates)

    model_cost_no_const = route_cost_no_const(
        depot,
        templates,
        model_pi,
        N_TEMPLATES,
    )

    exact_cost_no_const = route_cost_no_const(
        depot,
        templates,
        exact_pi,
        N_TEMPLATES,
    )

    gap_no_const = (
        model_cost_no_const / exact_cost_no_const - 1.0
    ) * 100

    print(f"Экземпляр: {INSTANCE_ID}")
    print(f"Константа покрытия:              {const:.6f}")
    print(f"Model cost из файла:             {model_costs[INSTANCE_ID]:.6f}")
    print(f"Exact cost из файла:             {exact_costs[INSTANCE_ID]:.6f}")
    print(f"Модель :            {model_cost_no_const:.6f}")
    print(f"Точное решение :    {exact_cost_no_const:.6f}")
    print(f"Gap :               {gap_no_const:.3f}%")
    print(f"Маршрут модели:                  {model_pi}")
    print(f"Точный маршрут:                  {exact_pi}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_route(
        axes[0],
        depot,
        templates,
        model_pi,
        title="Маршрут модели",
        saved_cost_no_const=model_cost_no_const,
    )

    plot_route(
        axes[1],
        depot,
        templates,
        exact_pi,
        title="Точный маршрут",
        saved_cost_no_const=exact_cost_no_const,
    )

    fig.suptitle(
        f"Экземпляр {INSTANCE_ID} | Gap  {gap_no_const:.3f}%",
        fontsize=14,
    )

    plt.tight_layout()

    out_path = f"NEWroute_instance_{INSTANCE_ID}_no_const.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Сохранено в {out_path}")


if __name__ == "__main__":
    main()