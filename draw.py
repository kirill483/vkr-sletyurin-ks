import pickle
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.patches import Rectangle


def draw_fields(ax, templates):
    """
    Рисует прямоугольники полей по min/max координатам всех in/out точек шаблонов.

    templates:
        [N, 8, 4]
        [x_in, y_in, x_out, y_out]
    """

    n_fields = templates.shape[0]

    for field_id in range(n_fields):
        field_templates = templates[field_id]  # [8, 4]

        points = np.concatenate(
            (
                field_templates[:, 0:2],  # inputs
                field_templates[:, 2:4],  # outputs
            ),
            axis=0
        )  # [16, 2]

        x_min = points[:, 0].min()
        y_min = points[:, 1].min()
        x_max = points[:, 0].max()
        y_max = points[:, 1].max()

        width = x_max - x_min
        height = y_max - y_min

        rect = Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=False,
            linewidth=1.5,
            alpha=0.7,
        )

        ax.add_patch(rect)

        ax.text(
            x_min,
            y_max,
            f"F{field_id}",
            fontsize=9,
            weight="bold",
            verticalalignment="bottom",
        )


MODEL_RESULTS = "results/model_200_20.pkl"
ORTOOLS_RESULTS = "results/exact_200_20.pkl"

# Датасет, на котором считались model/or-tools results
DATASET_PATH = "data/200_20.pkl"


INSTANCE_ID = 39
N_TEMPLATES = 8


def load_results(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # eval.py sometimes saves (results, parallelism)
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


def decode_tour(pi, n_templates=8):
    """
    pi values:
        1..N*K

    returns:
        [(field_id, template_id, action_id), ...]
    """

    decoded = []

    for action_id in pi:
        action_id = int(action_id)

        candidate_id = action_id - 1
        field_id = candidate_id // n_templates
        template_id = candidate_id % n_templates

        decoded.append((field_id, template_id, action_id))

    return decoded


def compute_route_points(depot, templates, pi, n_templates=8):
    """
    Builds route segments.

    travel_segments:
        movement between fields:
            depot -> input_1
            output_1 -> input_2
            ...
            output_last -> depot

    template_segments:
        inside selected template:
            input_i -> output_i
    """

    decoded = decode_tour(pi, n_templates)

    travel_segments = []
    template_segments = []
    labels = []

    current_point = depot

    for step, (field_id, template_id, action_id) in enumerate(decoded):
        template = templates[field_id, template_id]

        in_point = template[0:2]
        out_point = template[2:4]

        travel_segments.append((current_point, in_point))
        template_segments.append((in_point, out_point))

        labels.append({
            "step": step,
            "field_id": field_id,
            "template_id": template_id,
            "action_id": action_id,
            "in": in_point,
            "out": out_point,
        })

        current_point = out_point

    travel_segments.append((current_point, depot))

    return travel_segments, template_segments, labels


def route_cost(depot, templates, pi, n_templates=8):
    travel_segments, _, _ = compute_route_points(
        depot, templates, pi, n_templates
    )

    total = 0.0
    for a, b in travel_segments:
        total += np.linalg.norm(b - a)

    return total


def draw_arrow(ax, a, b, linewidth=1.5, alpha=0.9, linestyle="-"):
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


def plot_route(ax, depot, templates, pi, title, saved_cost=None):
    travel_segments, template_segments, labels = compute_route_points(
        depot, templates, pi, N_TEMPLATES
    )

    recomputed = route_cost(depot, templates, pi, N_TEMPLATES)
    draw_fields(ax, templates)
    ax.scatter([depot[0]], [depot[1]], marker="s", s=100, label="Depot")
    ax.text(depot[0], depot[1], " depot", fontsize=9)

    # Draw all template endpoints lightly
    all_in = templates[..., 0:2].reshape(-1, 2)
    all_out = templates[..., 2:4].reshape(-1, 2)
    ax.scatter(all_in[:, 0], all_in[:, 1], s=10, alpha=0.2, label="All inputs")
    ax.scatter(all_out[:, 0], all_out[:, 1], s=10, alpha=0.2, marker="x", label="All outputs")

    # Draw selected inner template segments: input -> output
    for step, (a, b) in enumerate(template_segments):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            linewidth=2.5,
            alpha=0.8,
        )

        mid = (a + b) / 2
        ax.text(
            mid[0],
            mid[1],
            f"{step}",
            fontsize=9,
            weight="bold",
        )

    # Draw travel route: current output -> next input
    for a, b in travel_segments:
        draw_arrow(ax, a, b, linewidth=1.3, alpha=0.8)

    # Draw selected input/output points and labels
    for item in labels:
        step = item["step"]
        field_id = item["field_id"]
        template_id = item["template_id"]

        in_point = item["in"]
        out_point = item["out"]

        ax.scatter([in_point[0]], [in_point[1]], s=40)
        ax.scatter([out_point[0]], [out_point[1]], s=40, marker="x")

        ax.text(
            in_point[0],
            in_point[1],
            f" in {step}\nF{field_id} T{template_id}",
            fontsize=8,
        )

        ax.text(
            out_point[0],
            out_point[1],
            f" out {step}",
            fontsize=8,
        )

    if saved_cost is None:
        title_text = f"{title}\nrecomputed cost={recomputed:.6f}"
    else:
        title_text = (
            f"{title}\n"
            f"saved cost={saved_cost:.6f}, recomputed={recomputed:.6f}"
        )

    ax.set_title(title_text)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main():
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    depot, templates = unpack_sample(dataset[INSTANCE_ID])

    model_costs, model_tours = load_results(MODEL_RESULTS)
    or_costs, or_tours = load_results(ORTOOLS_RESULTS)

    model_pi = model_tours[INSTANCE_ID]
    or_pi = or_tours[INSTANCE_ID]

    gap = (model_costs[INSTANCE_ID] / or_costs[INSTANCE_ID] - 1.0) * 100

    print(f"Instance: {INSTANCE_ID}")
    print(f"Model cost:   {model_costs[INSTANCE_ID]:.6f}")
    print(f"OR cost:      {or_costs[INSTANCE_ID]:.6f}")
    print(f"Gap:          {gap:.3f}%")
    print(f"Model pi:     {model_pi}")
    print(f"OR-Tools pi:  {or_pi}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_route(
        axes[0],
        depot,
        templates,
        model_pi,
        title="Model route",
        saved_cost=model_costs[INSTANCE_ID],
    )

    plot_route(
        axes[1],
        depot,
        templates,
        or_pi,
        title="OR-Tools route",
        saved_cost=or_costs[INSTANCE_ID],
    )

    fig.suptitle(
        f"Instance {INSTANCE_ID} | Gap {gap:.3f}%",
        fontsize=14,
    )

    plt.tight_layout()
    out_path = f"route_instance_{INSTANCE_ID}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()