import pickle
import numpy as np

MODEL_RESULTS = "results/model_200_20.pkl"
#ORTOOLS_RESULTS = "results/exact_200_20.pkl"

# Датасет, на котором считались model/or-tools results
DATASET_PATH = "data/200_20.pkl"

ORTOOLS_RESULTS = "results/200_20_nearest_neighbor_dp_gtsp20.pkl"
#MODEL_RESULTS = "results/300_10_model.pkl"
#ORTOOLS_RESULTS = "results/exact_300_10.pkl"

# Датасет, на котором считались model/or-tools results
#DATASET_PATH = "data/300_10.pkl"

N_TEMPLATES = 8
GAP_PRINT_THRESHOLD = 1


def load_results(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # eval сохраняет (results, parallelism)
    if isinstance(data, tuple) and len(data) == 2:
        data = data[0]

    costs = np.array([x[0] for x in data], dtype=np.float64)
    tours = [x[1] for x in data]

    return costs, tours


def unpack_sample(sample):
    """
    Поддерживает:
        (depot, templates)

    или:
        {
            "depot": depot,
            "templates": templates
        }
    """

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
    pi:
        список action ids в формате модели:
            1..N*K

    Возвращает:
        список (field_id, template_id, action_id)

    field_id:
        0..N-1

    template_id:
        0..K-1
    """

    decoded = []

    for action_id in pi:
        action_id = int(action_id)

        candidate_id = action_id - 1
        field_id = candidate_id // n_templates
        template_id = candidate_id % n_templates

        decoded.append((field_id, template_id, action_id))

    return decoded


def compute_cost_parts(depot, templates, pi, n_templates=8):
    """
    Считает cost по частям.

    templates:
        [N, K, 4] или [N, K, 5]

    Если templates.shape[-1] >= 5:
        template = [x_in, y_in, x_out, y_out, coverage_length]

    Если templates.shape[-1] == 4:
        coverage считается 0.

    return:
        transition_cost, coverage_cost, full_cost
    """

    n_fields, n_templates_from_data, template_dim = templates.shape

    assert n_templates_from_data == n_templates, (
        f"Expected {n_templates} templates, got {n_templates_from_data}"
    )

    selected = np.asarray(pi, dtype=np.int64) - 1

    field_ids = selected // n_templates
    template_ids = selected % n_templates

    assert np.array_equal(
        np.sort(field_ids),
        np.arange(n_fields)
    ), "Invalid tour: each field must be selected exactly once"

    chosen = templates[field_ids, template_ids]

    chosen_in = chosen[:, 0:2]
    chosen_out = chosen[:, 2:4]

    transition_cost = (
        np.linalg.norm(chosen_in[0] - depot)
        + np.linalg.norm(chosen_in[1:] - chosen_out[:-1], axis=1).sum()
        + np.linalg.norm(chosen_out[-1] - depot)
    )

    if template_dim >= 5:
        coverage_cost = chosen[:, 4].sum()
    else:
        coverage_cost = 0.0

    full_cost = transition_cost + coverage_cost

    return (
        float(transition_cost),
        float(coverage_cost),
        float(full_cost),
    )


def compute_average_parts(dataset, tours):
    transition_costs = []
    coverage_costs = []
    full_costs = []

    for sample, pi in zip(dataset, tours):
        depot, templates = unpack_sample(sample)

        transition, coverage, full = compute_cost_parts(
            depot,
            templates,
            pi,
            N_TEMPLATES
        )

        transition_costs.append(transition)
        coverage_costs.append(coverage)
        full_costs.append(full)

    return {
        "transition": np.array(transition_costs, dtype=np.float64),
        "coverage": np.array(coverage_costs, dtype=np.float64),
        "full": np.array(full_costs, dtype=np.float64),
    }

def print_group_stats(title, mask, gaps, model_parts, opt_parts):
    count = int(mask.sum())

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Count: {count}/{len(gaps)}")

    if count == 0:
        print("No instances in this group.")
        return

    print_metric_table(
        model_parts,
        opt_parts,
        mask=mask
    )

    group_gaps = gaps[mask]

    print()
    print(f"Model avg full cost:   {model_parts['full'][mask].mean():.6f}")
    print(f"OR-Tools avg full cost:{opt_parts['full'][mask].mean():.6f}")
    print(f"Avg full gap:          {group_gaps.mean():.3f}%")
    print(f"Median full gap:       {np.median(group_gaps):.3f}%")
    print(f"Max full gap:          {group_gaps.max():.3f}%")
    print(f"Min full gap:          {group_gaps.min():.3f}%")
    print(f"gaps: {group_gaps}")

def print_metric_table(model_parts, opt_parts, mask=None):
    rows = [
        ("Transition only", "transition"),
        ("Coverage only", "coverage"),
        ("Full cost", "full"),
    ]

    print()
    print("| Метрика         | Model | OR-Tools | Gap |")
    print("| --------------- | ----: | -------: | --: |")

    for name, key in rows:
        if mask is None:
            model_values = model_parts[key]
            opt_values = opt_parts[key]
        else:
            model_values = model_parts[key][mask]
            opt_values = opt_parts[key][mask]

        if len(model_values) == 0:
            model_mean = float("nan")
            opt_mean = float("nan")
            gap = float("nan")
        else:
            model_mean = model_values.mean()
            opt_mean = opt_values.mean()

            if abs(opt_mean) < 1e-12:
                gap = float("nan")
            else:
                gap = (model_mean / opt_mean - 1.0) * 100

        print(
            f"| {name:<15} | "
            f"{model_mean:5.2f} | "
            f"{opt_mean:8.2f} | "
            f"{gap:4.1f}% |"
        )


def compute_step_details(depot, templates, pi, n_templates=8):
    """
    Возвращает подробности маршрута:
        depot -> template_in
        + coverage_length шаблона
        template_out -> next_template_in
        + coverage_length следующего шаблона
        last_template_out -> depot

    templates:
        [N, K, 4] или [N, K, 5]

    Если templates.shape[-1] >= 5, учитывается coverage_length.
    """

    decoded = decode_tour(pi, n_templates)

    rows = []
    total = 0.0
    total_travel = 0.0
    total_coverage = 0.0

    cur_point = depot
    cur_label = "depot"

    has_coverage = templates.shape[-1] >= 5

    for step, (field_id, template_id, action_id) in enumerate(decoded):
        templ = templates[field_id, template_id]

        in_point = templ[0:2]
        out_point = templ[2:4]
        coverage = float(templ[4]) if has_coverage else 0.0

        travel = float(np.linalg.norm(in_point - cur_point))

        total_travel += travel
        total_coverage += coverage
        total += travel + coverage

        rows.append({
            "step": step,
            "from": cur_label,
            "to": f"field={field_id}, template={template_id}, action={action_id}",
            "travel": travel,
            "coverage": coverage,
            "step_cost": travel + coverage,
            "in": in_point,
            "out": out_point,
        })

        cur_point = out_point
        cur_label = f"field={field_id}, template={template_id}, action={action_id}"

    return_to_depot = float(np.linalg.norm(depot - cur_point))

    total_travel += return_to_depot
    total += return_to_depot

    rows.append({
        "step": len(decoded),
        "from": cur_label,
        "to": "depot",
        "travel": return_to_depot,
        "coverage": 0.0,
        "step_cost": return_to_depot,
        "in": depot,
        "out": depot,
    })

    return total, rows, total_travel, total_coverage


def print_decoded_route(name, cost, pi, depot=None, templates=None):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Cost: {cost:.6f}")
    print(f"Raw pi: {pi}")

    decoded = decode_tour(pi, N_TEMPLATES)

    print("Decoded route:")
    for step, (field_id, template_id, action_id) in enumerate(decoded):
        print(
            f"  step {step:02d}: "
            f"action={action_id:3d} | "
            f"field={field_id:2d} | "
            f"template={template_id}"
        )

    if depot is not None and templates is not None:
        recomputed_cost, rows, total_travel, total_coverage = compute_step_details(
            depot,
            templates,
            pi,
            N_TEMPLATES
        )

        print(f"Recomputed cost: {recomputed_cost:.6f}")
        print(f"  travel total:   {total_travel:.6f}")
        print(f"  coverage total: {total_coverage:.6f}")

        print("Step distances:")
        for row in rows:
            if row["to"] == "depot":
                print(
                    f"  step {row['step']:02d}: "
                    f"{row['from']} -> depot | "
                    f"travel={row['travel']:.6f} | "
                    f"coverage={row['coverage']:.6f} | "
                    f"step_cost={row['step_cost']:.6f}"
                )
            else:
                in_x, in_y = row["in"]
                out_x, out_y = row["out"]

                print(
                    f"  step {row['step']:02d}: "
                    f"{row['from']} -> {row['to']} | "
                    f"travel={row['travel']:.6f} | "
                    f"coverage={row['coverage']:.6f} | "
                    f"step_cost={row['step_cost']:.6f} | "
                    f"in=({in_x:.4f}, {in_y:.4f}) | "
                    f"out=({out_x:.4f}, {out_y:.4f})"
                )


if __name__ == "__main__":
    model_costs, model_tours = load_results(MODEL_RESULTS)
    opt_costs, opt_tours = load_results(ORTOOLS_RESULTS)

    assert len(model_costs) == len(opt_costs)

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    assert len(dataset) == len(model_tours), (
        f"Dataset size {len(dataset)} != model tours {len(model_tours)}"
    )
    assert len(dataset) == len(opt_tours), (
        f"Dataset size {len(dataset)} != OR-Tools tours {len(opt_tours)}"
    )

    model_parts = compute_average_parts(dataset, model_tours)
    opt_parts = compute_average_parts(dataset, opt_tours)

    # Gap считаем по full cost
    gaps = (model_parts["full"] / opt_parts["full"] - 1.0) * 100

    print(f"Instances: {len(gaps)}")

    print()
    print("=" * 80)
    print("ALL INSTANCES")
    print("=" * 80)

    print_metric_table(model_parts, opt_parts)

    print()
    print(f"Model avg full cost:   {model_parts['full'].mean():.6f}")
    print(f"OR-Tools avg full cost:{opt_parts['full'].mean():.6f}")
    print(f"Avg full gap:          {gaps.mean():.3f}%")
    print(f"Median full gap:       {np.median(gaps):.3f}%")
    print(f"Max full gap:          {gaps.max():.3f}%")
    print(f"Min full gap:          {gaps.min():.3f}%")
    print(f"gaps: {gaps}")

    good_mask = (gaps > -1) & (gaps <= GAP_PRINT_THRESHOLD)
    bad_mask = gaps > GAP_PRINT_THRESHOLD

    print_group_stats(
        title=f"GOOD INSTANCES: GAP <= {GAP_PRINT_THRESHOLD:.3f}%",
        mask=good_mask,
        gaps=gaps,
        model_parts=model_parts,
        opt_parts=opt_parts,
    )

    print_group_stats(
        title=f"BAD INSTANCES: GAP > {GAP_PRINT_THRESHOLD:.3f}%",
        mask=bad_mask,
        gaps=gaps,
        model_parts=model_parts,
        opt_parts=opt_parts,
    )

    for idx in range(len(gaps)):
        if gaps[idx] >= -10:
            continue

        depot, templates = unpack_sample(dataset[idx])

        print("\n" + "=" * 80)
        print(f"INSTANCE {idx}")
        print("=" * 80)
        print(f"Model full cost: {model_parts['full'][idx]:.6f}")
        print(f"OR full cost:    {opt_parts['full'][idx]:.6f}")
        print(f"Gap:             {gaps[idx]:.3f}%")

        print_decoded_route(
            name=f"MODEL ROUTE / INSTANCE {idx}",
            cost=model_parts["full"][idx],
            pi=model_tours[idx],
            depot=depot,
            templates=templates,
        )

        print_decoded_route(
            name=f"OR-TOOLS ROUTE / INSTANCE {idx}",
            cost=opt_parts["full"][idx],
            pi=opt_tours[idx],
            depot=depot,
            templates=templates,
        )