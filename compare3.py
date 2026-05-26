import pickle
import numpy as np


#MODEL_RESULTS = "results/model_200_20.pkl"
#ORTOOLS_RESULTS = "results/exact_200_20.pkl"
#MODEL_RESULTS = "results/200_20_nearest_neighbor_dp_gtsp20.pkl"

# Датасет, на котором считались model/or-tools results
#DATASET_PATH = "data/200_20.pkl"

#ORTOOLS_RESULTS = "results/200_20_nearest_neighbor_dp_gtsp20.pkl"
#MODEL_RESULTS = "results/300_10_model.pkl"
ORTOOLS_RESULTS = "results/exact_300_10.pkl"

# Датасет, на котором считались model/or-tools results
DATASET_PATH = "data/300_10.pkl"
MODEL_RESULTS= "results/300_10_nearest_neighbor_dp_gtsp20.pkl"


N_TEMPLATES = 8

# Порог для разбиения на GOOD/BAD
GAP_PRINT_THRESHOLD = 1.0

# Если None — считать статистику по всем instances.
# Если 100 — считать основную статистику только по top-100 худших gap.
TOP_K_BY_GAP = True
TOP_K_BY_GAP = 200

# True = top-K самых плохих gap, то есть максимальные gap.
# False = top-K самых хороших gap, то есть минимальные gap.
TOP_K_WORST = False


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


def compute_coverage_constant(templates):
    """
    Для каждого поля берём минимальную длину шаблона.
    Сумма этих минимумов — константа instance-а.

    templates:
        [N, K, 4] или [N, K, 5]

    Если template_dim == 4, coverage_constant = 0.
    """

    if templates.shape[-1] < 5:
        return 0.0

    lengths = templates[:, :, 4]  # [N, K]
    min_per_field = lengths.min(axis=1)  # [N]

    return float(min_per_field.sum())


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
        {
            "transition": ...,
            "coverage": ...,
            "coverage_constant": ...,
            "coverage_delta": ...,
            "full": ...,
            "full_no_const": ...
        }
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
        coverage_cost = float(chosen[:, 4].sum())
    else:
        coverage_cost = 0.0

    coverage_constant = compute_coverage_constant(templates)
    coverage_delta = coverage_cost - coverage_constant

    full_cost = transition_cost + coverage_cost
    full_no_const = full_cost - coverage_constant

    return {
        "transition": float(transition_cost),
        "coverage": float(coverage_cost),
        "coverage_constant": float(coverage_constant),
        "coverage_delta": float(coverage_delta),
        "full": float(full_cost),
        "full_no_const": float(full_no_const),
    }


def compute_average_parts(dataset, tours):
    transition_costs = []
    coverage_costs = []
    coverage_constants = []
    coverage_deltas = []
    full_costs = []
    full_no_const_costs = []

    for sample, pi in zip(dataset, tours):
        depot, templates = unpack_sample(sample)

        parts = compute_cost_parts(
            depot,
            templates,
            pi,
            N_TEMPLATES
        )

        transition_costs.append(parts["transition"])
        coverage_costs.append(parts["coverage"])
        coverage_constants.append(parts["coverage_constant"])
        coverage_deltas.append(parts["coverage_delta"])
        full_costs.append(parts["full"])
        full_no_const_costs.append(parts["full_no_const"])

    return {
        "transition": np.array(transition_costs, dtype=np.float64),
        "coverage": np.array(coverage_costs, dtype=np.float64),
        "coverage_constant": np.array(coverage_constants, dtype=np.float64),
        "coverage_delta": np.array(coverage_deltas, dtype=np.float64),
        "full": np.array(full_costs, dtype=np.float64),
        "full_no_const": np.array(full_no_const_costs, dtype=np.float64),
    }


def make_top_k_mask(gaps, top_k=None, worst=True):
    """
    Возвращает mask для top-K по gap.

    worst=True:
        берём K самых больших gap.

    worst=False:
        берём K самых маленьких gap.
    """

    n = len(gaps)

    if top_k is None:
        return np.ones(n, dtype=bool)

    top_k = min(int(top_k), n)

    if worst:
        indices = np.argsort(gaps)[-top_k:]
    else:
        indices = np.argsort(gaps)[:top_k]

    mask = np.zeros(n, dtype=bool)
    mask[indices] = True

    return mask


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def print_distribution_stats(title, values):
    stats = summarize_values(values)

    print(f"{title}")
    print(f"  mean:   {stats['mean']:.6f}")
    print(f"  median: {stats['median']:.6f}")
    print(f"  std:    {stats['std']:.6f}")
    print(f"  min:    {stats['min']:.6f}")
    print(f"  max:    {stats['max']:.6f}")


def print_metric_table(model_parts, opt_parts, mask=None):
    rows = [
        ("Transition only", "transition"),
        ("Coverage full", "coverage"),
        ("Coverage const", "coverage_constant"),
        ("Coverage delta", "coverage_delta"),
        ("Full cost", "full"),
        ("Full no const", "full_no_const"),
    ]

    print()
    print("| Метрика          | Model | OR-Tools | Gap |")
    print("| ---------------- | ----: | -------: | --: |")

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
            f"| {name:<16} | "
            f"{model_mean:8.4f} | "
            f"{opt_mean:8.4f} | "
            f"{gap:7.3f}% |"
        )


def print_group_stats(title, mask, gaps_no_const, model_parts, opt_parts):
    count = int(mask.sum())

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Count: {count}/{len(gaps_no_const)}")

    if count == 0:
        print("No instances in this group.")
        return

    print_metric_table(
        model_parts,
        opt_parts,
        mask=mask
    )

    group_gaps = gaps_no_const[mask]

    print()
    print_distribution_stats(
        "Model full_no_const",
        model_parts["full_no_const"][mask]
    )

    print()
    print_distribution_stats(
        "OR-Tools full_no_const",
        opt_parts["full_no_const"][mask]
    )

    print()
    print_distribution_stats(
        "Gap by full_no_const, %",
        group_gaps
    )

    print()
    print(f"gaps_no_const: {group_gaps}")


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

    Также считает:
        coverage_constant
        coverage_delta
        total_no_const
    """

    decoded = decode_tour(pi, n_templates)

    rows = []
    total = 0.0
    total_travel = 0.0
    total_coverage = 0.0

    cur_point = depot
    cur_label = "depot"

    has_coverage = templates.shape[-1] >= 5
    coverage_constant = compute_coverage_constant(templates)

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

    coverage_delta = total_coverage - coverage_constant
    total_no_const = total - coverage_constant

    return {
        "total": float(total),
        "total_no_const": float(total_no_const),
        "rows": rows,
        "total_travel": float(total_travel),
        "total_coverage": float(total_coverage),
        "coverage_constant": float(coverage_constant),
        "coverage_delta": float(coverage_delta),
    }


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
        details = compute_step_details(
            depot,
            templates,
            pi,
            N_TEMPLATES
        )

        print(f"Recomputed cost:          {details['total']:.6f}")
        print(f"Recomputed cost no const: {details['total_no_const']:.6f}")
        print(f"  travel total:           {details['total_travel']:.6f}")
        print(f"  coverage total:         {details['total_coverage']:.6f}")
        print(f"  coverage constant:      {details['coverage_constant']:.6f}")
        print(f"  coverage delta:         {details['coverage_delta']:.6f}")

        print("Step distances:")
        for row in details["rows"]:
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

    # Проверяем, что coverage_constant одинаковый для model и opt,
    # потому что он зависит только от instance, а не от маршрута.
    assert np.allclose(
        model_parts["coverage_constant"],
        opt_parts["coverage_constant"]
    ), "Coverage constants differ between model and OR-Tools. Something is wrong."

    coverage_constants = model_parts["coverage_constant"]

    # Gap считаем по cost без константы
    eps = 1e-12
    gaps_no_const = (
        model_parts["full_no_const"] / np.maximum(opt_parts["full_no_const"], eps)
        - 1.0
    ) * 100

    # Для справки можно считать и старый full gap.
    gaps_full = (
        model_parts["full"] / np.maximum(opt_parts["full"], eps)
        - 1.0
    ) * 100

    print(f"Instances: {len(gaps_no_const)}")

    # Маска для top-K, если включено.
    selected_mask = make_top_k_mask(
        gaps_no_const,
        top_k=TOP_K_BY_GAP,
        worst=TOP_K_WORST
    )

    if TOP_K_BY_GAP is None:
        selected_title = "ALL INSTANCES"
    else:
        direction = "WORST" if TOP_K_WORST else "BEST"
        selected_title = f"TOP {int(selected_mask.sum())} {direction} INSTANCES BY GAP_NO_CONST"

    print()
    print("=" * 80)
    print(selected_title)
    print("=" * 80)

    print_metric_table(
        model_parts,
        opt_parts,
        mask=selected_mask
    )

    print()
    print_distribution_stats(
        "Coverage constant",
        coverage_constants[selected_mask]
    )

    print()
    print_distribution_stats(
        "Model full cost without constant",
        model_parts["full_no_const"][selected_mask]
    )

    print()
    print_distribution_stats(
        "OR-Tools full cost without constant",
        opt_parts["full_no_const"][selected_mask]
    )

    print()
    print_distribution_stats(
        "Gap without constant, %",
        gaps_no_const[selected_mask]
    )

    print()
    print_distribution_stats(
        "Old full gap, %",
        gaps_full[selected_mask]
    )

    print()
    print(f"gaps_no_const selected: {gaps_no_const[selected_mask]}")

    good_mask = selected_mask & (gaps_no_const > -1) & (gaps_no_const <= GAP_PRINT_THRESHOLD)
    bad_mask = selected_mask & (gaps_no_const > GAP_PRINT_THRESHOLD)

    print_group_stats(
        title=f"GOOD INSTANCES: GAP_NO_CONST <= {GAP_PRINT_THRESHOLD:.3f}%",
        mask=good_mask,
        gaps_no_const=gaps_no_const,
        model_parts=model_parts,
        opt_parts=opt_parts,
    )

    print_group_stats(
        title=f"BAD INSTANCES: GAP_NO_CONST > {GAP_PRINT_THRESHOLD:.3f}%",
        mask=bad_mask,
        gaps_no_const=gaps_no_const,
        model_parts=model_parts,
        opt_parts=opt_parts,
    )

    # Подробный вывод маршрутов.
    # Сейчас оставлена логика: печатать странные случаи,
    # где model сильно лучше opt по gap_no_const.
    for idx in range(len(gaps_no_const)):
        if not selected_mask[idx]:
            continue

        if gaps_no_const[idx] >= -1000:
            continue

        depot, templates = unpack_sample(dataset[idx])

        print("\n" + "=" * 80)
        print(f"INSTANCE {idx}")
        print("=" * 80)
        print(f"Coverage constant:      {coverage_constants[idx]:.6f}")
        print(f"Model full cost:        {model_parts['full'][idx]:.6f}")
        print(f"OR full cost:           {opt_parts['full'][idx]:.6f}")
        print(f"Model full no const:    {model_parts['full_no_const'][idx]:.6f}")
        print(f"OR full no const:       {opt_parts['full_no_const'][idx]:.6f}")
        print(f"Gap no const:           {gaps_no_const[idx]:.3f}%")
        print(f"Old full gap:           {gaps_full[idx]:.3f}%")

      #  print_decoded_route(
    #        name=f"MODEL ROUTE / INSTANCE {idx}",
   #         cost=model_parts["full_no_const"][idx],
   #         pi=model_tours[idx],
   #         depot=depot,
   #         templates=templates,
   #     )

  #      print_decoded_route(
  #          name=f"OR-TOOLS ROUTE / INSTANCE {idx}",
 #           cost=opt_parts["full_no_const"][idx],
   #         pi=opt_tours[idx],
  #          depot=depot,
    #        templates=templates,
   #     )