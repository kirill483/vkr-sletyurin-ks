import pickle
import numpy as np


MODEL_RESULTS = "results/NEW2020sMANY30gtsp5.pkl"
ORTOOLS_RESULTS = "results/30rtools_gtsp20.pkl"

# Добавь путь к тому же датасету, на котором считались model/or-tools results
DATASET_PATH = "data/30GTSP20_val_seed1234.pkl"  # поменяй на свой путь

N_TEMPLATES = 8


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


def compute_step_details(depot, templates, pi, n_templates=8):
    """
    Возвращает подробности маршрута:
        depot -> template_in
        template_out -> next_template_in
        last_template_out -> depot
    """

    decoded = decode_tour(pi, n_templates)

    rows = []
    total = 0.0

    cur_point = depot
    cur_label = "depot"

    for step, (field_id, template_id, action_id) in enumerate(decoded):
        templ = templates[field_id, template_id]

        in_point = templ[0:2]
        out_point = templ[2:4]

        travel = float(np.linalg.norm(in_point - cur_point))
        total += travel

        rows.append({
            "step": step,
            "from": cur_label,
            "to": f"field={field_id}, template={template_id}, action={action_id}",
            "travel": travel,
            "in": in_point,
            "out": out_point,
        })

        cur_point = out_point
        cur_label = f"field={field_id}, template={template_id}, action={action_id}"

    return_to_depot = float(np.linalg.norm(depot - cur_point))
    total += return_to_depot

    rows.append({
        "step": len(decoded),
        "from": cur_label,
        "to": "depot",
        "travel": return_to_depot,
        "in": depot,
        "out": depot,
    })

    return total, rows


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
        recomputed_cost, rows = compute_step_details(
            depot,
            templates,
            pi,
            N_TEMPLATES
        )

        print(f"Recomputed cost: {recomputed_cost:.6f}")

        print("Step distances:")
        for row in rows:
            if row["to"] == "depot":
                print(
                    f"  step {row['step']:02d}: "
                    f"{row['from']} -> depot | "
                    f"dist={row['travel']:.6f}"
                )
            else:
                in_x, in_y = row["in"]
                out_x, out_y = row["out"]

                print(
                    f"  step {row['step']:02d}: "
                    f"{row['from']} -> {row['to']} | "
                    f"dist={row['travel']:.6f} | "
                    f"in=({in_x:.4f}, {in_y:.4f}) | "
                    f"out=({out_x:.4f}, {out_y:.4f})"
                )


if __name__ == "__main__":
    model_costs, model_tours = load_results(MODEL_RESULTS)
    opt_costs, opt_tours = load_results(ORTOOLS_RESULTS)

    assert len(model_costs) == len(opt_costs)

    gaps = (model_costs / opt_costs - 1.0) * 100

    print(f"Instances: {len(gaps)}")
    print(f"Model avg cost:   {model_costs.mean():.6f}")
    print(f"OR-Tools avg cost:{opt_costs.mean():.6f}")
    print(f"Avg gap:          {gaps.mean():.3f}%")
    print(f"Median gap:       {np.median(gaps):.3f}%")
    print(f"Max gap:          {gaps.max():.3f}%")
    print(f"Min gap:          {gaps.min():.3f}%")
    print(f"gaps: {gaps}")

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    for idx in range(len(gaps)):
        if gaps[idx] <= 15:
            continue

        depot, templates = unpack_sample(dataset[idx])

        print("\n" + "=" * 80)
        print(f"INSTANCE {idx}")
        print("=" * 80)
        print(f"Model cost: {model_costs[idx]:.6f}")
        print(f"OR cost:    {opt_costs[idx]:.6f}")
        print(f"Gap:        {gaps[idx]:.3f}%")

        print_decoded_route(
            name=f"MODEL ROUTE / INSTANCE {idx}",
            cost=model_costs[idx],
            pi=model_tours[idx],
            depot=depot,
            templates=templates,
        )

        print_decoded_route(
            name=f"OR-TOOLS ROUTE / INSTANCE {idx}",
            cost=opt_costs[idx],
            pi=opt_tours[idx],
            depot=depot,
            templates=templates,
        )