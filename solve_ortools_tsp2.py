import pickle
import time
import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

DATASET_PATH = "data/5_400GTSP20_val_seed1234.pkl"
OUT_PATH = "results/5_NEW400rtools_gtsp20.pkl"

SCALE = 1_000_000
TIME_LIMIT_SECONDS = 5

# Большой штраф за то, что поле вообще не выбрано.
# Важно: он должен быть сильно больше любого разумного travel cost.
PENALTY = 10**12


def compute_real_cost(depot, templates, pi):
    """
    depot:
        [2]

    templates:
        [N, K, 5]
        где template = [x_in, y_in, x_out, y_out, coverage_length]

    pi:
        [N], значения 1..N*K

    Считает:
        depot -> input первого шаблона
        + coverage первого шаблона
        + output шаблона -> input следующего шаблона
        + coverage следующего шаблона
        + output последнего шаблона -> depot
    """

    n_fields, n_templates, template_dim = templates.shape
    assert template_dim >= 5, "Expected templates with coverage length: [N, K, 5]"

    selected = np.asarray(pi, dtype=np.int64) - 1

    field_ids = selected // n_templates
    template_ids = selected % n_templates

    assert np.array_equal(
        np.sort(field_ids), np.arange(n_fields)
    ), "Invalid tour: each field must be selected exactly once"

    chosen = templates[field_ids, template_ids]  # [N, 5]

    chosen_in = chosen[:, 0:2]
    chosen_out = chosen[:, 2:4]
    coverage_lengths = chosen[:, 4]

    travel_cost = (
        np.linalg.norm(chosen_in[0] - depot)
        + np.linalg.norm(chosen_in[1:] - chosen_out[:-1], axis=1).sum()
        + np.linalg.norm(chosen_out[-1] - depot)
    )

    coverage_cost = coverage_lengths.sum()

    return travel_cost + coverage_cost

def solve_gtsp_ortools(depot, templates):
    """
    templates:
        [N, K, 5]
        template = [x_in, y_in, x_out, y_out, coverage_length]

    OR-Tools nodes:
        0       = depot
        1..N*K  = candidate-шаблоны
    """

    n_fields, n_templates, template_dim = templates.shape
    assert template_dim >= 5, "Expected templates with coverage length: [N, K, 5]"

    n_candidates = n_fields * n_templates
    n_total = 1 + n_candidates

    template_in = templates[..., 0:2].reshape(n_candidates, 2)
    template_out = templates[..., 2:4].reshape(n_candidates, 2)
    template_len = templates[..., 4].reshape(n_candidates)

    dist = np.zeros((n_total, n_total), dtype=np.float64)

    # depot -> candidate_j:
    # travel depot -> input_j + coverage_length_j
    dist[0, 1:] = (
        np.linalg.norm(template_in - depot[None, :], axis=1)
        + template_len
    )

    # candidate_i -> depot:
    # только travel output_i -> depot
    # coverage_i уже была оплачена при входе в candidate_i
    dist[1:, 0] = np.linalg.norm(template_out - depot[None, :], axis=1)

    # candidate_i -> candidate_j:
    # travel output_i -> input_j + coverage_length_j
    dist[1:, 1:] = (
        np.linalg.norm(
            template_out[:, None, :] - template_in[None, :, :],
            axis=-1
        )
        + template_len[None, :]
    )

    dist_int = np.rint(dist * SCALE).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n_total, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_int[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    for field_id in range(n_fields):
        candidate_nodes = [
            1 + field_id * n_templates + template_id
            for template_id in range(n_templates)
        ]

        candidate_indices = [manager.NodeToIndex(node) for node in candidate_nodes]

        routing.AddDisjunction(candidate_indices, PENALTY, 1)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = TIME_LIMIT_SECONDS

    solution = routing.SolveWithParameters(search_params)

    if solution is None:
        return None, None

    pi = []
    index = routing.Start(0)

    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)

        if node != 0:
            pi.append(node)

        index = solution.Value(routing.NextVar(index))

    pi = np.asarray(pi, dtype=np.int64)

    if len(pi) != n_fields:
        return None, None

    cost = compute_real_cost(depot, templates, pi)

    return cost, pi.tolist()

def unpack_sample(sample):
    """
    Поддерживает оба формата:

    1) tuple:
        (depot, templates)

    2) dict:
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


if __name__ == "__main__":
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    results = []

    for i, sample in enumerate(dataset):
        depot, templates = unpack_sample(sample)

        start = time.time()
        cost, pi = solve_gtsp_ortools(depot, templates)
        duration = time.time() - start

        results.append((cost, pi, duration))

        if cost is None:
            print(f"{i + 1}/{len(dataset)} no solution, time={duration:.3f}s")
        else:
            print(f"{i + 1}/{len(dataset)} cost={cost:.6f}, time={duration:.3f}s")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved OR-Tools results to: {OUT_PATH}")
