import pickle
import numpy as np


ORTOOLS_RESULTS = ""
DATASET_PATH = ""
MODEL_RESULTS = ""

N_TEMPLATES = 8


def load_results(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        data = data[0]

    tours = [x[1] for x in data]
    return tours


def unpack_sample(sample):
    if isinstance(sample, dict):
        depot, templates = sample["depot"], sample["templates"]
    else:
        depot, templates = sample

    return np.asarray(depot, dtype=np.float64), np.asarray(templates, dtype=np.float64)


def coverage_constant(templates):
    return float(templates[:, :, 4].min(axis=1).sum())


def cost_no_const(depot, templates, pi, n_templates=N_TEMPLATES):
    selected = np.asarray(pi, dtype=np.int64) - 1
    field_ids = selected // n_templates
    template_ids = selected % n_templates

    assert np.array_equal(np.sort(field_ids), np.arange(len(field_ids)))

    chosen = templates[field_ids, template_ids]
    chosen_in = chosen[:, 0:2]
    chosen_out = chosen[:, 2:4]

    transition = (
        np.linalg.norm(chosen_in[0] - depot)
        + np.linalg.norm(chosen_in[1:] - chosen_out[:-1], axis=1).sum()
        + np.linalg.norm(chosen_out[-1] - depot)
    )
    coverage = float(chosen[:, 4].sum())

    return (transition + coverage) - coverage_constant(templates)


def summarize(values, title):
    values = np.asarray(values, dtype=np.float64)
    print(f"{title}: mean={values.mean():.4f}, max={values.max():.4f}, min={values.min():.4f}")


if __name__ == "__main__":
    model_tours = load_results(MODEL_RESULTS)
    opt_tours = load_results(ORTOOLS_RESULTS)

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    assert len(dataset) == len(model_tours) == len(opt_tours)

    model_costs = []
    opt_costs = []

    for sample, model_pi, opt_pi in zip(dataset, model_tours, opt_tours):
        depot, templates = unpack_sample(sample)
        model_costs.append(cost_no_const(depot, templates, model_pi))
        opt_costs.append(cost_no_const(depot, templates, opt_pi))

    model_costs = np.array(model_costs)
    opt_costs = np.array(opt_costs)

    gap = (model_costs / opt_costs - 1.0) * 100

    print(f"Instances: {len(gap)}\n")
    summarize(model_costs, "Model cost")
    summarize(opt_costs, "OR-Tools cost")
    summarize(gap, "Gap, %")