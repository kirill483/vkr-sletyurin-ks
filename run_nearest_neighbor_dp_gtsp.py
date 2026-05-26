#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nearest Neighbor + DP baseline for your GTSP dataset.

What it does:
    1. Loads dataset from DATASET_PATH.
    2. For each sample:
        - builds cluster/field order by Nearest Neighbor;
        - chooses optimal template for this fixed order using DP;
        - computes real cost;
    3. Saves results to OUT_PATH.

Result format:
    results[i] = (cost, pi, duration)

where:
    cost     - real route cost
    pi       - selected template node ids in visiting order, values 1..N*K
    duration - solve time for this instance

Your sample format can be either:
    (depot, templates)

or:
    {
        "depot": depot,
        "templates": templates
    }

templates shape:
    [N, K, 5]

template:
    [x_in, y_in, x_out, y_out, coverage_length]
"""

import pickle
import time
from pathlib import Path

import numpy as np


# ==========================
# CONFIG
# ==========================
DATASET_PATH = "data/300_10.pkl"

#DATASET_PATH = "data/200_20.pkl"
OUT_PATH = "results/300_10_nearest_neighbor_dp_gtsp20.pkl"


# ==========================
# COST
# ==========================

def compute_real_cost(depot, templates, pi):
    """
    pi:
        list of selected template node ids in format 1..N*K
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

    return float(travel_cost + coverage_cost)


# ==========================
# NEAREST NEIGHBOR ORDER
# ==========================

def nearest_neighbor_cluster_order(depot, templates):
    """
    Builds field/cluster order by nearest neighbor.

    First field:
        nearest field to depot, where distance to a field is:
            min over templates t:
                distance(depot, input(field, t))

    Next fields:
        nearest unvisited field to current field, where field-to-field distance is:
            min over templates a,b:
                distance(output(current_field, a), input(next_field, b))

    Returns:
        order: list of field ids, e.g. [3, 0, 5, 1, ...]
    """

    n_fields, n_templates, _ = templates.shape

    template_in = templates[:, :, 0:2]   # [N, K, 2]
    template_out = templates[:, :, 2:4]  # [N, K, 2]

    unvisited = set(range(n_fields))
    order = []

    # Choose first field nearest to depot
    best_field = None
    best_dist = float("inf")

    for field_id in unvisited:
        dists = np.linalg.norm(template_in[field_id] - depot[None, :], axis=1)
        dist = float(np.min(dists))

        if dist < best_dist:
            best_dist = dist
            best_field = field_id

    current_field = best_field
    order.append(current_field)
    unvisited.remove(current_field)

    # Choose next fields greedily
    while unvisited:
        best_next_field = None
        best_dist = float("inf")

        current_outputs = template_out[current_field]  # [K, 2]

        for next_field in unvisited:
            next_inputs = template_in[next_field]  # [K, 2]

            dists = np.linalg.norm(
                current_outputs[:, None, :] - next_inputs[None, :, :],
                axis=-1
            )

            dist = float(np.min(dists))

            if dist < best_dist:
                best_dist = dist
                best_next_field = next_field

        current_field = best_next_field
        order.append(current_field)
        unvisited.remove(current_field)

    return order


# ==========================
# DP TEMPLATE SELECTION FOR FIXED ORDER
# ==========================

def choose_templates_dp_for_order(depot, templates, order):
    """
    For a fixed field order, chooses the best template in each field exactly.

    DP:
        dp[pos, t] = minimum cost up to position pos
                     if template t is selected for field order[pos]

    Complexity:
        O(N * K^2)

    Returns:
        pi: selected template node ids in visiting order, values 1..N*K
    """

    n_fields, n_templates, _ = templates.shape

    order = list(order)
    assert len(order) == n_fields, "Order length must equal number of fields"

    template_in = templates[:, :, 0:2]
    template_out = templates[:, :, 2:4]
    template_len = templates[:, :, 4]

    dp = np.full((n_fields, n_templates), np.inf, dtype=np.float64)
    parent = np.full((n_fields, n_templates), -1, dtype=np.int64)

    # First field: depot -> input(template) + coverage(template)
    first_field = order[0]

    for t in range(n_templates):
        dp[0, t] = (
            np.linalg.norm(template_in[first_field, t] - depot)
            + template_len[first_field, t]
        )

    # Transitions between consecutive fields
    for pos in range(1, n_fields):
        prev_field = order[pos - 1]
        curr_field = order[pos]

        for curr_t in range(n_templates):
            best_cost = float("inf")
            best_prev_t = -1

            for prev_t in range(n_templates):
                transition_cost = (
                    np.linalg.norm(
                        template_out[prev_field, prev_t]
                        - template_in[curr_field, curr_t]
                    )
                    + template_len[curr_field, curr_t]
                )

                candidate_cost = dp[pos - 1, prev_t] + transition_cost

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_prev_t = prev_t

            dp[pos, curr_t] = best_cost
            parent[pos, curr_t] = best_prev_t

    # Final transition: output(last template) -> depot
    last_field = order[-1]

    best_total_cost = float("inf")
    best_last_t = -1

    for t in range(n_templates):
        total_cost = dp[-1, t] + np.linalg.norm(template_out[last_field, t] - depot)

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_t = t

    # Reconstruct selected template ids
    selected_templates = [-1] * n_fields
    selected_templates[-1] = int(best_last_t)

    for pos in range(n_fields - 1, 0, -1):
        selected_templates[pos - 1] = int(parent[pos, selected_templates[pos]])

    # Convert to pi:
    # node id = field_id * K + template_id + 1
    pi = []

    for pos, field_id in enumerate(order):
        template_id = selected_templates[pos]
        node_id = field_id * n_templates + template_id + 1
        pi.append(int(node_id))

    return pi


# ==========================
# SOLVER
# ==========================

def solve_gtsp_nearest_neighbor_dp(depot, templates):
    """
    Complete baseline:

        1. Nearest Neighbor builds field order.
        2. DP chooses optimal templates for this fixed order.

    Returns:
        cost, pi
    """

    order = nearest_neighbor_cluster_order(depot, templates)
    pi = choose_templates_dp_for_order(depot, templates, order)
    cost = compute_real_cost(depot, templates, pi)

    return cost, pi


def unpack_sample(sample):
    """
    Supports both formats:

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


# ==========================
# MAIN
# ==========================

def main():
    dataset_path = Path(DATASET_PATH)
    out_path = Path(OUT_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Edit DATASET_PATH in this script if needed."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    results = []

    for i, sample in enumerate(dataset):
        depot, templates = unpack_sample(sample)

        start = time.time()
        cost, pi = solve_gtsp_nearest_neighbor_dp(depot, templates)
        duration = time.time() - start

        results.append((cost, pi, duration))

        print(
            f"{i + 1}/{len(dataset)} "
            f"cost={cost:.6f}, time={duration:.6f}s"
        )

    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved Nearest Neighbor + DP results to: {out_path}")


if __name__ == "__main__":
    main()
