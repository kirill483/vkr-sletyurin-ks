#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run GLKH for your GTSP dataset.

What it does:
    .pkl dataset -> .gtsp/.par files -> external GLKH solver -> .tour -> .pkl results

Your problem structure:
    depot:     [2]
    templates: [N, K, 5]
        template = [x_in, y_in, x_out, y_out, coverage_length]

GTSP interpretation:
    depot is a singleton cluster
    each field is one cluster
    each field has K candidate template nodes
    GLKH must choose exactly one node from every cluster

Important:
    GLKH is heuristic, not exact.
    This script assumes you already built GLKH and have its binary path.

Typical install:
    wget http://webhotel4.ruc.dk/~keld/research/GLKH/GLKH-1.0.tgz
    tar xzf GLKH-1.0.tgz
    cd GLKH-1.0
    make

Then set:
    GLKH_BIN = "/home/kirill/GLKH-1.0/GLKH"
"""

import os
import re
import pickle
import shutil
import time
import subprocess
from pathlib import Path

import numpy as np


# ==========================
# CONFIG
# ==========================

DATASET_PATH = "data/200_20.pkl"
OUT_PATH = "results/200_20_glkh_gtsp20.pkl"

# Put your real GLKH binary path here.
# Example:
# GLKH_BIN = "/home/kirill/GLKH-1.0/GLKH"
GLKH_BIN = "/home/kirill/GLKH-1.1/GLKH"

WORK_DIR = "glkh_work/gtsp20_200"

SCALE = 1_000_000

# GLKH is heuristic, so time limit is not "proof time".
TIME_LIMIT_SECONDS = 5
RUNS = 1
TRACE_LEVEL = 0

# GLKH/GTSPLIB format variant.
# Most GTSP/GTSPLIB readers use GTSP_SETS.
# If your GLKH build complains, try changing to "GTSP_SETS_SECTION".
GTSP_SET_SECTION_NAME = "GTSP_SET_SECTION"


# ==========================
# COST HELPERS
# ==========================

def compute_real_cost(depot, templates, pi):
    """
    pi:
        [N], values 1..N*K in visiting order.
    """

    n_fields, n_templates, template_dim = templates.shape
    assert template_dim >= 5, "Expected templates with coverage length: [N, K, 5]"

    selected = np.asarray(pi, dtype=np.int64) - 1

    field_ids = selected // n_templates
    template_ids = selected % n_templates

    assert np.array_equal(
        np.sort(field_ids), np.arange(n_fields)
    ), f"Invalid tour: each field must be selected exactly once, got field_ids={field_ids}"

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


def build_glkh_distance_matrix(depot, templates):
    """
    GLKH node ids:
        1           = depot
        2..N*K+1    = candidate templates

    Your candidate id:
        1..N*K

    Mapping:
        glkh_node = candidate_id + 1
        candidate_id = glkh_node - 1
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
    # only travel output_i -> depot
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

    # Do not allow self transitions.
    # For explicit ATSP/GTSP matrix, diagonal can be 0, but large diagonal is safer.
    big = 10**15 / SCALE
    np.fill_diagonal(dist, big)

    dist_int = np.rint(dist * SCALE).astype(np.int64)

    return dist_int


# ==========================
# FILE WRITERS
# ==========================

def write_gtsp_file(path, depot, templates, name="sample"):
    """
    Writes GTSPLIB-like explicit full matrix GTSP file.

    Clusters:
        cluster 1: depot node 1
        cluster 2..N+1: candidate templates for each field
    """

    n_fields, n_templates, _ = templates.shape
    n_candidates = n_fields * n_templates
    dimension = 1 + n_candidates
    gtsp_sets = 1 + n_fields

    dist_int = build_glkh_distance_matrix(depot, templates)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        # AGTSP is the asymmetric generalized TSP.
        # Some builds also accept GTSP. If your GLKH rejects AGTSP, change it to GTSP.
        f.write("TYPE: AGTSP\n")
        f.write(f"COMMENT: Generated from pkl dataset by run_glkh_gtsp.py\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write(f"GTSP_SETS: {gtsp_sets}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        for i in range(dimension):
            row = " ".join(str(int(v)) for v in dist_int[i])
            f.write(row + "\n")

        f.write(f"{GTSP_SET_SECTION_NAME}\n")

        # Cluster 1: depot
        f.write("1 1 -1\n")

        # Clusters 2..N+1: templates of each field
        for field_id in range(n_fields):
            cluster_id = field_id + 2
            nodes = [
                2 + field_id * n_templates + template_id
                for template_id in range(n_templates)
            ]
            f.write(f"{cluster_id} " + " ".join(map(str, nodes)) + " -1\n")

        f.write("EOF\n")


def write_par_file(path, problem_file, output_tour_file):
    problem_file = Path(problem_file).resolve()
    output_tour_file = Path(output_tour_file).resolve()

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PROBLEM_FILE = {problem_file}\n")
        f.write(f"OUTPUT_TOUR_FILE = {output_tour_file}\n")
        f.write(f"RUNS = {RUNS}\n")
        f.write(f"TIME_LIMIT = {TIME_LIMIT_SECONDS}\n")
        f.write(f"TRACE_LEVEL = {TRACE_LEVEL}\n")


# ==========================
# TOUR PARSING
# ==========================

def parse_tour_file(path):
    """
    Parses TSPLIB-like TOUR_SECTION.

    Returns list of positive node ids.
    """

    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    if "TOUR_SECTION" in text:
        after = text.split("TOUR_SECTION", 1)[1]
    else:
        after = text

    # Stop at EOF if present.
    if "EOF" in after:
        after = after.split("EOF", 1)[0]

    nums = [int(x) for x in re.findall(r"-?\d+", after)]
    nodes = [x for x in nums if x > 0]

    return nodes


def normalize_glkh_tour_to_pi(tour_nodes, n_fields, n_templates):
    """
    Converts GLKH node tour to your pi format.

    GLKH nodes:
        1 = depot
        2..N*K+1 = candidates

    pi:
        1..N*K candidates in route order

    If the tour is cyclic, rotate it so that depot is before the first candidate.
    """

    if not tour_nodes:
        raise ValueError("Empty tour")

    # Remove duplicates while preserving order.
    # Some tour files may contain repeated depot/end markers.
    cleaned = []
    seen = set()
    for node in tour_nodes:
        if node not in seen:
            cleaned.append(node)
            seen.add(node)

    tour_nodes = cleaned

    if 1 in tour_nodes:
        depot_pos = tour_nodes.index(1)
        ordered = tour_nodes[depot_pos + 1:] + tour_nodes[:depot_pos]
    else:
        # If output contains only selected non-depot nodes, use as is.
        ordered = tour_nodes

    # Keep only candidate nodes.
    cand_nodes = [node for node in ordered if node != 1]

    # Convert GLKH node -> your candidate id.
    pi = [node - 1 for node in cand_nodes]

    # Remove possible invalid transformed nodes if any.
    max_candidate = n_fields * n_templates
    pi = [x for x in pi if 1 <= x <= max_candidate]

    if len(pi) != n_fields:
        raise ValueError(
            f"Expected {n_fields} selected candidates, got {len(pi)}. "
            f"raw_tour={tour_nodes}, pi={pi}"
        )

    field_ids = [(p - 1) // n_templates for p in pi]
    if sorted(field_ids) != list(range(n_fields)):
        raise ValueError(
            f"Invalid GTSP tour: selected fields are not exactly 0..{n_fields - 1}. "
            f"field_ids={field_ids}, pi={pi}"
        )

    return pi


# ==========================
# SOLVER WRAPPER
# ==========================

def solve_gtsp_glkh(depot, templates, sample_id, work_dir):
    """
    Returns:
        cost, pi, status, raw_output
    """

    n_fields, n_templates, _ = templates.shape

    sample_name = f"sample_{sample_id:06d}"
    gtsp_path = work_dir / f"{sample_name}.gtsp"
    par_path = work_dir / f"{sample_name}.par"
    tour_path = work_dir / f"{sample_name}.tour"

    write_gtsp_file(gtsp_path, depot, templates, name=sample_name)
    write_par_file(
        par_path,
        problem_file=str(gtsp_path),
        output_tour_file=str(tour_path),
    )

    cmd = [str(Path(GLKH_BIN).resolve()), str(par_path.resolve())]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            cwd=str(Path(GLKH_BIN).resolve().parent),
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"GLKH binary not found: {GLKH_BIN}. "
            f"Set GLKH_BIN to your compiled GLKH executable path."
        ) from exc

    raw_output = proc.stdout

    if proc.returncode != 0:
        return None, None, f"GLKH_EXIT_{proc.returncode}", raw_output

    if not tour_path.exists():
        return None, None, "NO_TOUR_FILE", raw_output

    try:
        tour_nodes = parse_tour_file(tour_path)
        pi = normalize_glkh_tour_to_pi(tour_nodes, n_fields, n_templates)
        cost = compute_real_cost(depot, templates, pi)
        return cost, pi, "OK", raw_output
    except Exception as exc:
        return None, None, f"PARSE_ERROR: {exc}", raw_output


def unpack_sample(sample):
    """
    Supports:
        1) tuple: (depot, templates)
        2) dict: {"depot": depot, "templates": templates}
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
    glkh_bin_path = shutil.which(GLKH_BIN) if os.path.basename(GLKH_BIN) == GLKH_BIN else GLKH_BIN
    if glkh_bin_path is None or not Path(glkh_bin_path).exists():
        raise FileNotFoundError(
            f"GLKH binary not found: {GLKH_BIN}\n"
            f"Edit GLKH_BIN in this script, for example:\n"
            f'GLKH_BIN = "/home/kirill/GLKH-1.0/GLKH"'
        )

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(WORK_DIR)
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    results = []

    for i, sample in enumerate(dataset):
        depot, templates = unpack_sample(sample)

        start = time.time()
        cost, pi, status, raw_output = solve_gtsp_glkh(
            depot=depot,
            templates=templates,
            sample_id=i,
            work_dir=work_dir,
        )
        duration = time.time() - start

        results.append((cost, pi, duration, status))

        if cost is None:
            print(
                f"{i + 1}/{len(dataset)} no solution, "
                f"status={status}, time={duration:.3f}s"
            )
            # Save GLKH output for debugging.
            debug_path = work_dir / f"sample_{i:06d}.log"
            debug_path.write_text(raw_output or "", encoding="utf-8", errors="ignore")
        else:
            print(
                f"{i + 1}/{len(dataset)} "
                f"cost={cost:.6f}, status={status}, time={duration:.3f}s"
            )

    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved GLKH results to: {OUT_PATH}")


if __name__ == "__main__":
    main()
