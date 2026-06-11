import torch
import os
import json
import pickle
import time

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import timedelta

from problems.tsp.problem_tsp import TSP


class EvalConfig:
    datasets = ""
    model = ""

    val_size = 300
    offset = 0
    eval_batch_size = 1024

    no_cuda = False
    no_progress_bar = False

    results_dir = "results"
    results_filename = "1000_20_model.pkl"
    overwrite = True


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), "wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_args(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_model(path, epoch=None):
    from nets.attention_model import AttentionModel

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == ".pt"
            )
        model_filename = os.path.join(path, "epoch-{}.pt".format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, "args.json"))

    model = AttentionModel(
        args["embedding_dim"],
        args["hidden_dim"],
        TSP,
        n_encode_layers=args["n_encode_layers"],
        mask_inner=True,
        mask_logits=True,
        normalization=args["normalization"],
        tanh_clipping=args["tanh_clipping"],
        checkpoint_encoder=args.get("checkpoint_encoder", False),
    )

    load_data = torch.load(
        model_filename, map_location=lambda storage, loc: storage, weights_only=False
    )

    model_state_dict = load_data.get("model", load_data) if isinstance(load_data, dict) else load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(model_state_dict)
    model.load_state_dict(state_dict)

    model.eval()

    return model, args


def eval_dataset(dataset_path, opts):
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset = model.problem.make_dataset(
        filename=dataset_path, num_samples=opts.val_size, offset=opts.offset
    )

    results, total_duration = _eval_dataset(model, dataset, opts, device)
    costs, tours = zip(*results)

    print(
        "Average cost: {} +- {}".format(
            np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))
        )
    )
    print("Total evaluation time: {}".format(timedelta(seconds=int(total_duration))))
    print("Average time per instance: {:.4f}s".format(total_duration / len(costs)))

    os.makedirs(opts.results_dir, exist_ok=True)
    out_file = os.path.join(opts.results_dir, opts.results_filename)

    assert opts.overwrite or not os.path.isfile(out_file), "File already exists!"

    save_dataset(results, out_file)
    print(f"Saved results to: {out_file}")

    return costs, tours, total_duration


def _eval_dataset(model, dataset, opts, device):
    model.to(device)
    model.eval()
    model.set_decode_type("greedy")

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    total_duration = 0.0

    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            costs, _, tours = model(batch, return_pi=True)
        total_duration += time.time() - start

        for tour, cost in zip(tours.cpu().numpy(), costs.cpu().numpy()):
            results.append((cost, tour.tolist()))

    return results, total_duration

if __name__ == "__main__":
    opts = EvalConfig()
    for dataset_path in opts.datasets:
        eval_dataset(dataset_path, opts)