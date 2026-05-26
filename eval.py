import torch
import os

import numpy as np

from tqdm import tqdm
from utils.functions import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta


class EvalConfig:
   # datasets = ["data/5_128000GTSP20_val_seed1234.pkl"]
   # datasets = ["data/5_10000GTSP20_val_seed1234.pkl"]
   # datasets = ["data/5_400GTSP20_val_seed1234.pkl"]
 #   datasets = ["data/200_20.pkl"]
    datasets = ["data/300_10.pkl"]
    model = "outputs/tsp_20/=10_1ATTENTION20runtestdeportGTSP20_20260522T135527/epoch-95-baseline-update.pt"

   # model = "outputs/tsp_20/=5_1ATTENTION20runtestdeportGTSP20_20260519T123713/epoch-99.pt"
   # model = "outputs/tsp_20/=+2ENCODER_1ATTENTION20runtestdeportGTSP20_20260520T234557/epoch-99.pt"
   # model = "outputs/tsp_20/=SIMPLE5_1ATTENTION20runtestdeportGTSP20_20260520T203814/epoch-99.pt"
   # model = "outputs/tsp_20/Bad=SIMPLE5_1ATTENTION20runtestdeportGTSP20_20260520T185503/epoch-99.pt"

    val_size = 300
    offset = 0
    eval_batch_size = 1024

    no_cuda = False
    no_progress_bar = False

    results_dir = "results"
    results_filename = "300_10_model.pkl"
    overwrite = True


def eval_dataset(dataset_path, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda

    device = torch.device("cuda:0" if use_cuda else "cpu")
    dataset = model.problem.make_dataset(
        filename=dataset_path, num_samples=opts.val_size, offset=opts.offset
    )
    results = _eval_dataset(model, dataset, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(
        *results
    )  # Not really costs since they should be negative

    print(
        "Average cost: {} +- {}".format(
            np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))
        )
    )
    print(
        "Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))
        )
    )
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print(
        "Calculated total duration: {}".format(
            timedelta(seconds=int(np.sum(durations) / parallelism))
        )
    )

    os.makedirs(opts.results_dir, exist_ok=True)

    out_file = os.path.join(opts.results_dir, opts.results_filename)

    assert opts.overwrite or not os.path.isfile(out_file), "File already exists!"

    save_dataset((results, parallelism), out_file)

    print(f"Saved results to: {out_file}")

    return costs, tours, durations


def _eval_dataset(model, dataset, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type("greedy")

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            costs, _, tours = model(batch, return_pi=True)

        duration = time.time() - start

        for tour, cost in zip(tours.cpu().numpy(), costs.cpu().numpy()):
            results.append((cost, tour.tolist(), duration))

    return results


if __name__ == "__main__":
    opts = EvalConfig()

    for dataset_path in opts.datasets:
        eval_dataset(dataset_path, opts)
