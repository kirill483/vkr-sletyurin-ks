import json
import time
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import ttest_rel


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def append_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def rollout(model, dataset, opts):
    model.set_decode_type("greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def epoch_callback(self, model, epoch):
        return False

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"

        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        if self.alpha == 1:
            return self.baseline.eval(x, c)

        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)

        v = self.baseline.eval(x, c)
        vw = self.warmup_baseline.eval(x, c)

        # Convex combination of the two baselines
        return self.alpha * v + (1 - self.alpha) * vw

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner baseline
        baseline_updated = self.baseline.epoch_callback(model, epoch)

        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

            if getattr(self.baseline.opts, "log_jsonl", False):
                append_jsonl(self.baseline.opts.log_path, {
                    "type": "warmup_baseline",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": int(epoch),
                    "alpha": float(self.alpha),
                })
        return baseline_updated

    def state_dict(self):
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        self.baseline.load_state_dict(state_dict)


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * c.mean()

        self.v = v.detach()  
        return self.v

    def state_dict(self):
        return {
            "v": self.v,
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict["v"]


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, opts, epoch=0, dataset=None):
        super(Baseline, self).__init__()

        self.problem = problem
        self.opts = opts

        
        self.opts.val_size = 10000

        self._update_model(model, epoch, dataset=dataset)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)

        if dataset is not None:
            self.dataset = dataset
        elif not hasattr(self, "dataset"):
            self.dataset = self.problem.make_dataset(
                filename="data/10000GTSP5_val_seed1234.pkl",
                size=self.opts.graph_size,
                num_samples=self.opts.val_size,
                distribution=self.opts.data_distribution,
            )
        # If self.dataset already exists, reuse it.

        print("Evaluating baseline model on evaluation dataset")

        start_time = time.time()

        self.bl_vals = rollout(
            self.model,
            self.dataset,
            self.opts,
        ).cpu().numpy()

        duration = time.time() - start_time

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

        if getattr(self.opts, "log_jsonl", False):
            append_jsonl(self.opts.log_path, {
                "type": "rollout_baseline_model_update",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": int(epoch),
                "baseline_mean": float(self.mean),
                "num_samples": int(len(self.bl_vals)),
                "duration_sec": float(duration),
            })

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")

        start_time = time.time()

        baseline_values = rollout(self.model, dataset, self.opts).view(-1, 1)

        duration = time.time() - start_time

        if getattr(self.opts, "log_jsonl", False):
            append_jsonl(self.opts.log_path, {
                "type": "rollout_baseline_wrap_dataset",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "baseline_epoch": int(self.epoch),
                "baseline_mean_eval_dataset": float(self.mean),
                "dataset_size": int(len(dataset)),
                "wrapped_baseline_mean": float(baseline_values.mean().item()),
                "duration_sec": float(duration),
            })

        
        return BaselineDataset(dataset, baseline_values)

    def unwrap_batch(self, batch):
        return batch["data"], batch["baseline"].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # Use no_grad for efficient inference.
        # Single batch, so we do not use rollout function.
        with torch.no_grad():
            v, _ = self.model(x)
        return v

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the candidate model and
        replaces the baseline model if the candidate is significantly improved.
        """
        print("Evaluating candidate model on evaluation dataset")

        start_time = time.time()

        candidate_vals = rollout(
            model,
            self.dataset,
            self.opts,
        ).cpu().numpy()

        duration = time.time() - start_time

        candidate_mean = candidate_vals.mean()

        old_baseline_epoch = self.epoch
        old_baseline_mean = self.mean

        difference = candidate_mean - old_baseline_mean

        print(
            "Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
                epoch,
                candidate_mean,
                old_baseline_epoch,
                old_baseline_mean,
                difference,
            )
        )

        p_val = None
        t_stat = None
        baseline_updated = False

        if difference < 0:
            # Calc p-value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            t_stat = t
            p_val = p / 2  # one-sided

            assert t < 0, "T-statistic should be negative"

            print("p-value: {}".format(p_val))

            if p_val < self.opts.bl_alpha:
                print("Update baseline")
                baseline_updated = True
                self._update_model(model, epoch)

        if getattr(self.opts, "log_jsonl", False):
            append_jsonl(self.opts.log_path, {
                "type": "rollout_baseline",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": int(epoch),

                "candidate_mean": float(candidate_mean),

                "baseline_epoch_before": int(old_baseline_epoch),
                "baseline_mean_before": float(old_baseline_mean),

                "difference": float(difference),
                "t_stat": float(t_stat) if t_stat is not None else None,
                "p_value": float(p_val) if p_val is not None else None,
                "bl_alpha": float(self.opts.bl_alpha),
                "baseline_updated": bool(baseline_updated),

                "baseline_epoch_after": int(self.epoch),
                "baseline_mean_after": float(self.mean),

                "num_samples": int(len(candidate_vals)),
                "duration_sec": float(duration),
            })
        return baseline_updated

    def state_dict(self):
        return {
            "model": self.model,
            "dataset": self.dataset,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        # Make it work whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        load_model.load_state_dict(state_dict["model"].state_dict())
        self._update_model(load_model, state_dict["epoch"], state_dict["dataset"])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline

        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, item):
        return {
            "data": self.dataset[item],
            "baseline": self.baseline[item],
        }

    def __len__(self):
        return len(self.dataset)