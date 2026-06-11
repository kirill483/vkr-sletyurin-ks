#!/usr/bin/env python

import os
import json
import time
import math
import pprint as pp

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from problems.tsp.problem_tsp import TSP
from nets.attention_model import AttentionModel
from reinforce_baselines import (
    ExponentialBaseline,
    RolloutBaseline,
    WarmupBaseline,
    rollout,
    append_jsonl,
    to_float,
    move_to,
)


class Config:
    # -------------------------
    # Data
    # -------------------------
    problem = "tsp"
    graph_size = 20

    batch_size = 512
    epoch_size = 128000
    data_distribution = None

    train_dataset = "data/5_128000GTSP20_val_seed1234.pkl"
    baseline_dataset = "data/5_10000GTSP20_val_seed1234.pkl"

    val_size = 400
    val_dataset = "data/5_400GTSP20_val_seed1234.pkl"
    eval_batch_size = 1024

    # -------------------------
    # Model
    # -------------------------
    embedding_dim = 128
    hidden_dim = 128
    n_encode_layers = 3

    tanh_clipping = 10.0
    normalization = "batch"

    checkpoint_encoder = False

    # -------------------------
    # Training
    # -------------------------
    lr_model = 1e-4
    lr_decay = 1.0

    n_epochs = 100

    seed = 1235
    max_grad_norm = 1.0

    no_cuda = False

    # Baseline: "exponential" / "rollout"
    baseline = "rollout"
    exp_beta = 0.8
    bl_alpha = 0.05

    bl_warmup_epochs = 1

    # -------------------------
    # Saving
    # -------------------------
    output_dir = "outputs"
    run_name = "=sSUDTRYNEWBEST_1ATTENTION20runtestdeportGTSP20"
    checkpoint_epochs = 25
    save_only_updates_after_epoch = 80

    # -------------------------
    # Logging
    # -------------------------
    log_step = 500
    no_progress_bar = True

    log_jsonl = True
    log_filename = "train_log.jsonl"



def get_grad_norm_value(grad_norms, index=0):
    try:
        return to_float(grad_norms[index][0])
    except Exception:
        return None


def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped




def prepare_config(opts):
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"

    opts.save_dir = os.path.join(
        opts.output_dir,
        f"{opts.problem}_{opts.graph_size}",
        opts.run_name,
    )

    opts.log_path = os.path.join(opts.save_dir, opts.log_filename)

    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == "rollout" else 0

    assert opts.epoch_size % opts.batch_size == 0, "epoch_size must be divisible by batch_size"
    assert opts.bl_warmup_epochs == 0 or opts.baseline == "rollout", \
        "Warmup baseline is only supported with rollout baseline"

    return opts


def build_model(opts, problem):
    return AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
    ).to(opts.device)


def build_baseline(opts, model, problem, baseline_dataset):
    if opts.baseline == "exponential":
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == "rollout":
        baseline = RolloutBaseline(model, problem, opts, dataset=baseline_dataset)
    else:
        assert False, f"Unknown baseline: {opts.baseline}"

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    return baseline




def validate(model, dataset, opts, epoch=None, log=False):
    print("Validating...")

    start_time = time.time()
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    std_error = torch.std(cost) / math.sqrt(len(cost))
    duration = time.time() - start_time

    print("Validation overall avg_cost: {} +- {}".format(avg_cost, std_error))

    if log and opts.log_jsonl:
        append_jsonl(opts.log_path, {
            "type": "validation",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "avg_cost": to_float(avg_cost),
            "std_error": to_float(std_error),
            "num_samples": int(len(cost)),
            "duration_sec": float(duration),
        })

    return avg_cost




def train_epoch(model, optimizer, baseline, lr_scheduler, epoch,
                 val_dataset, train_dataset, problem, opts):
    lr = optimizer.param_groups[0]["lr"]
    print("Start train epoch {}, lr={} for run {}".format(epoch, lr, opts.run_name))

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if opts.log_jsonl:
        append_jsonl(opts.log_path, {
            "type": "epoch_start",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "step": step,
            "lr": float(lr),
            "run_name": opts.run_name,
        })

    training_dataset = baseline.wrap_dataset(train_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1, shuffle=True)

    model.train()
    model.set_decode_type("sampling")

    last_train_metrics = None

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        last_train_metrics = train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))))

    avg_reward = validate(model, val_dataset, opts, epoch=epoch, log=True)

    baseline_updated = baseline.epoch_callback(model, epoch)
    if baseline_updated is None:
        baseline_updated = False

    save_only_updates_after = opts.save_only_updates_after_epoch
    if save_only_updates_after is not None and epoch >= save_only_updates_after:
        should_save = bool(baseline_updated)
    else:
        should_save = opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0

    if should_save:
        print("Saving model and state...")

        if save_only_updates_after is not None and epoch >= save_only_updates_after and baseline_updated:
            checkpoint_name = "epoch-{}-baseline-update.pt".format(epoch)
        else:
            checkpoint_name = "epoch-{}.pt".format(epoch)

        checkpoint_path = os.path.join(opts.save_dir, checkpoint_name)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "baseline": baseline.state_dict(),
        }, checkpoint_path)

        if opts.log_jsonl:
            append_jsonl(opts.log_path, {
                "type": "checkpoint",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "step": step,
                "path": checkpoint_path,
                "baseline_updated": bool(baseline_updated),
                "save_only_updates_after_epoch": save_only_updates_after,
            })

    if opts.log_jsonl:
        epoch_log = {
            "type": "epoch_end",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "step": step,
            "lr": float(lr),
            "epoch_duration_sec": float(epoch_duration),
            "val_avg_cost": to_float(avg_reward),
            "baseline_updated": bool(baseline_updated),
            "checkpoint_saved": bool(should_save),
        }

        if last_train_metrics is not None:
            epoch_log.update({
                "last_train_avg_cost": last_train_metrics.get("train_avg_cost"),
                "last_train_loss": last_train_metrics.get("loss"),
                "last_reinforce_loss": last_train_metrics.get("reinforce_loss"),
                "last_log_likelihood": last_train_metrics.get("log_likelihood"),
                "last_grad_norm": last_train_metrics.get("grad_norm"),
                "last_grad_norm_clipped": last_train_metrics.get("grad_norm_clipped"),
            })

        append_jsonl(opts.log_path, epoch_log)

    lr_scheduler.step()


def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, opts):
    x, bl_val = baseline.unwrap_batch(batch)

    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    cost, log_likelihood = model(x)

    if bl_val is None:
        bl_val = baseline.eval(x, cost)

    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss

    optimizer.zero_grad()
    loss.backward()

    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    metrics = {
        "train_avg_cost": to_float(cost.mean()),
        "train_std_cost": to_float(cost.std()),
        "loss": to_float(loss),
        "reinforce_loss": to_float(reinforce_loss),
        "log_likelihood": to_float(log_likelihood.mean()),
        "grad_norm": get_grad_norm_value(grad_norms, index=0),
        "grad_norm_clipped": get_grad_norm_value(grad_norms, index=1),
    }

    if step % int(opts.log_step) == 0:
        grad_norms_raw, grad_norms_clipped = grad_norms
        print("epoch: {}, train_batch_id: {}, avg_cost: {}".format(epoch, batch_id, metrics["train_avg_cost"]))
        print("grad_norm: {}, clipped: {}".format(grad_norms_raw[0], grad_norms_clipped[0]))

        if opts.log_jsonl:
            append_jsonl(opts.log_path, {
                "type": "train_batch",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "batch_id": batch_id,
                "step": step,
                **metrics,
            })

    return metrics




def run(opts):
    pp.pprint(vars(opts))

    torch.manual_seed(opts.seed)
    os.makedirs(opts.save_dir, exist_ok=True)

    with open(os.path.join(opts.save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(opts), f, indent=True, ensure_ascii=False, default=str)

    if opts.log_jsonl:
        with open(opts.log_path, "w", encoding="utf-8") as f:
            pass

        append_jsonl(opts.log_path, {
            "type": "run_start",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": opts.run_name,
            "save_dir": opts.save_dir,
            "problem": opts.problem,
            "graph_size": opts.graph_size,
            "batch_size": opts.batch_size,
            "epoch_size": opts.epoch_size,
            "val_size": opts.val_size,
            "lr_model": opts.lr_model,
            "lr_decay": opts.lr_decay,
            "n_epochs": opts.n_epochs,
            "seed": opts.seed,
            "baseline": opts.baseline,
            "device": str(opts.device),
            "use_cuda": bool(opts.use_cuda),
        })

    problem = TSP

    model = build_model(opts, problem)

    baseline_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=10000,
        filename=opts.baseline_dataset,
        distribution=opts.data_distribution,
    )
    baseline = build_baseline(opts, model, problem, baseline_dataset)

    optimizer = optim.Adam([{"params": model.parameters(), "lr": opts.lr_model}])

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    val_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    train_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.epoch_size,
        filename=opts.train_dataset,
        distribution=opts.data_distribution,
    )

    for epoch in range(opts.n_epochs):
        train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, train_dataset, problem, opts)

    if opts.log_jsonl:
        append_jsonl(opts.log_path, {
            "type": "run_end",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": opts.run_name,
        })


if __name__ == "__main__":
    opts = prepare_config(Config())
    run(opts)