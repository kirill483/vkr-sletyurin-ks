#!/usr/bin/env python

import os
import json
import time
import pprint as pp

# from tensorboard_logger import Logger as TbLogger
import torch
import torch.optim as optim

from train import train_epoch, validate
from reinforce_baselines import (
    NoBaseline,
    ExponentialBaseline,
    RolloutBaseline,
    WarmupBaseline,
)
from nets.attention_model import AttentionModel
from utils.functions import torch_load_cpu, load_problem


class Config:
    # -------------------------
    # Data
    # -------------------------
    # training dataset
    # problem нужен для чекпоинтов и графсайз
    problem = "tsp"
    graph_size = 20

    batch_size = 512
    epoch_size = 128000
    data_distribution = None
    # validation
    val_size = 400
    val_dataset = "data/5_400GTSP20_val_seed1234.pkl"  # можно указать путь, например "data/tsp20_val_seed1234.pkl"
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
    epoch_start = 0

    seed = 1235
    max_grad_norm = 1.0

    no_cuda = False

    # Baseline: None / "exponential" / "rollout"
    baseline = "rollout"
    exp_beta = 0.8
    bl_alpha = 0.05

    # Для rollout baseline обычно 1 warmup epoch
    bl_warmup_epochs = 1

    # -------------------------
    # Saving / loading
    # -------------------------
    # названия как сохранятся чекпоинты и как часто сохраняются
    output_dir = "outputs"
    run_name = "=SIMPLE5_1ATTENTION20runtestdeportGTSP20"
    checkpoint_epochs = 10
    # загрузить готовую модель дообучить
    load_path = None
    resume = None

    eval_only = False

    # -------------------------
    # Logging
    # -------------------------
    log_step = 500
    no_tensorboard = True
    log_dir = "logs"
    no_progress_bar = True


def prepare_config(opts):
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"

    opts.save_dir = os.path.join(
        opts.output_dir,
        f"{opts.problem}_{opts.graph_size}",
        opts.run_name,
    )

    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == "rollout" else 0

    assert (
        opts.epoch_size % opts.batch_size == 0
    ), "epoch_size must be divisible by batch_size"

    assert (
        opts.bl_warmup_epochs == 0 or opts.baseline == "rollout"
    ), "Warmup baseline is only supported with rollout baseline"

    return opts


def build_model(opts, problem):
    model = AttentionModel(
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

    return model


def build_baseline(opts, model, problem, baseline_dataset):
    if opts.baseline == "exponential":
        baseline = ExponentialBaseline(opts.exp_beta)

    elif opts.baseline == "rollout":
        baseline = RolloutBaseline(model, problem, opts, dataset=baseline_dataset)

    else:
        assert opts.baseline is None, f"Unknown baseline: {opts.baseline}"
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(
            baseline,
            opts.bl_warmup_epochs,
            warmup_exp_beta=opts.exp_beta,
        )

    return baseline


def run(opts):
    pp.pprint(vars(opts))

    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir, exist_ok=True)

    # Сохраняем config рядом с checkpoint'ами.
    # Он нужен потом для load_model/eval.
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True, default=str)

    problem = load_problem(opts.problem)

    load_data = {}

    assert (
        opts.load_path is None or opts.resume is None
    ), "Only one of load_path and resume can be given"

    load_path = opts.load_path if opts.load_path is not None else opts.resume

    if load_path is not None:
        print(f"[*] Loading data from {load_path}")
        load_data = torch_load_cpu(load_path)

    model = build_model(opts, problem)

    # Загружаем веса модели, если продолжаем обучение или load_path задан
    if "model" in load_data:
        model.load_state_dict(
            {
                **model.state_dict(),
                **load_data["model"],
            }
        )

    baseline_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=10000,
        filename="data/5_10000GTSP20_val_seed1234.pkl",
        distribution=opts.data_distribution,
    )

    baseline = build_baseline(opts, model, problem, baseline_dataset)

    if "baseline" in load_data:
        baseline.load_state_dict(load_data["baseline"])

    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": opts.lr_model}]
        + (
            [{"params": baseline.get_learnable_parameters(), "lr": opts.lr_model}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    if "optimizer" in load_data:
        optimizer.load_state_dict(load_data["optimizer"])

        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(opts.device)

    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: opts.lr_decay**epoch,
    )

    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(
            os.path.join(
                opts.log_dir, f"{opts.problem}_{opts.graph_size}", opts.run_name
            )
        )

    val_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    train_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.epoch_size,
        filename="data/5_128000GTSP20_val_seed1234.pkl",
        distribution=opts.data_distribution,
    )

    if opts.resume is not None:
        epoch_resume = int(
            os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1]
        )

        torch.set_rng_state(load_data["rng_state"])

        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])

        baseline.epoch_callback(model, epoch_resume)

        print(f"Resuming after epoch {epoch_resume}")
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
        return

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_dataset,
            train_dataset,
            problem,
            tb_logger,
            opts,
        )


if __name__ == "__main__":
    opts = prepare_config(Config())
    run(opts)
