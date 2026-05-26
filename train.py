import os
import time
import json
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader

from utils.log_utils import log_values
from utils.functions import move_to


def append_jsonl(path, data):
    """
    Appends one JSON object as one line.
    Safe helper for training logs.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def to_float(x):
    """
    Converts tensor/scalar to python float for JSON logging.
    """
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def get_grad_norm_value(grad_norms, index=0):
    """
    grad_norms returned by clip_grad_norms:
        (grad_norms, grad_norms_clipped)

    Each item is a list over optimizer param groups.
    """
    try:
        value = grad_norms[index][0]
        return to_float(value)
    except Exception:
        return None


def validate(model, dataset, opts, epoch=None, log=False):
    print("Validating...")

    start_time = time.time()

    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    std_error = torch.std(cost) / math.sqrt(len(cost))

    print("Validation overall avg_cost: {} +- {}".format(
        avg_cost,
        std_error,
    ))

    duration = time.time() - start_time

    if log and getattr(opts, "log_jsonl", False):
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


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    model.set_decode_type("greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(
            DataLoader(dataset, batch_size=opts.eval_batch_size),
            disable=opts.no_progress_bar,
        )
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping.

    Returns:
        grad_norms, clipped_grad_norms
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2,
        )
        for group in param_groups
    ]

    grad_norms_clipped = [
        min(g_norm, max_norm) for g_norm in grad_norms
    ] if max_norm > 0 else grad_norms

    return grad_norms, grad_norms_clipped


def train_epoch(
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
):
    lr = optimizer.param_groups[0]["lr"]

    print("Start train epoch {}, lr={} for run {}".format(
        epoch,
        lr,
        opts.run_name,
    ))

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if getattr(opts, "log_jsonl", False):
        append_jsonl(opts.log_path, {
            "type": "epoch_start",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "step": step,
            "lr": float(lr),
            "run_name": opts.run_name,
        })

    if not opts.no_tensorboard:
        tb_logger.log_value("learnrate_pg0", lr, step)

    # Generate / wrap training data for each epoch
    training_dataset = baseline.wrap_dataset(train_dataset)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=opts.batch_size,
        num_workers=1,
        shuffle=True,
    )

    # Put model in train mode!
    model.train()
    model.set_decode_type("sampling")

    last_train_metrics = None

    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        last_train_metrics = train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
        )

        step += 1

    epoch_duration = time.time() - start_time

    print("Finished epoch {}, took {} s".format(
        epoch,
        time.strftime("%H:%M:%S", time.gmtime(epoch_duration)),
    ))

    if hasattr(model, "bias_scale"):
        print("bias scale:", model.bias_scale.item())

    avg_reward = validate(model, val_dataset, opts, epoch=epoch, log=True)

    if not opts.no_tensorboard:
        tb_logger.log_value("val_avg_reward", avg_reward, step)

    # Rollout baseline update / candidate comparison happens here.
    # Важно: epoch_callback должен возвращать True/False.
    baseline_updated = baseline.epoch_callback(model, epoch)
    distance_alpha = None
    if baseline_updated is None:
        baseline_updated = False

    save_regular_checkpoint = (
        (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0)
        or epoch == opts.n_epochs + 100
    )

    save_only_updates_after = getattr(opts, "save_only_updates_after_epoch", None)

    if save_only_updates_after is not None and epoch >= save_only_updates_after:
        should_save = bool(baseline_updated)
    else:
        should_save = save_regular_checkpoint

    if should_save:
        print("Saving model and state...")

        if (
            save_only_updates_after is not None
            and epoch >= save_only_updates_after
            and baseline_updated
        ):
            checkpoint_name = "epoch-{}-baseline-update.pt".format(epoch)
        else:
            checkpoint_name = "epoch-{}.pt".format(epoch)

        checkpoint_path = os.path.join(
            opts.save_dir,
            checkpoint_name,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            checkpoint_path,
        )

        if getattr(opts, "log_jsonl", False):
            append_jsonl(opts.log_path, {
                "type": "checkpoint",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "step": step,
                "path": checkpoint_path,
                "baseline_updated": bool(baseline_updated),
                "save_only_updates_after_epoch": save_only_updates_after,
            })

    if getattr(opts, "log_jsonl", False):
        epoch_log = {
            "type": "epoch_end",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "step": step,
            "lr": float(lr),
            "epoch_duration_sec": float(epoch_duration),
            "val_avg_cost": to_float(avg_reward),
            "distance_alpha": float(distance_alpha) if distance_alpha is not None else None,
            "baseline_updated": bool(baseline_updated),
            "checkpoint_saved": bool(should_save),
        }

        if last_train_metrics is not None:
            epoch_log.update({
                "last_train_avg_cost": last_train_metrics.get("train_avg_cost"),
                "last_train_loss": last_train_metrics.get("loss"),
                "last_reinforce_loss": last_train_metrics.get("reinforce_loss"),
                "last_bl_loss": last_train_metrics.get("bl_loss"),
                "last_log_likelihood": last_train_metrics.get("log_likelihood"),
                "last_grad_norm": last_train_metrics.get("grad_norm"),
                "last_grad_norm_clipped": last_train_metrics.get("grad_norm_clipped"),
            })

        append_jsonl(opts.log_path, epoch_log)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
    model,
    optimizer,
    baseline,
    epoch,
    batch_id,
    step,
    batch,
    tb_logger,
    opts,
):
    x, bl_val = baseline.unwrap_batch(batch)

    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Backward + optimization
    optimizer.zero_grad()
    loss.backward()

    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)

    optimizer.step()

    metrics = {
        "train_avg_cost": to_float(cost.mean()),
        "train_std_cost": to_float(cost.std()),
        "loss": to_float(loss),
        "reinforce_loss": to_float(reinforce_loss),
        "bl_loss": to_float(bl_loss) if torch.is_tensor(bl_loss) else float(bl_loss),
        "log_likelihood": to_float(log_likelihood.mean()),
        "grad_norm": get_grad_norm_value(grad_norms, index=0),
        "grad_norm_clipped": get_grad_norm_value(grad_norms, index=1),
    }

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(
            cost,
            grad_norms,
            epoch,
            batch_id,
            step,
            log_likelihood,
            reinforce_loss,
            bl_loss,
            tb_logger,
            opts,
        )

        if getattr(opts, "log_jsonl", False):
            append_jsonl(opts.log_path, {
                "type": "train_batch",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "batch_id": batch_id,
                "step": step,
                **metrics,
            })

    return metrics