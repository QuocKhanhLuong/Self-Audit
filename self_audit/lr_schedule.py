"""Opt-in ReduceLROnPlateau learning rate scheduler for supervised TRAIN loss."""
import copy
import numpy as np
import torch


DEFAULT_PLATEAU = {
    "name": "train_plateau",
    "interval_steps": 100,
    "factor": 0.5,
    "patience": 5,
    "threshold": 0.0001,
    "threshold_mode": "rel",
    "cooldown": 0,
    "min_lr_factor": 0.01,
    "eps": 1e-8,
}


def validate_lr_scheduler_config(config):
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("lr_scheduler must be a dict or None")
    c = copy.deepcopy(config)
    name = c.get("name")
    if name != "train_plateau":
        raise ValueError(f"unknown lr_scheduler name: {name}")
    required_keys = {"name", "interval_steps", "factor", "patience", "threshold",
                     "threshold_mode", "cooldown", "min_lr_factor", "eps"}
    if set(c.keys()) != required_keys:
        raise ValueError(f"exact lr_scheduler keys required: {sorted(required_keys)}")

    for k in ("interval_steps", "patience", "cooldown"):
        v = c[k]
        if type(v) is not int or v < (1 if k == "interval_steps" else 0):
            raise ValueError(f"invalid lr_scheduler {k}: {v}")

    for k in ("factor", "threshold", "min_lr_factor", "eps"):
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f"invalid lr_scheduler {k}: {v}")

    if not (0.0 < c["factor"] < 1.0):
        raise ValueError("lr_scheduler factor must be in (0, 1)")
    if c["threshold"] < 0:
        raise ValueError("lr_scheduler threshold must be non-negative")
    if c["threshold_mode"] not in {"rel", "abs"}:
        raise ValueError("lr_scheduler threshold_mode must be 'rel' or 'abs'")
    if not (0.0 < c["min_lr_factor"] <= 1.0):
        raise ValueError("lr_scheduler min_lr_factor must be in (0, 1]")
    if c["eps"] <= 0:
        raise ValueError("lr_scheduler eps must be positive")
    return c


def validate_scheduler_state_dict(state_dict, config=None, step=None, warmup_steps=None,
                                  initial_body_lr=None, initial_head_lr=None,
                                  payload_optimizers=None):
    if not isinstance(state_dict, dict):
        raise ValueError("lr_scheduler state must be a dict")
    exact_keys = {"body", "head", "metric_sum", "metric_count", "num_updates"}
    if set(state_dict.keys()) != exact_keys:
        raise ValueError(f"lr_scheduler state keys mismatch: expected {sorted(exact_keys)}, got {sorted(state_dict.keys())}")

    num_updates = state_dict["num_updates"]
    metric_count = state_dict["metric_count"]
    metric_sum = state_dict["metric_sum"]

    if type(num_updates) is not int or num_updates < 0:
        raise ValueError(f"invalid num_updates in lr_scheduler state: {num_updates}")
    if type(metric_count) is not int or metric_count < 0:
        raise ValueError(f"invalid metric_count in lr_scheduler state: {metric_count}")
    if (isinstance(metric_sum, bool) or
            not isinstance(metric_sum, (int, float)) or
            not np.isfinite(metric_sum) or
            metric_sum < 0):
        raise ValueError(f"invalid metric_sum in lr_scheduler state: {metric_sum}")

    # Zero sum for empty window
    if metric_count == 0 and metric_sum != 0.0:
        raise ValueError("metric_sum must be 0.0 when metric_count is 0")

    if step is not None and warmup_steps is not None and config is not None:
        interval_steps = config["interval_steps"]
        if step <= warmup_steps:
            expected_updates = 0
            expected_metric_count = 0
        else:
            accum_steps = step - warmup_steps
            expected_updates = accum_steps // interval_steps
            expected_metric_count = accum_steps % interval_steps
        if num_updates != expected_updates:
            raise ValueError(f"lr_scheduler num_updates ({num_updates}) inconsistent with step ({step}), warmup ({warmup_steps}), interval ({interval_steps})")
        if metric_count != expected_metric_count:
            raise ValueError(f"lr_scheduler metric_count ({metric_count}) inconsistent with step ({step}), warmup ({warmup_steps}), interval ({interval_steps})")

    plateau_keys = {'_last_lr', 'best', 'cooldown', 'cooldown_counter', 'default_min_lr',
                    'eps', 'factor', 'last_epoch', 'min_lrs', 'mode', 'mode_worse',
                    'num_bad_epochs', 'patience', 'threshold', 'threshold_mode'}

    configs_and_opts = [
        ("body", state_dict["body"], initial_body_lr, payload_optimizers[0] if payload_optimizers else None),
        ("head", state_dict["head"], initial_head_lr, payload_optimizers[1] if payload_optimizers else None),
    ]

    for name, sub, init_lr, opt_payload in configs_and_opts:
        if not isinstance(sub, dict):
            raise ValueError(f"lr_scheduler {name} state must be a dict")
        if set(sub.keys()) != plateau_keys:
            raise ValueError(f"lr_scheduler {name} keys mismatch: expected {sorted(plateau_keys)}, got {sorted(sub.keys())}")

        # Typed integer counters
        if type(sub["last_epoch"]) is not int or sub["last_epoch"] < 0:
            raise ValueError(f"invalid last_epoch in {name}: {sub['last_epoch']}")
        if sub["last_epoch"] != num_updates:
            raise ValueError(f"lr_scheduler {name} last_epoch ({sub['last_epoch']}) != num_updates ({num_updates})")
        if type(sub["num_bad_epochs"]) is not int or sub["num_bad_epochs"] < 0:
            raise ValueError(f"invalid num_bad_epochs in {name}: {sub['num_bad_epochs']}")
        if type(sub["cooldown_counter"]) is not int or sub["cooldown_counter"] < 0:
            raise ValueError(f"invalid cooldown_counter in {name}: {sub['cooldown_counter']}")

        # mode_worse must be +inf
        if sub["mode_worse"] != float("inf"):
            raise ValueError(f"lr_scheduler {name} mode_worse must be +inf, got {sub['mode_worse']}")

        # best: exactly +inf before observation; finite float after
        if np.isnan(sub["best"]):
            raise ValueError(f"lr_scheduler {name} best is NaN")
        if num_updates == 0:
            if sub["best"] != float("inf"):
                raise ValueError(f"lr_scheduler {name} best must be exactly +inf before first update, got {sub['best']}")
        else:
            if not np.isfinite(sub["best"]) or np.isinf(sub["best"]):
                raise ValueError(f"lr_scheduler {name} best must be finite after {num_updates} updates")

        # _last_lr: list of length 1 containing finite positive float
        if not isinstance(sub["_last_lr"], list) or len(sub["_last_lr"]) != 1:
            raise ValueError(f"lr_scheduler {name} _last_lr must be a list of length 1")
        lr_val = sub["_last_lr"][0]
        if isinstance(lr_val, bool) or not isinstance(lr_val, (int, float)) or not np.isfinite(lr_val) or lr_val <= 0:
            raise ValueError(f"lr_scheduler {name} _last_lr must contain finite positive float, got {lr_val}")

        if config is not None:
            if sub["factor"] != config["factor"]:
                raise ValueError(f"lr_scheduler {name} factor changed: {sub['factor']} != {config['factor']}")
            if sub["patience"] != config["patience"]:
                raise ValueError(f"lr_scheduler {name} patience changed: {sub['patience']} != {config['patience']}")
            if sub["threshold"] != config["threshold"]:
                raise ValueError(f"lr_scheduler {name} threshold changed: {sub['threshold']} != {config['threshold']}")
            if sub["threshold_mode"] != config["threshold_mode"]:
                raise ValueError(f"lr_scheduler {name} threshold_mode changed")
            if sub["mode"] != "min":
                raise ValueError(f"lr_scheduler {name} mode changed")
            if sub["cooldown"] != config["cooldown"]:
                raise ValueError(f"lr_scheduler {name} cooldown changed")
            if sub["eps"] != config["eps"]:
                raise ValueError(f"lr_scheduler {name} eps changed")
            if init_lr is not None:
                expected_min = init_lr * config["min_lr_factor"]
                if len(sub["min_lrs"]) != 1 or abs(sub["min_lrs"][0] - expected_min) > 1e-12:
                    raise ValueError(f"lr_scheduler {name} min_lrs changed")
                if abs(sub["default_min_lr"] - expected_min) > 1e-12:
                    raise ValueError(f"lr_scheduler {name} default_min_lr ({sub['default_min_lr']}) != expected ({expected_min})")
                # LR bounds: min_lr <= lr_val <= initial_lr
                if lr_val < expected_min - 1e-12 or lr_val > init_lr + 1e-12:
                    raise ValueError(f"lr_scheduler {name} _last_lr ({lr_val}) outside [min_lr={expected_min}, initial_lr={init_lr}] bounds")

        if opt_payload is not None:
            opt_lr = opt_payload["param_groups"][0]["lr"]
            sched_last_lr = sub["_last_lr"][0]
            if abs(opt_lr - sched_last_lr) > 1e-12:
                raise ValueError(f"mismatched LR between {name} optimizer ({opt_lr}) and scheduler ({sched_last_lr})")


class TrainPlateauScheduler:
    def __init__(self, optimizers, config, initial_body_lr, initial_head_lr):
        self.config = validate_lr_scheduler_config(config)
        self.optimizers = optimizers
        self.interval_steps = self.config["interval_steps"]
        min_body_lr = initial_body_lr * self.config["min_lr_factor"]
        min_head_lr = initial_head_lr * self.config["min_lr_factor"]
        self.sched_body = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[0],
            mode="min",
            factor=self.config["factor"],
            patience=self.config["patience"],
            threshold=self.config["threshold"],
            threshold_mode=self.config["threshold_mode"],
            cooldown=self.config["cooldown"],
            min_lr=min_body_lr,
            eps=self.config["eps"],
        )
        self.sched_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[1],
            mode="min",
            factor=self.config["factor"],
            patience=self.config["patience"],
            threshold=self.config["threshold"],
            threshold_mode=self.config["threshold_mode"],
            cooldown=self.config["cooldown"],
            min_lr=min_head_lr,
            eps=self.config["eps"],
        )
        self.metric_sum = 0.0
        self.metric_count = 0
        self.num_updates = 0

    def observe(self, loss_value, is_warmup):
        if is_warmup:
            return
        self.metric_sum += float(loss_value)
        self.metric_count += 1

    def step_if_interval_complete(self):
        if self.metric_count >= self.interval_steps:
            monitored_metric = self.metric_sum / self.metric_count
            body_lr_before = float(self.optimizers[0].param_groups[0]["lr"])
            self.sched_body.step(monitored_metric)
            self.sched_head.step(monitored_metric)
            body_lr_after = float(self.optimizers[0].param_groups[0]["lr"])
            lr_reduced = body_lr_after < body_lr_before
            self.metric_sum = 0.0
            self.metric_count = 0
            self.num_updates += 1
            return True, float(monitored_metric), lr_reduced
        return False, None, False

    def state_dict(self):
        return {
            "body": self.sched_body.state_dict(),
            "head": self.sched_head.state_dict(),
            "metric_sum": float(self.metric_sum),
            "metric_count": int(self.metric_count),
            "num_updates": int(self.num_updates),
        }

    def manifest_metadata(self):
        best_val = self.sched_body.best
        return {
            "name": self.config["name"],
            "num_updates": int(self.num_updates),
            "metric_count": int(self.metric_count),
            "metric_sum": float(self.metric_sum),
            "current_mean_metric": float(self.metric_sum / self.metric_count) if self.metric_count > 0 else None,
            "body_lr": float(self.optimizers[0].param_groups[0]["lr"]),
            "head_lr": float(self.optimizers[1].param_groups[0]["lr"]),
            "best": float(best_val) if np.isfinite(best_val) else None,
            "num_bad_epochs": int(self.sched_body.num_bad_epochs),
            "cooldown_counter": int(self.sched_body.cooldown_counter),
        }

    def load_state_dict(self, state_dict):
        self.sched_body.load_state_dict(state_dict["body"])
        self.sched_head.load_state_dict(state_dict["head"])
        self.metric_sum = float(state_dict["metric_sum"])
        self.metric_count = int(state_dict["metric_count"])
        self.num_updates = int(state_dict["num_updates"])
