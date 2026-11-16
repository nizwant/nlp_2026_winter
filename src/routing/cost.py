"""Cost of a routing policy against annotating everything by hand.

All rates are explicit parameters read from configs/cost.yaml; nothing is assumed silently and a
saving is never taken for granted: a policy that accepts few items can cost more than it saves.

    human time   seconds a person needs for one item (median of the timed sample)
    labour rate  price of one hour of annotation
    GPU time     seconds of wall-clock GPU time per item for the requests the policy makes
    GPU rate     price of one GPU hour = electricity + share of the purchase price
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .data import REPO_ROOT

COST_CONFIG = REPO_ROOT / "configs" / "cost.yaml"


def load_rates(path: Path = COST_CONFIG) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def gpu_rate_per_hour(rates: dict) -> float:
    """Price of one GPU hour: electricity drawn plus the purchase price spread over the lifetime."""
    gpu = rates["gpu"]
    electricity = gpu["power_watts"] / 1000.0 * gpu["electricity_per_kwh"]
    wear = gpu["purchase_price"] / gpu["lifetime_hours"]
    return electricity + wear


def policy_cost(
    n_items: int,
    coverage: float,
    human_seconds_per_item: float,
    gpu_seconds_per_item: float,
    rates: dict,
) -> dict:
    """Hours and money for `n_items` under a policy that accepts a `coverage` share automatically.

    The model runs on every item (it has to, to decide what to accept); people annotate the rest.
    """
    labour_rate = rates["labour_per_hour"]
    human_hours_all = n_items * human_seconds_per_item / 3600.0
    human_hours = n_items * (1.0 - coverage) * human_seconds_per_item / 3600.0
    gpu_hours = n_items * gpu_seconds_per_item / 3600.0
    compute_cost = gpu_hours * gpu_rate_per_hour(rates)
    return {
        "human_hours_without_model": human_hours_all,
        "human_hours": human_hours,
        "human_hours_saved": human_hours_all - human_hours,
        "gpu_hours": gpu_hours,
        "cost_without_model": human_hours_all * labour_rate,
        "cost_human": human_hours * labour_rate,
        "cost_compute": compute_cost,
        "net_saving": (human_hours_all - human_hours) * labour_rate - compute_cost,
        "currency": rates.get("currency", ""),
    }


def break_even_coverage(human_seconds_per_item: float, gpu_seconds_per_item: float, rates: dict) -> float:
    """Coverage above which a policy saves money: compute cost per item / human cost per item."""
    human_cost = human_seconds_per_item / 3600.0 * rates["labour_per_hour"]
    compute_cost = gpu_seconds_per_item / 3600.0 * gpu_rate_per_hour(rates)
    return compute_cost / human_cost if human_cost > 0 else float("inf")
