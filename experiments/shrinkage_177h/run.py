"""Evaluate a fixed shrinkage weight on a later 177-hour cohort."""
import argparse
import json
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd

from forecasting.data import HORIZON, UTC, clean, collect, reference_price
from forecasting.finalise import verified_history
from forecasting.model import backtest, metrics, samples

ROOT = Path(__file__).resolve().parent
DEV = Path("results/backtest_lag1_7d.csv")
HISTORY = Path("data/clean/research_lag_references.json")
WEIGHTS = (0, 0.25, 0.5, 0.75, 1)
PROTOCOL = ROOT / "protocol.json"
START = datetime(2026, 1, 9, 12, tzinfo=UTC)
END = datetime(2026, 10, 30, 12, tzinfo=UTC)
LAG = timedelta(minutes=1)
WEEK = timedelta(days=7)


def predict_with_weight(frame, weight):
    return frame["baseline"] + weight * (frame["prediction"] - frame["baseline"])


def score(frame, weight):
    scored = frame.assign(adjusted=predict_with_weight(frame, weight))
    return {
        **metrics(scored, "adjusted"),
        "max_absolute_error_usd": float((scored["adjusted"] - scored["realised"]).abs().max()),
    }


def select():
    dev = pd.read_csv(DEV)
    dev = dev[dev["split"] == "development"]
    if len(dev) != 20 or pd.to_datetime(dev["target"], utc=True).max() >= datetime(2025, 10, 24, 12, tzinfo=UTC):
        raise ValueError("Wrong development slice")
    scores = [{"weight": weight, **score(dev, weight)} for weight in WEIGHTS]
    chosen = min(scores, key=lambda row: (row["mae_usd"], row["weight"]))["weight"]
    protocol = {
        "locked_at": datetime.now(UTC).isoformat(),
        "development_file": str(DEV),
        "development_sha256": sha256(DEV.read_bytes()).hexdigest(),
        "rule": "Lowest MAE across the 20 time-eligible development errors, ties to smaller weight",
        "weight": chosen,
        "development_scores": scores,
        "evaluation_start": START.isoformat(),
        "evaluation_end": END.isoformat(),
        "horizon_hours": HORIZON.total_seconds() / 3600,
        "baseline_delay_minutes": 1,
        "fit": "Expanding seven-day OLS, with only training targets strictly before each origin",
        "evaluation": "Compare the frozen weight to w=0 and w=1; do not reselect from these scores",
        "scope": "Retrospective. Some raw Friday 21:00 candles were previously collected for a different horizon. The 2026 177-hour errors are not used to choose the weight.",
    }
    with PROTOCOL.open("x") as handle:
        json.dump(protocol, handle, indent=2)
        handle.write("\n")
    print(json.dumps(protocol, indent=2))


def get_reference(end, allow_network):
    key = end.strftime("%Y%m%dT%H%M")
    folder = ROOT / "raw" / key
    saved = folder / "manifest.json"
    prior = ROOT.parent / "nov13_short_horizon" / "raw" / key / "manifest.json"
    manifest = saved if saved.exists() or not prior.exists() else prior
    if not manifest.exists():
        if not allow_network:
            raise FileNotFoundError(manifest)
        collect(end - LAG, end, 60, folder)
    meta = json.loads(manifest.read_text())
    if (meta["product"], meta["venue"], meta["granularity"]) != ("BTC-USD", "Coinbase Exchange", 60):
        raise ValueError(f"Unexpected market at {end}")
    output = ROOT / "clean" / (key + ".csv")
    if clean(manifest, output)["missing_count"]:
        raise ValueError(f"Missing candle at {end}")
    return {"instant": end.isoformat(), "price": reference_price(output, end),
            "manifest": str(manifest.relative_to(Path.cwd()))}


def evaluate(allow_network):
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(DEV.read_bytes()).hexdigest() != protocol["development_sha256"]:
        raise ValueError("Development data changed after selection")
    old = verified_history(HISTORY)
    origins = [START + i * WEEK for i in range((END - START) // WEEK + 1)]
    required = sorted({t - LAG for t in origins + [START - WEEK]} | {t + HORIZON for t in origins})
    new = []
    for index, end in enumerate(required, start=1):
        new.append(get_reference(end, allow_network))
        if allow_network and (index % 20 == 0 or index == len(required)):
            print(f"Verified {index}/{len(required)} exact references", flush=True)
    joined = {
        "origins": old["origins"] + [t.isoformat() for t in origins],
        "references": old["references"] + new,
    }
    inputs = ROOT / "references.json"
    inputs.write_text(json.dumps(joined, indent=2) + "\n")
    frame = backtest(samples(inputs, 7, 1))
    evaluation = frame[pd.to_datetime(frame["origin"], utc=True) >= START].copy()
    if len(evaluation) != len(origins) or not (pd.to_datetime(evaluation["latest_train_target"], utc=True) < pd.to_datetime(evaluation["origin"], utc=True)).all():
        raise ValueError("Incomplete evaluation or an ineligible training label")
    weight = protocol["weight"]
    evaluation["shrinkage_prediction"] = predict_with_weight(evaluation, weight)
    evaluation.to_csv(ROOT / "backtest.csv", index=False)
    summary = {
        "weight_selected_before_evaluation": weight,
        "evaluation": {"start": START.isoformat(), "end": END.isoformat(),
                       "last_target": (END + HORIZON).isoformat()},
        "scores": {"no_change": score(evaluation, 0), "fixed_shrinkage": score(evaluation, weight),
                   "full_momentum": score(evaluation, 1)},
        "scope": protocol["scope"],
    }
    (ROOT / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("select", "evaluate"))
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.phase == "select":
        select()
    else:
        evaluate(args.collect)
