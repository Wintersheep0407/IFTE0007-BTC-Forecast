"""Reproduce slope scenarios and chronological backtest diagnostics offline."""
import csv
import json
import math
import sys
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "experiments/shrinkage_177h"
sys.path.insert(0, str(BASE))
from verify import raw_close, utc

MINUTE = timedelta(minutes=1)
WEEK = timedelta(days=7)
HORIZON = timedelta(hours=177)


def read(path):
    return json.loads(path.read_text())


def score(errors):
    return {"n": len(errors), "mae_usd": mean(map(abs, errors)),
            "rmse_usd": math.sqrt(mean(e * e for e in errors)),
            "max_absolute_error_usd": max(map(abs, errors))}


def ols(xs, ys):
    mx, my = mean(xs), mean(ys)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - beta * mx, beta


def main():
    record_path = ROOT / "results/final_forecast.json"
    final = read(record_path)
    policy = read(ROOT / "config/research_policy.json")
    cutoff, target = utc(final["information_cutoff"]), utc(final["target"])
    assert target - cutoff == HORIZON
    assert (policy["model"], policy["momentum_weight"], policy["availability_lag_minutes"]) == ("7", .25, 1)
    index_path = ROOT / policy["training_reference_file"]
    assert sha256(index_path.read_bytes()).hexdigest() == policy["training_references_sha256"]
    index = read(index_path)
    origins = [utc(s) for s in index["origins"]]
    assert origins == sorted(set(origins))
    needed = {t for o in origins for t in (o - MINUTE, o - MINUTE - WEEK, o + HORIZON)}
    prices = {}
    for ref in index["references"]:
        end = utc(ref["instant"])
        if end in needed:
            manifest = read(ROOT / ref["manifest"])
            assert all(utc(r["retrieved_at"]) <= cutoff for r in manifest["requests"])
            prices[end] = raw_close(ref)
    assert len(prices) == 192 and prices.keys() == needed
    baseline = {o: prices[o - MINUTE] for o in origins}
    momentum = {o: baseline[o] / prices[o - MINUTE - WEEK] - 1 for o in origins}
    labels = {o: prices[o + HORIZON] / baseline[o] - 1 for o in origins}
    assert all(o + HORIZON + MINUTE < cutoff for o in origins)
    alpha, beta = ols([momentum[o] for o in origins], [labels[o] for o in origins])
    assert math.isclose(alpha, final["intercept"], abs_tol=1e-12)
    assert math.isclose(beta, final["slope"], abs_tol=1e-12)
    calc_path = ROOT / final["calculation_file"]
    assert sha256(calc_path.read_bytes()).hexdigest() == final["calculation_sha256"]
    calc = read(calc_path)
    for role, end, price in (
        ("baseline", cutoff - MINUTE, final["baseline_price"]),
        ("factor", cutoff - MINUTE - WEEK, final["seven_day_reference_price"]),
    ):
        src = calc["sources"][role]
        assert sha256((ROOT / src["manifest"]).read_bytes()).hexdigest() == src["sha256"]
        assert raw_close({"instant": end.isoformat(), "manifest": src["manifest"], "price": price}) == price

    factor = final["baseline_price"] / final["seven_day_reference_price"] - 1
    scenarios = []
    for multiple in (.5, 1, 1.5):
        full_return = alpha + multiple * beta * factor
        weighted = .25 * full_return
        price = Decimal(str(final["baseline_price"] * (1 + weighted))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        scenarios.append({"slope_multiplier": multiple, "slope": multiple * beta,
                          "full_return": full_return, "weighted_return": weighted,
                          "price_usd": str(price),
                          "status": "approved_reference" if multiple == 1 else "sensitivity_only"})
    assert scenarios[1]["price_usd"] == final["forecast_usd"]

    with (BASE / "backtest.csv").open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    comparisons = []
    for prior in saved:
        origin = utc(prior["origin"])
        target = origin + HORIZON
        eligible = [o for o in origins if o < origin and o + HORIZON + MINUTE < origin]
        assert len(eligible) >= 20
        assert eligible == list(map(utc, json.loads(prior["training_origins"])))
        assert int(prior["train_n"]) == len(eligible)
        assert utc(prior["target"]) == target and utc(prior["latest_train_target"]) == eligible[-1] + HORIZON
        a, b = ols([momentum[o] for o in eligible], [labels[o] for o in eligible])
        base, actual = baseline[origin], prices[target]
        predicted = base * (1 + .25 * (a + b * momentum[origin]))
        assert math.isclose(predicted, float(prior["shrinkage_prediction"]), abs_tol=1e-8)
        assert actual == float(prior["realised"]) and base == float(prior["baseline"])
        # A separate diagnostic tests disjoint price-to-price training intervals.
        disjoint = []
        for t in reversed(eligible):
            if not disjoint or t + HORIZON < disjoint[-1] - MINUTE:
                disjoint.append(t)
        assert len(disjoint) >= 20
        da, db = ols([momentum[o] for o in disjoint], [labels[o] for o in disjoint])
        diagnostic = base * (1 + .25 * (da + db * momentum[origin]))
        comparisons.append({
            "origin_utc": origin.isoformat(), "target_utc": target.isoformat(),
            "latest_training_target_utc": (eligible[-1] + HORIZON).isoformat(),
            "training_n": len(eligible), "no_change_usd": base, "weighted_ols_usd": predicted,
            "actual_usd": actual, "no_change_signed_error": base - actual,
            "weighted_ols_signed_error": predicted - actual,
            "weighted_ols_absolute_error": abs(predicted - actual),
            "disjoint_training_n": len(disjoint), "disjoint_diagnostic_usd": diagnostic,
            "disjoint_diagnostic_signed_error": diagnostic - actual,
        })
    assert len(comparisons) == 43
    baseline_scores = score([r["no_change_signed_error"] for r in comparisons])
    model_scores = score([r["weighted_ols_signed_error"] for r in comparisons])
    original = read(BASE / "result.json")["scores"]
    for recomputed, stored in ((baseline_scores, original["no_change"]), (model_scores, original["fixed_shrinkage"])):
        assert all(math.isclose(value, stored[key], abs_tol=1e-8) for key, value in recomputed.items())
    worst = max(comparisons, key=lambda row: row["weighted_ols_absolute_error"])
    result = {"approved_forecast_usd": final["forecast_usd"], "slope_sensitivity": scenarios,
              "backtest": {"no_change": baseline_scores, "weighted_ols": model_scores,
                           "disjoint_training_diagnostic": score([r["disjoint_diagnostic_signed_error"] for r in comparisons]),
                           "weighted_ols_wins": sum(abs(r["weighted_ols_signed_error"]) <
                                                    abs(r["no_change_signed_error"]) for r in comparisons),
                           "worst_weighted_ols": worst},
              "training_count": len(origins), "training_candles_verified": len(prices),
              "cutoff_availability_independently_verified": False,
              "official_record_sha256": sha256(record_path.read_bytes()).hexdigest()}
    for name, rows in (("backtest_errors.csv", comparisons), ("slope_sensitivity.csv", scenarios)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
