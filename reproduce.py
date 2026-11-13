"""Reproduce the approved BTC-USD forecast from frozen Coinbase responses."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
MINUTE = timedelta(minutes=1)
WEEK = timedelta(days=7)
HORIZON = timedelta(hours=177)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return json.loads((ROOT / name).read_bytes())


def hash_file(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def stamp(value):
    time = datetime.fromisoformat(value)
    require(time.utcoffset() == timedelta(0), f"Expected UTC timestamp: {value}")
    return time.astimezone(timezone.utc)


def reproduce():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(hash_file(name) == digest, f"Hash mismatch: {name}")
    final = read("final_forecast.json")
    snapshot = read("data/snapshot.json")
    policy = read("data/research_policy.json")
    pinned = read("data/pinned_reference_index.json")
    approval = read("evidence/numerical_approval.json")
    require(hash_file("data/pinned_reference_index.json") == policy["training_references_sha256"],
            "Training index differs from pre-cutoff policy")
    require(hash_file("evidence/cutoff_recheck/stdout.txt") == approval["calculation_sha256"],
            "Approval calculation hash mismatch")
    calculation = read("evidence/cutoff_recheck/stdout.txt")
    require(hash_file("data/research_policy.json") == calculation["policy_sha256"], "Policy hash mismatch")
    cutoff = stamp(final["information_cutoff_utc"])
    target = stamp(final["target_utc"])
    require(cutoff == stamp("2026-11-13T12:00:00+00:00") and
            target == stamp("2026-11-20T21:00:00+00:00"), "Wrong cutoff or target")
    require(stamp(final["forecast_origin_utc"]) == cutoff and target - cutoff == HORIZON and
            stamp(final["price_reference"]["candle_start_utc"]) == target - MINUTE and
            stamp(policy["recorded_at"]) < cutoff and stamp(policy["cutoff"]) == cutoff and
            stamp(policy["target"]) == target and stamp(approval["target"]) == target and
            stamp(calculation["target"]) == target and stamp(calculation["cutoff"]) == cutoff,
            "Policy, approval or scoring boundary mismatch")
    require((policy["market"], policy["venue"], policy["model"], policy["momentum_weight"],
             policy["lookback_days"], policy["availability_lag_minutes"]) ==
            ("BTC-USD", "Coinbase Exchange", "7", 0.25, 7, 1), "Unexpected method")

    def close(entry, end, late_expected):
        folder = Path(entry["manifest"]).parent
        manifest = read(entry["manifest"])
        require((manifest["product"], manifest["venue"], manifest["granularity"]) ==
                ("BTC-USD", "Coinbase Exchange", 60), "Wrong candle source")
        require(stamp(manifest["start"]) == end - MINUTE and stamp(manifest["end"]) == end and
                len(manifest["requests"]) == 1, "Wrong candle interval")
        request = manifest["requests"][0]
        url = urlsplit(request["url"])
        params = parse_qs(url.query)
        require((url.scheme, url.netloc, url.path) ==
                ("https", "api.exchange.coinbase.com", "/products/BTC-USD/candles") and
                stamp(params["start"][0]) == end - MINUTE and stamp(params["end"][0]) == end and
                params["granularity"] == ["60"] and request["status_code"] == 200, "Wrong API request")
        received = stamp(request["retrieved_at"])
        require(end <= received and (received > cutoff) == late_expected, "Unexpected receipt timing")
        payload = folder / request["file"]
        require(hash_file(payload) == request["sha256"], "Raw response hash mismatch")
        rows = [row for row in read(payload) if row[0] == int((end - MINUTE).timestamp())]
        require(len(rows) == 1, "Missing or duplicated exact candle")
        _, low, high, opening, price, volume = rows[0]
        require(all(math.isfinite(v) for v in rows[0]) and
                0 < low <= opening <= high and low <= price <= high and volume >= 0, "Invalid OHLCV")
        return price

    origins = [stamp(value) for value in pinned["origins"]]
    require(origins == sorted(set(origins)) and len(origins) == 95, "Invalid training origins")
    indexed = {stamp(ref["instant"]): ref for ref in pinned["references"]}
    needed = {t for origin in origins for t in (origin - MINUTE, origin - MINUTE - WEEK, origin + HORIZON)}
    refs = {stamp(entry["instant"]): entry for entry in snapshot["training_candles"]}
    require(len(needed) == len(refs) == 192 and needed == refs.keys() and
            len(snapshot["training_candles"]) == 192, "Incorrect exact training candle set")
    prices = {}
    for end, entry in refs.items():
        prices[end] = close(entry, end, False)
        require(prices[end] == indexed[end]["price"], "Raw close disagrees with pinned index")
    features, returns = [], []
    for origin in origins:
        label = origin + HORIZON
        require(label + MINUTE < cutoff, "Training outcome was not mature at cutoff")
        base = prices[origin - MINUTE]
        features.append(base / prices[origin - MINUTE - WEEK] - 1)
        returns.append(prices[label] / base - 1)

    end = cutoff - MINUTE
    baseline = close(snapshot["forecast_candles"]["baseline"], end, True)
    previous = close(snapshot["forecast_candles"]["factor"], end - WEEK, True)
    require(baseline == calculation["prices"]["baseline"] and previous == calculation["prices"]["factor"],
            "Forecast inputs disagree with approved run")
    x_mean, y_mean = mean(features), mean(returns)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(features, returns)) / sum(
        (x - x_mean) ** 2 for x in features
    )
    intercept = y_mean - slope * x_mean
    momentum = baseline / previous - 1
    raw_return = intercept + slope * momentum
    adjusted_return = policy["momentum_weight"] * raw_return
    value = str(Decimal(str(baseline * (1 + adjusted_return))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    for field, computed in (("intercept", intercept), ("slope", slope), ("momentum", momentum),
                            ("unadjusted_return", raw_return), ("adjusted_return", adjusted_return)):
        require(math.isclose(calculation[field], computed, rel_tol=1e-10, abs_tol=1e-12),
                f"Independent fit mismatch: {field}")
    require(value == final["forecast_value_usd"] == approval["forecast_usd"] == calculation["estimate_usd"],
            "Rounded forecast mismatch")
    print(json.dumps({"status": "passed", "market": final["market"], "unit": final["unit"],
                      "forecast_value_usd": value, "forecast_origin_utc": cutoff.isoformat(),
                      "target_utc": target.isoformat(), "training_observations": len(origins),
                      "historical_candles_verified": len(prices), "forecast_candles_verified": 2,
                      "latest_training_target_utc": (origins[-1] + HORIZON).isoformat(),
                      "intercept": intercept, "slope": slope, "momentum": momentum,
                      "unadjusted_return": raw_return, "adjusted_return": adjusted_return,
                      "cutoff_availability_independently_verified": False}, indent=2))


if __name__ == "__main__":
    reproduce()
