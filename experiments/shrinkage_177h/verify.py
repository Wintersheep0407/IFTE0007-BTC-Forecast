"""Independently replay the fixed-weight experiment from saved Coinbase responses."""
import csv
import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean
from urllib.parse import parse_qs, urlparse

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = EXPERIMENT.parents[1]
HORIZON = timedelta(hours=177)
MINUTE = timedelta(minutes=1)
WEEK = timedelta(days=7)
WEIGHTS = (0, 0.25, 0.5, 0.75, 1)


def utc(value):
    stamp = datetime.fromisoformat(value)
    assert stamp.utcoffset() == timedelta(0), f"Non-UTC timestamp: {value}"
    return stamp


def raw_close(reference):
    """Check provenance and recover the last fully closed candle independently."""
    end = utc(reference["instant"])
    start = end - MINUTE
    path = PROJECT / reference["manifest"]
    manifest = json.loads(path.read_text())
    assert (manifest["product"], manifest["venue"], manifest["granularity"]) == (
        "BTC-USD", "Coinbase Exchange", 60
    )
    assert (utc(manifest["start"]), utc(manifest["end"])) == (start, end)
    closes = []
    for request in manifest["requests"]:
        url = urlparse(request["url"])
        assert (url.scheme, url.netloc, url.path) == (
            "https", "api.exchange.coinbase.com", "/products/BTC-USD/candles"
        )
        query = parse_qs(url.query)
        assert (utc(query["start"][0]), utc(query["end"][0]), query["granularity"]) == (
            start, end, ["60"]
        )
        assert request["status_code"] == 200 and utc(request["retrieved_at"]) >= end
        payload = (path.parent / request["file"]).read_bytes()
        assert sha256(payload).hexdigest() == request["sha256"], path
        for epoch, low, high, opening, close, volume in json.loads(payload):
            if datetime.fromtimestamp(epoch, timezone.utc) == start:
                assert all(isfinite(v) for v in (low, high, opening, close, volume))
                assert 0 < low <= opening <= high and low <= close <= high and volume >= 0
                closes.append(close)
    assert closes == [reference["price"]], f"Missing, duplicate or inconsistent raw close at {end}"
    return closes[0]


def scores(pairs):
    errors = [abs(prediction - actual) for prediction, actual in pairs]
    return {
        "n": len(errors),
        "mae_usd": mean(errors),
        "rmse_usd": sqrt(mean(error * error for error in errors)),
        "mape_percent": 100 * mean(error / actual for error, (_, actual) in zip(errors, pairs)),
        "max_absolute_error_usd": max(errors),
    }


def match(observed, expected, context):
    for key, value in expected.items():
        assert isclose(float(observed[key]), value, rel_tol=1e-10, abs_tol=1e-8), (
            f"{context}: {key} differs: stored {observed[key]}, recomputed {value}"
        )


def main():
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text())
    result = json.loads((EXPERIMENT / "result.json").read_text())
    references = json.loads((EXPERIMENT / "references.json").read_text())
    dev_path = PROJECT / protocol["development_file"]
    assert sha256(dev_path.read_bytes()).hexdigest() == protocol["development_sha256"]
    assert protocol["horizon_hours"] == 177 and protocol["baseline_delay_minutes"] == 1

    prices = {}
    for reference in references["references"]:
        instant = utc(reference["instant"])
        close = raw_close(reference)
        assert instant not in prices or prices[instant] == close
        prices[instant] = close
    origins = [utc(value) for value in references["origins"]]
    assert origins == sorted(set(origins))
    assert all(origin.weekday() == 4 and (origin.hour, origin.minute) == (12, 0) for origin in origins)

    baseline = {origin: prices[origin - MINUTE] for origin in origins}
    momentum = {origin: baseline[origin] / prices[origin - WEEK - MINUTE] - 1 for origin in origins}
    realised = {origin: prices[origin + HORIZON] for origin in origins}
    returns = {origin: realised[origin] / baseline[origin] - 1 for origin in origins}

    def verify_fit(row):
        origin = utc(row["origin"])
        training = [earlier for earlier in origins if earlier + HORIZON + MINUTE < origin]
        assert len(training) >= 20
        assert utc(row["target"]) == origin + HORIZON
        assert utc(row["baseline_time"]) == origin - MINUTE
        assert utc(row["factor_past_time"]) == origin - WEEK - MINUTE
        assert int(row["train_n"]) == len(training)
        assert [utc(value) for value in json.loads(row["training_origins"])] == training
        assert utc(row["latest_train_target"]) == training[-1] + HORIZON
        x = [momentum[earlier] for earlier in training]
        y = [returns[earlier] for earlier in training]
        x_mean, y_mean = mean(x), mean(y)
        slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum((a - x_mean) ** 2 for a in x)
        intercept = y_mean - slope * x_mean
        expected_return = intercept + slope * momentum[origin]
        prediction = baseline[origin] * (1 + expected_return)
        match(row, {"baseline": baseline[origin], "momentum": momentum[origin],
                    "realised": realised[origin], "intercept": intercept, "slope": slope,
                    "expected_return": expected_return, "prediction": prediction}, origin)
        return baseline[origin], prediction, realised[origin]

    with dev_path.open(newline="") as handle:
        development = [row for row in csv.DictReader(handle) if row["split"] == "development"]
    selection_boundary = utc("2025-10-24T12:00:00+00:00")
    eligible = [origin for origin in origins if origin + HORIZON + MINUTE < selection_boundary
                and sum(earlier + HORIZON + MINUTE < origin for earlier in origins) >= 20]
    assert len(development) == len(eligible) == 20
    assert [utc(row["origin"]) for row in development] == eligible
    development_values = [verify_fit(row) for row in development]

    def blended_metrics(values, weight):
        return scores([(base + weight * (prediction - base), actual) for base, prediction, actual in values])

    development_scores = {weight: blended_metrics(development_values, weight) for weight in WEIGHTS}
    stored_development = {row["weight"]: row for row in protocol["development_scores"]}
    assert set(stored_development) == set(WEIGHTS)
    for weight in WEIGHTS:
        match(stored_development[weight], development_scores[weight], f"development w={weight}")
    weight = min(WEIGHTS, key=lambda candidate: (development_scores[candidate]["mae_usd"], candidate))
    assert weight == protocol["weight"] == result["weight_selected_before_evaluation"]

    start, end = utc(protocol["evaluation_start"]), utc(protocol["evaluation_end"])
    assert (start, end) == (utc("2026-01-09T12:00:00+00:00"), utc("2026-10-30T12:00:00+00:00"))
    expected_origins = [start + index * WEEK for index in range(43)]
    assert [origin for origin in origins if origin >= start] == expected_origins
    assert end == expected_origins[-1]
    assert result["evaluation"] == {"start": start.isoformat(), "end": end.isoformat(),
                                    "last_target": (end + HORIZON).isoformat()}
    with (EXPERIMENT / "backtest.csv").open(newline="") as handle:
        evaluation = list(csv.DictReader(handle))
    assert [utc(row["origin"]) for row in evaluation] == expected_origins
    evaluation_values = [verify_fit(row) for row in evaluation]
    for row, (base, prediction, _) in zip(evaluation, evaluation_values):
        match(row, {"shrinkage_prediction": base + weight * (prediction - base)}, row["origin"])
    independent_scores = {name: blended_metrics(evaluation_values, candidate) for name, candidate in
                          (("no_change", 0), ("fixed_shrinkage", weight), ("full_momentum", 1))}
    for name, metrics in independent_scores.items():
        match(result["scores"][name], metrics, name)

    print(json.dumps({"status": "passed", "raw_reference_records": len(references["references"]),
                      "distinct_exact_closes": len(prices), "development_fits_checked": len(development),
                      "evaluation_fits_checked": len(evaluation), "fixed_weight": weight,
                      "historical_publication_vintages_verified": False,
                      "scores": independent_scores}, indent=2))


if __name__ == "__main__":
    main()
