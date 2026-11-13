# IFTE0007 Assessment 1 · BTC-USD forecast

This repository contains the calculation and validation code for a Coinbase Exchange BTC-USD forecast, with a reproducible offline entry point. The local handoff contains the approved forecast record and its frozen inputs in `local_evidence/IFTE0007_BTC_Evidence_Package.zip`. The JSON inside the ZIP is the forecast record to use for reproduction and comparison with the report cover.

| Forecast specification | Value |
| --- | --- |
| Market and unit | Coinbase Exchange BTC-USD, USD per BTC |
| Forecast origin and information cutoff | 13 November 2026, 12:00 UTC |
| Target | 20 November 2026, 21:00 UTC |
| Price reference | Close of the 20:59:00–20:59:59 UTC Coinbase Exchange BTC-USD minute |
| Decision-to-target horizon | 177 hours |

## Forecasting method

Let `B(t)` be the one-minute candle close for the minute ending at `t − 1 minute`. At a decision time of 12:00 UTC, this means the minute beginning at 11:58 UTC. The factor is the simple seven-day momentum `M(t) = B(t) / B(t − 7 days) − 1`. The training label is `C(t + 177 hours) / B(t) − 1`, where `C` is the exact minute close at the label endpoint. The price-to-price interval is 177 hours and one minute. Chronological evaluation refits using earlier rows whose labels have matured by the evaluation origin.

The model fits an intercept and slope by ordinary least squares. The approved specification multiplies the full fitted return, including the intercept, by the fixed weight 0.25. The price formula is `B(t) × [1 + 0.25 × (intercept + slope × M(t))]`. Round the resulting USD price to cents. The no-change benchmark is `B(t)`. Slope sensitivity uses 0.5×, 1.0× and 1.5× of the fitted slope with the intercept, weight and inputs held fixed.

## Frozen inputs and evidence

The forecast archive contains saved one-minute candle responses from `https://api.exchange.coinbase.com/products/BTC-USD/candles`. Its manifests record exact UTC intervals, receipt timestamps, request URLs and SHA-256 checksums. The offline calculation verifies 192 distinct training candles for 95 labels and the two forecast input candles. The two input candles ended before noon. Their first receipts were around 12:08 UTC and the recheck around 14:16 UTC. The replay verifies the completed observations and disclosure. First publication by noon remains unverified.

The archive contains the agent instructions, original command receipts with recorded output, software versions and the forecast record. The complete original chat transcript is unavailable in the archive. The additional saved data for independently replaying the historical evaluation are in the local handoff under `local_evidence/backtest_snapshot/`. The saved benchmark and sensitivity tables are in `local_evidence/validation/`.

The Git index excludes the local evidence, numeric result files and the identified report and slides. A fresh clone needs the separately supplied evidence ZIP to reproduce the exact forecast. Section 3 of the [Coinbase Market Data Terms](https://www.coinbase.com/legal/market_data) restricts redistribution of data and derived material. Permission to share the saved evidence is unverified.

## Reproduce the final forecast offline

From the repository directory, place the authorized ZIP under `local_evidence/` using the filename below. Tested on macOS with CPython 3.12.14. No third-party packages are required. The command uses Python's ZIP support, so it requires no extra extraction utility.

```sh
python3 --version
work_dir="$(mktemp -d)"
python3 -m venv --without-pip "$work_dir/venv"
"$work_dir/venv/bin/python" -I -B -m zipfile -e \
  local_evidence/IFTE0007_BTC_Evidence_Package.zip "$work_dir/evidence"
cmp reproduce.py "$work_dir/evidence/reproduce.py"
"$work_dir/venv/bin/python" -I -B "$work_dir/evidence/reproduce.py"
```

Use the verified archive with SHA-256 `ab0cdac7f690c5c49331e6725bdea1d9523e74a127406e57453d66512332fcbc`. The command verifies the frozen-file hashes and timestamps, independently fits OLS, and checks that the rounded value equals the approved record. Expect `"status": "passed"` and a `forecast_value_usd` matching the extracted `final_forecast.json`. The calculation runs without network access, AI services, logins or API credentials.

## Replay historical verification offline

The local backtest snapshot preserves the saved reference index, exact raw inputs and reported output files. These commands run the independent historical verifier and slope review in a temporary directory and compare the regenerated tables against the saved results.

```sh
review_dir="$(mktemp -d)"
cp -R local_evidence/backtest_snapshot/. "$review_dir/"
mkdir -p "$review_dir/experiments/methodology_review"
cp experiments/shrinkage_177h/verify.py "$review_dir/experiments/shrinkage_177h/"
cp experiments/methodology_review/review.py "$review_dir/experiments/methodology_review/"
python3 -I -B "$review_dir/experiments/shrinkage_177h/verify.py"
python3 -I -B "$review_dir/experiments/methodology_review/review.py"
for result in result.json backtest_errors.csv slope_sensitivity.csv; do
  cmp "local_evidence/validation/$result" "$review_dir/experiments/methodology_review/$result"
done
```

## Source layout and tests

- `reproduce.py` is the unchanged standard-library replay included in the forecast ZIP.
- `src/forecasting/` contains the original data collection, cleaning, regression and approval workflow.
- `experiments/shrinkage_177h/` contains the original fixed-weight experiment and its independent verifier.
- `experiments/methodology_review/` contains the independent chronological backtest and sensitivity review.
- `docs/agent_instructions.md` preserves the research protocol. The run records stay in the local evidence ZIP.
- `tests/` exercises the pipeline with synthetic fixtures. The original workspace also contains artifact-dependent tests.
- `requirements.txt` records the standard-library runtime. `requirements-research.txt` pins the original research and test stack, including NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1 and pytest 8.4.2. See `software_versions.json` for the original forecast-run versions.

With the research packages already available or provided as local wheels, run the repository tests as follows:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links /path/to/local/wheels -r requirements-research.txt
.venv/bin/python -m pytest -q
```
