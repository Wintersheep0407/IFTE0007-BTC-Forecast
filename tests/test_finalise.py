from datetime import timedelta
import json
import pytest
from forecasting.data import CUTOFF
from forecasting.finalise import check_clock,policy,propose

def test_early_finalisation_refused(tmp_path):
    with pytest.raises(ValueError,match='cutoff has not occurred'):
        propose('not_read.json','not_read.json',tmp_path/'proposal.json',now=CUTOFF-timedelta(seconds=1))
    assert not (tmp_path/'proposal.json').exists()

def test_clock_exact_boundary():
    assert check_clock(CUTOFF)==CUTOFF

def test_late_policy_refused(tmp_path):
    p=tmp_path/'policy.json';p.write_text(json.dumps({'recorded_at':(CUTOFF+timedelta(seconds=1)).isoformat()}))
    with pytest.raises(ValueError,match='Policy recorded after'):policy(p)

def test_post_cutoff_receipt_refused(tmp_path):
    from forecasting.finalise import eligible_manifest
    p=tmp_path/'manifest.json'
    p.write_text(json.dumps({'product':'BTC-USD','venue':'Coinbase Exchange','granularity':60,
                            'requests':[{'retrieved_at':(CUTOFF+timedelta(microseconds=1)).isoformat()}],
                            'end':CUTOFF.isoformat()}))
    with pytest.raises(ValueError,match='receipt after cutoff'):eligible_manifest(p)

def test_impossible_pre_candle_receipt_refused(tmp_path):
    from forecasting.finalise import eligible_manifest
    from urllib.parse import urlencode
    end=CUTOFF-timedelta(minutes=1)
    params={'start':(end-timedelta(minutes=1)).isoformat(),'end':end.isoformat(),'granularity':60}
    request={'retrieved_at':(end-timedelta(seconds=1)).isoformat(),'params':params,
             'url':'https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode(params),
             'sha256':'synthetic_unused','status_code':200,'file':'not_read.json'}
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'product':'BTC-USD','venue':'Coinbase Exchange','granularity':60,
                   'end':end.isoformat(),'requests':[request]}))
    with pytest.raises(ValueError,match='Receipt predates'):eligible_manifest(p)

def test_missing_final_request_metadata_refused(tmp_path):
    from forecasting.finalise import eligible_manifest
    end=CUTOFF-timedelta(minutes=1)
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'product':'BTC-USD','venue':'Coinbase Exchange','granularity':60,
         'end':end.isoformat(),'requests':[{'retrieved_at':end.isoformat()}]}))
    with pytest.raises(ValueError,match='Missing final source provenance'):eligible_manifest(p)

def test_cached_history_price_must_match_raw_check(tmp_path,monkeypatch):
    import forecasting.finalise as f
    p=tmp_path/'synthetic_history.json'
    p.write_text(json.dumps({'references':[{'manifest':'synthetic','instant':'2025-01-01T00:01:00Z','price':999}]}))
    monkeypatch.setattr(f,'eligible_manifest',lambda path:'synthetic_clean.csv')
    monkeypatch.setattr(f,'reference_price',lambda path,instant:100)
    with pytest.raises(ValueError,match='derived price disagrees'):f.verified_history(p)

def test_wrong_final_source_endpoint_refused(tmp_path):
    from forecasting.finalise import eligible_manifest
    from urllib.parse import urlencode
    end=CUTOFF-timedelta(minutes=1)
    params={'start':(end-timedelta(minutes=1)).isoformat(),'end':end.isoformat(),'granularity':60}
    p=tmp_path/'manifest.json'
    req={'retrieved_at':end.isoformat(),'params':params,'url':'https://example.invalid/candles?'+urlencode(params),
         'sha256':'synthetic_unused','file':'not_read.json','status_code':200}
    p.write_text(json.dumps({'product':'BTC-USD','venue':'Coinbase Exchange','granularity':60,'end':end.isoformat(),'requests':[req]}))
    with pytest.raises(ValueError,match='Incorrect final request endpoint'):eligible_manifest(p)


def test_proposal_applies_weight_to_full_fitted_return(tmp_path, monkeypatch):
    """Synthetic prices exercise the proposal path without real cutoff inputs."""
    import pandas as pd
    import forecasting.finalise as finalise

    source = tmp_path / 'synthetic_history.json'
    source.write_text('{}')
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps({
        'recorded_at': (CUTOFF - timedelta(days=1)).isoformat(),
        'cutoff': CUTOFF.isoformat(),
        'target': finalise.TARGET.isoformat(),
        'availability_lag_minutes': 1,
        'model': '7',
        'momentum_weight': 0.25,
        'training_reference_file': str(source),
        'training_references_sha256': finalise.digest(source),
    }))
    baseline_end = CUTOFF - timedelta(minutes=1)
    references = []
    prices = {}
    for role, end, price in (
        ('baseline', baseline_end, 110.0),
        ('factor', baseline_end - timedelta(days=7), 100.0),
    ):
        manifest = tmp_path / f'{role}.json'
        manifest.write_text('{}')
        references.append({'role': role, 'instant': end.isoformat(), 'price': price,
                           'manifest': str(manifest)})
        prices[str(manifest)] = price
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps({'policy_sha256': finalise.digest(policy_path),
                                  'references': references}))
    monkeypatch.setattr(finalise, 'eligible_manifest', lambda path: path)
    monkeypatch.setattr(finalise, 'reference_price', lambda path, instant: prices[path])
    verified_paths = []
    monkeypatch.setattr(finalise, 'verified_history', verified_paths.append)

    def synthetic_samples(path, days, lag_minutes):
        assert (path, days, lag_minutes) == (source, 7, 1)
        momentum = [i / 100 for i in range(20)]
        return pd.DataFrame({
            'target': [CUTOFF - timedelta(days=1)] * 20 + [CUTOFF],
            'momentum': momentum + [1.0],
            'return': [0.02 + 0.2 * value for value in momentum] + [9.0],
        })

    monkeypatch.setattr(finalise, 'samples', synthetic_samples)
    proposal = finalise.propose(inputs, policy_path, tmp_path / 'synthetic_proposal.json',
                                now=CUTOFF)
    assert verified_paths == [source]
    assert proposal['train_n'] == 20
    assert proposal['momentum_weight'] == 0.25
    assert proposal['intercept'] == pytest.approx(0.02)
    assert proposal['slope'] == pytest.approx(0.2)
    assert proposal['unadjusted_expected_return'] == pytest.approx(0.04)
    assert proposal['expected_return'] == pytest.approx(0.01)
    assert proposal['forecast_usd'] == '111.10'

    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Training references changed'):
        finalise.policy(policy_path)
