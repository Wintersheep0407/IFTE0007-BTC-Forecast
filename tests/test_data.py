import hashlib
import json
from datetime import timedelta
import pytest
from forecasting.data import windows,utc,clean,reference_price,HORIZON,TARGET,collect

def snapshot(tmp_path, rows):
    raw=json.dumps(rows).encode();(tmp_path/'raw.json').write_bytes(raw)
    m={'start':'2025-01-01T00:00:00Z','end':'2025-01-01T00:03:00Z','granularity':60,
       'requests':[{'file':'raw.json','sha256':hashlib.sha256(raw).hexdigest(),'status_code':200,'retrieved_at':'2026-11-09T00:00:00Z'}]}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m));return p

def test_horizon_and_target():
    assert HORIZON==timedelta(days=7,hours=9)
    assert (TARGET-timedelta(minutes=1)).isoformat()=='2026-11-20T20:59:00+00:00'

def test_windows():
    w=list(windows('2025-01-01T00:00:00Z','2025-01-02T00:00:00Z',60))
    assert len(w)==5
    assert all((b-a).total_seconds()<=299*60 for a,b in w)
    assert all(w[i][1]==w[i+1][0] for i in range(len(w)-1))

def test_timezone():
    with pytest.raises(ValueError):utc('2025-01-01')

def test_gaps_and_reference(tmp_path):
    t=int(utc('2025-01-01T00:00:00Z').timestamp())
    row=[t,99,102,100,101,4]
    p=snapshot(tmp_path,[row,row,[t+120,99,102,100,102,5]])
    out=tmp_path/'clean.csv';q=clean(p,out)
    assert q['duplicates']==1 and q['missing_count']==1 and q['candles']==2
    assert reference_price(out,'2025-01-01T00:01:00Z')==101
    with pytest.raises(ValueError):reference_price(out,'2025-01-01T00:02:00Z')

def test_anomaly_and_tamper(tmp_path):
    t=int(utc('2025-01-01T00:00:00Z').timestamp())
    p=snapshot(tmp_path,[[t,100,99,100,101,2]])
    with pytest.raises(ValueError,match='Anomalous'):clean(p,tmp_path/'out.csv')
    (tmp_path/'raw.json').write_text('[]')
    with pytest.raises(ValueError,match='hash'):clean(p,tmp_path/'out.csv')

def test_future_guard(tmp_path):
    with pytest.raises(ValueError):collect('2026-11-20T00:00:00Z','2026-11-21T00:00:00Z',60,tmp_path/'future')

def test_conflicting_duplicate(tmp_path):
    t=int(utc('2025-01-01T00:00:00Z').timestamp())
    p=snapshot(tmp_path,[[t,99,102,100,101,2],[t,99,102,100,102,2]])
    with pytest.raises(ValueError,match='Conflicting'):clean(p,tmp_path/'out.csv')

def test_grid_and_granularity():
    with pytest.raises(ValueError):list(windows('2025-01-01T00:00:01Z','2025-01-01T01:00:00Z',60))
    with pytest.raises(ValueError):list(windows('2025-01-01T00:00:00Z','2025-01-01T01:00:00Z',61))

def test_reference_rejects_hourly(tmp_path):
    t=int(utc('2025-01-01T00:00:00Z').timestamp())
    p=snapshot(tmp_path,[[t,99,102,100,101,2]])
    m=json.loads(p.read_text());m['granularity']=3600;m['end']='2025-01-01T01:00:00Z';p.write_text(json.dumps(m))
    out=tmp_path/'hour.csv';clean(p,out)
    with pytest.raises(ValueError):reference_price(out,'2025-01-01T00:01:00Z')

def test_rejects_partial_manifest(tmp_path):
    p=snapshot(tmp_path,[])
    m=json.loads(p.read_text());m['collection_state']='failed';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Incomplete'):clean(p,tmp_path/'out.csv')

def test_rejects_invalid_top_level_payload(tmp_path):
    p=snapshot(tmp_path,{'error':'not candles'})
    with pytest.raises(ValueError,match='schema'):clean(p,tmp_path/'out.csv')

def test_reports_large_jump_without_changing_price(tmp_path):
    t=int(utc('2025-01-01T00:00:00Z').timestamp())
    p=snapshot(tmp_path,[[t,99,101,100,100,2],[t+60,149,151,150,150,2]])
    out=tmp_path/'out.csv';q=clean(p,out)
    assert len(q['large_adjacent_return_flags'])==1
    assert reference_price(out,'2025-01-01T00:02:00Z')==150

def test_failed_request_is_recorded(tmp_path,monkeypatch):
    import requests
    from forecasting.data import collect
    def fail(*args,**kwargs):raise requests.ConnectionError('synthetic transport failure')
    monkeypatch.setattr(requests.Session,'get',fail)
    folder=tmp_path/'failed'
    with pytest.raises(requests.ConnectionError):collect('2025-01-01T00:00:00Z','2025-01-01T00:01:00Z',60,folder)
    m=json.loads((folder/'manifest.json').read_text())
    assert m['collection_state']=='failed' and m['error_type']=='ConnectionError'
    with pytest.raises(ValueError,match='Incomplete'):clean(folder/'manifest.json',tmp_path/'out.csv')
