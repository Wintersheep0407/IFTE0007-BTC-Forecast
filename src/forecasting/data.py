"""Saved-response Coinbase ingestion. No interpolation or replacement prices."""
import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UTC = timezone.utc
ENDPOINT = 'https://api.exchange.coinbase.com/products/BTC-USD/candles'
CUTOFF = datetime(2026, 11, 13, 12, tzinfo=UTC)
TARGET = datetime(2026, 11, 20, 21, tzinfo=UTC)
HORIZON = TARGET - CUTOFF

def utc(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError('Timezone-aware timestamp required')
    return value.astimezone(UTC)

def windows(start, end, granularity):
    start, end = utc(start), utc(end)
    if granularity not in (60, 300, 900, 3600, 21600, 86400) or start >= end:
        raise ValueError('Invalid interval or granularity')
    if start.timestamp() % granularity or end.timestamp() % granularity:
        raise ValueError('Boundaries must align with candle grid')
    while start < end:
        stop = min(end, start + timedelta(seconds=granularity * 299))
        yield start, stop
        start = stop

def collect(start, end, granularity, folder):
    """End-exclusive interval; refuse future/incomplete requests."""
    start, end = utc(start), utc(end)
    if end > datetime.now(UTC) or end > CUTOFF:
        raise ValueError('Development input end exceeds now or assignment cutoff')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    manifest = {'product':'BTC-USD', 'venue':'Coinbase Exchange', 'start':start.isoformat(),
                'end':end.isoformat(), 'granularity':granularity, 'status':'development', 'requests':[]}
    manifest['collection_state']='in_progress'
    manifest['collection_started_at']=datetime.now(UTC).isoformat()
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        for i, (a,b) in enumerate(windows(start,end,granularity)):
            params={'start':a.isoformat(),'end':b.isoformat(),'granularity':granularity}
            response=session.get(ENDPOINT,params=params,timeout=30)
            retrieved=datetime.now(UTC).isoformat()
            name=f'{i:05d}.json'
            (folder/name).write_bytes(response.content)
            record={'file':name,'url':response.url,'params':params,'retrieved_at':retrieved,
                    'status_code':response.status_code,'sha256':hashlib.sha256(response.content).hexdigest(),
                    'publication_time':'unknown','http_date':response.headers.get('Date')}
            manifest['requests'].append(record)
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
            response.raise_for_status()
            if not isinstance(response.json(),list):raise ValueError('Unexpected response schema')
            time.sleep(0.15)
        manifest['collection_state']='complete'
    except Exception as error:
        manifest['collection_state']='failed'
        manifest['error_type']=type(error).__name__
        raise
    finally:
        manifest['collection_ended_at']=datetime.now(UTC).isoformat()
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
        session.close()
    return folder/'manifest.json'

def clean(manifest_path, output):
    manifest_path=Path(manifest_path)
    m=json.loads(manifest_path.read_text())
    start,end=utc(m['start']),utc(m['end'])
    g=m['granularity']; rows={}; duplicates=0; outside=0
    expected_windows=list(windows(start,end,g))
    if m.get('collection_state') in ('failed','in_progress'):raise ValueError('Incomplete collection')
    if not m['requests']:raise ValueError('Empty request manifest')
    if all('params' in r for r in m['requests']):
        actual=[(utc(r['params']['start']),utc(r['params']['end'])) for r in m['requests']]
        if actual!=expected_windows or any(int(r['params']['granularity'])!=g for r in m['requests']):
            raise ValueError('Incomplete or inconsistent request pagination')

    for req in m['requests']:
        raw=(manifest_path.parent/req['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=req['sha256']:
            raise ValueError('Raw snapshot hash mismatch')
        if req['status_code']!=200:
            raise ValueError('Failed API response in snapshot')
        payload=json.loads(raw)
        if not isinstance(payload,list):raise ValueError('Unexpected response schema')
        for row in payload:
            if not isinstance(row,list) or len(row)!=6 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in row):
                raise ValueError('Malformed candle')
            t,low,high,op,cl,vol=row
            if t % g or min(low,high,op,cl)<=0 or vol<0 or not low<=min(op,cl)<=max(op,cl)<=high:
                raise ValueError('Anomalous candle')
            stamp=datetime.fromtimestamp(t,UTC)
            if not start<=stamp<end:
                outside+=1; continue
            if t in rows:
                duplicates+=1
                if rows[t]['values']!=row:
                    raise ValueError('Conflicting duplicate candle')
                continue
            rows[t]={'values':row,'retrieved_at':req['retrieved_at']}
    expected=set(range(int(start.timestamp()),int(end.timestamp()),g))
    missing=sorted(expected-set(rows))
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['time','low','high','open','close','volume','candle_end','retrieved_at','publication_time'])
        for t,record in sorted(rows.items()):
            writer.writerow([datetime.fromtimestamp(t,UTC).isoformat(),*record['values'][1:],
                             datetime.fromtimestamp(t+g,UTC).isoformat(),record['retrieved_at'],'unknown'])
    ordered=sorted(rows)
    jumps=[{'time':datetime.fromtimestamp(t,UTC).isoformat(),
            'return':rows[t]['values'][4]/rows[prev]['values'][4]-1}
           for prev,t in zip(ordered,ordered[1:])
           if t-prev==g and abs(rows[t]['values'][4]/rows[prev]['values'][4]-1)>0.20]
    report={'candles':len(rows),'duplicates':duplicates,'outside_range':outside,'missing_count':len(missing),
            'missing_utc':[datetime.fromtimestamp(t,UTC).isoformat() for t in missing],
            'large_adjacent_return_flags':jumps,'last_usable_candle_end':datetime.fromtimestamp(ordered[-1]+g,UTC).isoformat() if ordered else None,
            'granularity':g,'raw_manifest':str(manifest_path),'clean_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
            'availability_warning':'Candle end is an earliest bound, not verified publication. Historical snapshots are retrospective.'}
    out.with_suffix('.quality.json').write_text(json.dumps(report,indent=2))
    return report

def reference_price(csv_path, instant):
    """Exact one-minute close preceding instant; never substitutes another time."""
    instant=utc(instant)
    if instant.second or instant.microsecond:
        raise ValueError('Reference instant must be minute aligned')
    wanted=instant-timedelta(minutes=1)
    with Path(csv_path).open() as f:
        rows=[r for r in csv.DictReader(f) if utc(r['time'])==wanted]
    if len(rows)!=1 or utc(rows[0]['candle_end'])!=instant:
        raise ValueError('Exact one-minute reference candle unavailable')
    return float(rows[0]['close'])

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect');c.add_argument('--start',required=True);c.add_argument('--end',required=True)
    c.add_argument('--granularity',type=int,default=60);c.add_argument('--folder',required=True)
    c=sub.add_parser('clean');c.add_argument('manifest');c.add_argument('output')
    a=p.parse_args()
    if a.command=='collect': print(collect(a.start,a.end,a.granularity,a.folder))
    else: print(json.dumps(clean(a.manifest,a.output),indent=2))
if __name__=='__main__': main()
