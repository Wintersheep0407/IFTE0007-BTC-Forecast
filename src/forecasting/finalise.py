"""Cutoff-safe proposal and explicit interactive human approval. No trading."""
import argparse
from datetime import datetime,timedelta
from decimal import Decimal,ROUND_HALF_UP
import hashlib,json
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import numpy as np
from sklearn.linear_model import LinearRegression
from forecasting.data import UTC,CUTOFF,TARGET,HORIZON,utc,collect,clean,reference_price
from forecasting.model import samples

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_clock(now=None):
    now=utc(now or datetime.now(UTC))
    if now<CUTOFF:raise ValueError('Official cutoff has not occurred; final forecast must remain pending')
    return now

def eligible_manifest(path):
    m=json.loads(Path(path).read_text())
    if m.get('product')!='BTC-USD' or m.get('venue')!='Coinbase Exchange' or m.get('granularity')!=60:
        raise ValueError('Incorrect final data source')
    if not m['requests'] or any(utc(r['retrieved_at'])>CUTOFF for r in m['requests']):
        raise ValueError('Final input receipt after cutoff or absent')
    if utc(m['end'])>CUTOFF:raise ValueError('Final input candle after cutoff')
    for r in m['requests']:
        if not all(key in r for key in ('url','params','retrieved_at','sha256','status_code','file')):
            raise ValueError('Missing final source provenance')
        params=r['params']
        if not all(key in params for key in ('start','end','granularity')):
            raise ValueError('Missing final request parameters')
        if utc(r['retrieved_at'])<utc(params['end']):
            raise ValueError('Receipt predates completed requested candle')
        endpoint=urlparse(r['url']);query=parse_qs(endpoint.query)
        if (endpoint.scheme!='https' or endpoint.netloc!='api.exchange.coinbase.com'
                or endpoint.path!='/products/BTC-USD/candles'):
            raise ValueError('Incorrect final request endpoint')
        try:
            matches=(utc(query['start'][0])==utc(params['start'])
                     and utc(query['end'][0])==utc(params['end'])
                     and int(query['granularity'][0])==int(params['granularity'])==60)
        except (KeyError,ValueError,IndexError):
            matches=False
        if not matches:raise ValueError('Final URL and request parameters disagree')
    # Validate raw hashes, pagination and missing candles using production cleaner.
    cache=Path('data/clean/final_validation')/(digest(path)+'.csv')
    q=clean(path,cache)
    if q['missing_count']:raise ValueError('Missing required final candle')
    return cache

def policy(path):
    d=json.loads(Path(path).read_text())
    if utc(d['recorded_at'])>CUTOFF:raise ValueError('Policy recorded after cutoff')
    if d['model'] not in ('no_change','7','3','14'):raise ValueError('Unsupported model')
    regression = d['model'] != 'no_change'
    weight = d.get('momentum_weight', 1.0 if regression else 0.0)
    if type(weight) not in (int, float) or not 0 <= weight <= 1:
        raise ValueError('Momentum weight must be a number between zero and one')
    if regression != (weight > 0):
        raise ValueError('Only a regression can have a positive momentum weight')
    d['momentum_weight'] = weight
    if ('training_reference_file' in d) != ('training_references_sha256' in d):
        raise ValueError('Specify both the training reference file and its SHA-256')
    if 'training_reference_file' in d:
        if not regression:
            raise ValueError('A no-change policy does not use a training file')
        if digest(d['training_reference_file']) != d['training_references_sha256']:
            raise ValueError('Training references changed after policy selection')
    if d['availability_lag_minutes']!=1:raise ValueError('Final policy requires one-minute observation buffer')
    if utc(d['cutoff'])!=CUTOFF or utc(d['target'])!=TARGET:raise ValueError('Wrong policy boundary')
    return d

def capture(policy_path,folder):
    d=policy(policy_path);now=datetime.now(UTC)
    instant=CUTOFF-timedelta(minutes=d['availability_lag_minutes'])
    if not instant<=now<CUTOFF:raise ValueError('Run capture between 11:59 and 12:00 UTC on 13 November; never backdate retrieval')
    root=Path(folder);root.mkdir(parents=True,exist_ok=False)
    records=[]
    days=int(d['model']) if d['model']!='no_change' else d['lookback_days']
    for name,end in (('baseline',instant),('factor',instant-timedelta(days=days))):
        m=collect(end-timedelta(minutes=1),end,60,root/name)
        cache=eligible_manifest(m)  # Also refuses a response that finishes after cutoff.
        records.append({'role':name,'instant':end.isoformat(),'price':reference_price(cache,end),'manifest':str(m)})
    record={'status':'captured_not_approved','policy_sha256':digest(policy_path),'references':records}
    (root/'inputs.json').write_text(json.dumps(record,indent=2)+'\n')
    print(root/'inputs.json')

def verified_history(source):
    history=json.loads(Path(source).read_text())
    for r in history['references']:
        cache=eligible_manifest(r['manifest'])
        if reference_price(cache,r['instant'])!=r['price']:
            raise ValueError('Historical derived price disagrees with raw source')
    return history

def propose(inputs,policy_path,output,now=None):
    current=check_clock(now);d=policy(policy_path)
    if Path(output).exists():raise ValueError('Proposal output already exists; choose a new filename')
    record=json.loads(Path(inputs).read_text())
    if record['policy_sha256']!=digest(policy_path):raise ValueError('Policy changed after input capture')
    refs={}
    for r in record['references']:
        cache=eligible_manifest(r['manifest']);refs[r['role']]=(utc(r['instant']),reference_price(cache,r['instant']))
        if refs[r['role']][1]!=r['price']:raise ValueError('Input price mismatch')
    baseline_time,base=refs['baseline'];past_time,past=refs['factor']
    days=int(d['model']) if d['model']!='no_change' else d['lookback_days']
    if baseline_time!=CUTOFF-timedelta(minutes=1) or baseline_time-past_time!=timedelta(days=days):
        raise ValueError('Incorrect final reference timing')
    momentum=base/past-1;alpha=beta=0.0;train_n=0
    sources={'policy':{'path':str(policy_path),'sha256':digest(policy_path)},
             'input_record':{'path':str(inputs),'sha256':digest(inputs)}}
    for r in record['references']:sources[r['role']]={'path':r['manifest'],'sha256':digest(r['manifest'])}
    if d['model']!='no_change':
        source=Path(d.get('training_reference_file', 'data/clean/research_lag_references.json'))
        verified_history(source)
        f=samples(source,days,1);train=f[f.target<CUTOFF]
        if len(train)<20:raise ValueError('Insufficient eligible training observations')
        fitted=LinearRegression().fit(train[['momentum']],train['return'])
        alpha=float(fitted.intercept_);beta=float(fitted.coef_[0]);train_n=len(train)
        sources['training']={'path':str(source),'sha256':digest(source)}
    unadjusted = alpha + beta * momentum
    expected = d['momentum_weight'] * unadjusted
    value = base * (1 + expected)
    if not np.isfinite(value) or value<=0:raise ValueError('Invalid numerical proposal')
    price=str(Decimal(str(value)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
    result={'status':'awaiting_explicit_human_approval','market':'Bitcoin','identifier':'BTC-USD',
        'unit':'USD per BTC','venue':'Coinbase Exchange','information_cutoff':CUTOFF.isoformat(),'target':TARGET.isoformat(),
        'scoring_candle_start':(TARGET-timedelta(minutes=1)).isoformat(),'granularity_seconds':60,
        'horizon_hours':177,'model':d['model'],'baseline_time':baseline_time.isoformat(),'baseline_price':base,
        'momentum':momentum,'lookback_days':days,'intercept':alpha,'slope':beta,'train_n':train_n,
        'coefficient_note':'No-change coefficients are defined zeros, not fitted' if d['model']=='no_change' else 'OLS estimated',
        'momentum_weight':d['momentum_weight'],'unadjusted_expected_return':unadjusted,
        'expected_return':expected,'forecast_usd':price,'sources':sources,'calculated_at':current.isoformat(),'approved_by':None}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

def approve(proposal,reviewer,output):
    check_clock()
    if Path(output).exists():raise ValueError('Approval record already exists')
    d=json.loads(Path(proposal).read_text())
    if d['status']!='awaiting_explicit_human_approval':raise ValueError('Not a reviewable proposal')
    sha=digest(proposal)
    print(json.dumps(d,indent=2))
    print('Review methodology, sources, independent checks, identifiers and matching draft materials before approval.')
    required=f'APPROVE {d["forecast_usd"]} {sha}'
    typed=input(f'The student must type exactly:\n{required}\n> ')
    if typed!=required:raise ValueError('Human approval not supplied')
    approval={'reviewer':reviewer,'approved_at':datetime.now(UTC).isoformat(),'proposal_path':str(proposal),'proposal_sha256':sha,'approval_text':typed}
    Path(output).write_text(json.dumps(approval,indent=2)+'\n');print(output)

def record(proposal,approval,output):
    check_clock()
    p=Path(proposal);a=json.loads(Path(approval).read_text());d=json.loads(p.read_text())
    if Path(output).exists():
        existing=json.loads(Path(output).read_text())
        if existing.get('forecast_usd') is not None:raise ValueError('Final record already populated; refusing overwrite')
    if a['proposal_sha256']!=digest(p) or not a.get('reviewer','').strip():raise ValueError('Approval identity or proposal mismatch')
    if utc(a['approved_at'])<CUTOFF or a['approval_text']!=f'APPROVE {d["forecast_usd"]} {digest(p)}':
        raise ValueError('Invalid human approval record')
    for source in d['sources'].values():
        if digest(source['path'])!=source['sha256']:raise ValueError('Approved source changed')
    for key in ('baseline','factor'):eligible_manifest(d['sources'][key]['path'])
    if 'training' in d['sources']:
        verified_history(d['sources']['training']['path'])
    if d['status']!='awaiting_explicit_human_approval':raise ValueError('Invalid proposal status')
    d['status']='human_approved_pending_materials_freeze';d['approved_by']=a['reviewer'];d['approved_at']=a['approved_at']
    d['proposal_sha256']=digest(p)
    Path(output).write_text(json.dumps(d,indent=2)+'\n');print(output)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('capture');c.add_argument('--policy',default='config/research_policy.json');c.add_argument('--folder',required=True)
    c=sub.add_parser('propose');c.add_argument('--inputs',required=True);c.add_argument('--policy',default='config/research_policy.json');c.add_argument('--output',required=True)
    c=sub.add_parser('approve');c.add_argument('--proposal',required=True);c.add_argument('--reviewer',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('record');c.add_argument('--proposal',required=True);c.add_argument('--approval',required=True);c.add_argument('--output',default='results/final_forecast.json')
    a=p.parse_args()
    if a.command=='capture':capture(a.policy,a.folder)
    elif a.command=='propose':propose(a.inputs,a.policy,a.output)
    elif a.command=='approve':approve(a.proposal,a.reviewer,a.output)
    else:record(a.proposal,a.approval,a.output)
if __name__=='__main__':main()
