"""Chronological expanding OLS, label purging and time-safe model selection."""
import json
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from forecasting.data import utc,HORIZON

LOOKBACKS=(3,7,14)
MIN_TRAIN=20
HOLDOUT=10

def samples(path,lookback,lag_minutes=0):
    if lookback not in LOOKBACKS or lag_minutes not in (0,1):
        raise ValueError('Unsupported lookback or availability lag')
    d=json.loads(Path(path).read_text())
    prices={}
    for r in d['references']:
        instant=utc(r['instant']);price=float(r['price'])
        if not np.isfinite(price) or price<=0:raise ValueError('Invalid reference price')
        if instant in prices and prices[instant]!=price:raise ValueError('Conflicting reference')
        prices[instant]=price
    origins=[utc(o) for o in d['origins']]
    if origins!=sorted(set(origins)):raise ValueError('Origins must be unique and chronological')
    rows=[];delay=timedelta(minutes=lag_minutes)
    for o in origins:
        reference=o-delay;factor_past=reference-timedelta(days=lookback)
        p=prices[reference];past=prices[factor_past];target=o+HORIZON;y=prices[target]
        rows.append({'origin':o,'target':target,'baseline_time':reference,'factor_past_time':factor_past,
                     'baseline':p,'momentum':p/past-1,'realised':y,'return':y/p-1})
    return pd.DataFrame(rows)

def validate_frame(frame):
    if frame.empty or frame.origin.duplicated().any() or not frame.origin.is_monotonic_increasing:
        raise ValueError('Unique chronological nonempty sample required')
    for col in ('origin','target'):
        for value in frame[col]:utc(value)
    if not ((frame.target-frame.origin)==HORIZON).all():raise ValueError('Incorrect target horizon')
    if not np.isfinite(frame[['baseline','momentum','realised','return']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite model input')
    if (frame[['baseline','realised']]<=0).any().any():raise ValueError('Nonpositive model price')

def backtest(frame,min_train=MIN_TRAIN):
    validate_frame(frame)
    if min_train<2:raise ValueError('At least two training observations required')
    results=[]
    for _,row in frame.iterrows():
        train=frame[(frame.target<row.origin)&(frame.origin<row.origin)]
        if len(train)<min_train:continue
        fit=LinearRegression().fit(train[['momentum']],train['return'])
        ret=float(fit.predict(pd.DataFrame({'momentum':[row.momentum]}))[0])
        prediction=row.baseline*(1+ret)
        if not np.isfinite(prediction) or prediction<=0:raise ValueError('Invalid forecast; do not clip or skip')
        results.append({'origin':row.origin.isoformat(),'target':row.target.isoformat(),
                        'baseline_time':row.get('baseline_time',row.origin).isoformat(),
                        'factor_past_time':row.get('factor_past_time',row.origin).isoformat(),
                        'train_n':len(train),'latest_train_target':train.target.max().isoformat(),
                        'training_origins':json.dumps([t.isoformat() for t in train.origin]),
                        'baseline':row.baseline,'momentum':row.momentum,
                        'intercept':float(fit.intercept_),'slope':float(fit.coef_[0]),
                        'expected_return':ret,'prediction':prediction,'realised':row.realised})
    return pd.DataFrame(results)

def metrics(frame,column):
    if frame.empty:raise ValueError('Cannot score empty evaluation set')
    e=frame[column]-frame.realised
    return {'n':len(frame),'mae_usd':float(np.abs(e).mean()),'rmse_usd':float(np.sqrt((e**2).mean())),
            'mape_percent':float((np.abs(e)/frame.realised).mean()*100)}

def split_evaluation(frame,holdout=HOLDOUT):
    if len(frame)<holdout+6:raise ValueError('Insufficient development and holdout origins')
    boundary=utc(frame.iloc[-holdout].origin)
    earlier=frame.iloc[:-holdout]
    eligible=earlier[pd.to_datetime(earlier.target,utc=True)<boundary]
    purged=earlier[pd.to_datetime(earlier.target,utc=True)>=boundary]
    return eligible,frame.iloc[-holdout:],purged,boundary

def run(path='data/clean/research_references.json',lag_minutes=0):
    out=Path('results');out.mkdir(exist_ok=True)
    reports={};frames={};table=[]
    for days in LOOKBACKS:
        f=backtest(samples(path,days,lag_minutes));frames[str(days)]=f
    for f in frames.values():
        f['coefficient_origin']='estimated_OLS'
        f['factor_used']=True
    frames['no_change']=frames['7'].copy()
    baseline=frames['no_change']
    baseline['prediction']=baseline.baseline
    baseline[['intercept','slope','expected_return']]=0.0
    baseline['train_n']=0
    baseline['training_origins']='[]'
    baseline['latest_train_target']=''
    baseline['coefficient_origin']='defined_zero_no_change'
    baseline['factor_used']=False
    # Benchmark is scored on identical comparison origins; it does not fit OLS.

    for name,f in frames.items():
        dev,hold,purged,boundary=split_evaluation(f)
        f['split']='development'
        f.loc[f.index.isin(purged.index),'split']='selection_embargo'
        f.loc[f.index.isin(hold.index),'split']='holdout'
        f['error_usd']=f.prediction-f.realised;f['absolute_error_usd']=f.error_usd.abs()
        f['absolute_percentage_error']=100*f.absolute_error_usd/f.realised
        suffix=f'lag1_{name}d' if lag_minutes else f'{name}d'
        if name=='no_change':suffix='lag1_no_change' if lag_minutes else 'no_change'
        f.to_csv(out/f'backtest_{suffix}.csv',index=False)
        reports[name]={'development':metrics(dev,'prediction'),'holdout':metrics(hold,'prediction'),'all':metrics(f,'prediction')}
        for split,stats in reports[name].items():table.append({'lag_minutes':lag_minutes,'model':name,'split':split,**stats})
    chosen=min(['no_change','7','3','14'],key=lambda k:reports[k]['development']['mae_usd'])
    dev,hold,purged,boundary=split_evaluation(frames['7'])
    summary={'status':'preliminary_historical_only','horizon_hours':177,'min_training_samples':MIN_TRAIN,
             'lag_minutes':lag_minutes,'selection_rule':'Minimum MAE using only development targets strictly before first holdout origin.',
             'selection_time_kind':'simulated historical boundary, not a contemporaneous agent decision',
             'selection_decision_time':boundary.isoformat(),'development_origins':[str(o) for o in dev.origin],
             'selection_embargo_origins':[str(o) for o in purged.origin],
             'holdout_origins':[str(o) for o in hold.origin],
             'selected_model':chosen,'metrics':reports,
             'holdout_disclosure':'Previously inspected during initial development; correction does not create a fresh untouched holdout.',
             'availability_limitation':'Retrospective final candles assumed available at end plus declared lag; historical publication and revisions unverified.',
             'final_forecast':None}
    name='lag_sensitivity.json' if lag_minutes else 'historical_summary.json'
    (out/name).write_text(json.dumps(summary,indent=2)+'\n')
    pd.DataFrame(table).to_csv(out/('error_table_lag1.csv' if lag_minutes else 'error_table.csv'),index=False)
    print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__':run()
