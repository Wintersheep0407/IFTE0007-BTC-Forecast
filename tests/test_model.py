from datetime import datetime,timedelta,timezone
import pandas as pd
from forecasting.model import backtest,metrics

def test_purges_unavailable_training_targets():
    origin=datetime(2020,1,1,12,tzinfo=timezone.utc)
    rows=[]
    for i in range(30):
        o=origin+timedelta(weeks=i)
        rows.append({'origin':o,'target':o+timedelta(hours=177),'baseline':100+i,
                     'momentum':i/1000,'realised':101+i,'return':0.01+i/10000})
    f=pd.DataFrame(rows);r=backtest(f,min_train=5)
    assert len(r)>1
    assert all(pd.to_datetime(r.latest_train_target)<pd.to_datetime(r.origin))
    # Mutating future labels cannot affect an earlier fit or prediction.
    changed=f.copy();changed.loc[15:,'return']=100
    r2=backtest(changed,min_train=5)
    assert r.iloc[0].prediction==r2.iloc[0].prediction
    assert r.iloc[0].train_n==5

def test_metrics():
    f=pd.DataFrame({'prediction':[90,120],'realised':[100,100]})
    m=metrics(f,'prediction')
    assert m['mae_usd']==15 and m['mape_percent']==15.000000000000002

def test_selection_purges_unmatured_development_error():
    from forecasting.model import split_evaluation
    o=datetime(2025,1,1,12,tzinfo=timezone.utc)
    f=pd.DataFrame({'origin':[(o+timedelta(weeks=i)).isoformat() for i in range(31)],
                    'target':[(o+timedelta(weeks=i,hours=177)).isoformat() for i in range(31)]})
    dev,hold,purged,boundary=split_evaluation(f)
    assert len(dev)==20 and len(hold)==10 and len(purged)==1
    assert all(pd.to_datetime(dev.target,utc=True)<boundary)
    assert pd.to_datetime(purged.iloc[0].target)>boundary

def test_rejects_unsorted_and_wrong_horizon():
    import pytest
    from forecasting.model import validate_frame
    o=datetime(2025,1,1,12,tzinfo=timezone.utc)
    f=pd.DataFrame([{'origin':o,'target':o+timedelta(days=7),'baseline':100,'momentum':0,'realised':100,'return':0}])
    with pytest.raises(ValueError,match='horizon'):validate_frame(f)

def test_future_labels_do_not_change_any_earlier_forecast():
    o=datetime(2025,1,1,12,tzinfo=timezone.utc)
    f=pd.DataFrame([{'origin':o+timedelta(weeks=i),'target':o+timedelta(weeks=i,hours=177),
                     'baseline':100+i,'momentum':i/1000,'realised':101+i,'return':i/10000} for i in range(30)])
    r=backtest(f,min_train=5);changed=f.copy();changed.loc[15:,'return']=999
    r2=backtest(changed,min_train=5)
    eligible=pd.to_datetime(r.origin,utc=True)<=f.iloc[15].target
    pd.testing.assert_series_equal(r.loc[eligible,'prediction'],r2.loc[eligible,'prediction'])
