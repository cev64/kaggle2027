"""Quick incremental-value checks behind docs/recommendation.md: partial correlations of
tracked 40-curve features (and in-game speed) with DL / WR outcomes after controlling for size,
role, draft slot and stopwatch. Run after scripts/eda/drill_quality.py and outcome_reliability.py."""

# --- DL: 40-curve shape vs get-off / pressure (controls: weight, interior share, log pick, peak speed[, ten_yd_split])
import polars as pl, numpy as np
from scipy import stats
o=pl.read_parquet('data/processed/outcomes_player.parquet')
print(o.filter(pl.col('group')=='DL')['metric'].unique().to_list())
a=pl.read_parquet('data/processed/eda/attempt_features.parquet').filter(pl.col('drillCanon')=='FORTY_YARD_DASH')
f=a.group_by('nflId').agg([pl.col(c).mean() for c in ['split2_10','d90','v10rel','peakSpeed','t40','fly10_40']])
cr=pl.read_csv('data/raw/combine_results.csv',null_values=['NA']).select(pl.col('nfl_id').alias('nflId'),'combine_weight','forty','ten_yd_split','combine_position')
p=pl.read_csv('data/raw/players.csv',null_values=['NA']).select(pl.col('nfl_id').alias('nflId'),'nfl_position','draft_overall_pick')
pp=pl.read_csv('data/raw/player_play.csv',null_values=['NA',''],infer_schema_length=None).filter(pl.col('player_get_off').is_not_null())
inter=pp.group_by(pl.col('nfl_id').alias('nflId')).agg(interior=(pl.col('lined_up_position')=='INTERIOR_LINE').mean())
def test(metric):
    y=o.filter((pl.col('group')=='DL')&(pl.col('metric')==metric)).select(pl.col('nfl_id').alias('nflId'),pl.col('eb_mean').alias('y'),'n')
    d=y.join(f,on='nflId').join(cr,on='nflId').join(p,on='nflId').join(inter,on='nflId',how='left').with_columns(lp=pl.col('draft_overall_pick').fill_null(260).log()).drop_nulls(['y','split2_10','combine_weight','interior','peakSpeed'])
    print(metric,'n=',d.height)
    import numpy.linalg as L
    def resid(v,X):
        X=np.c_[np.ones(len(v)),X]; b=L.lstsq(X,v,rcond=None)[0]; return v-X@b
    base=d.select('combine_weight','interior','lp','peakSpeed').to_numpy()
    for feat in ['split2_10','d90','v10rel','t40']:
        r=stats.pearsonr(resid(d['y'].to_numpy(),base),resid(d[feat].to_numpy(),base))
        print(f'  {feat}: partial r={r[0]:+.2f} p={r[1]:.3f}')
    dd=d.drop_nulls('ten_yd_split')
    base2=dd.select('combine_weight','interior','lp','peakSpeed','ten_yd_split').to_numpy()
    for feat in ['split2_10','d90','v10rel']:
        r=stats.pearsonr(resid(dd['y'].to_numpy(),base2),resid(dd[feat].to_numpy(),base2))
        print(f'  +ten_yd_split ctrl {feat}: n={dd.height} partial r={r[0]:+.2f} p={r[1]:.3f}')
for m in o.filter(pl.col('group')=='DL')['metric'].unique().to_list():
    if 'get' in m.lower() or 'pressure' in m.lower(): test(m)

# --- WR/DL: in-game speed, retention and 40-curve shape vs outcomes (controls: weight, height, log pick, forty)
import polars as pl, numpy as np
from scipy import stats
o=pl.read_parquet('data/processed/outcomes_player.parquet')
for g in ['WR','DB','DL']: print(g, o.filter(pl.col('group')==g)['metric'].unique().to_list())
a=pl.read_parquet('data/processed/eda/attempt_features.parquet').filter(pl.col('drillCanon')=='FORTY_YARD_DASH')
f=a.group_by('nflId').agg([pl.col(c).mean() for c in ['split2_10','d90','v10rel','peakSpeed']])
cr=pl.read_csv('data/raw/combine_results.csv',null_values=['NA']).select(pl.col('nfl_id').alias('nflId'),'combine_weight','combine_height','forty')
p=pl.read_csv('data/raw/players.csv',null_values=['NA']).select(pl.col('nfl_id').alias('nflId'),'draft_overall_pick')
def wide(g):
    w=o.filter(pl.col('group')==g).pivot(on='metric',index='nfl_id',values='eb_mean').rename({'nfl_id':'nflId'})
    return w.join(f,on='nflId').join(cr,on='nflId').join(p,on='nflId').with_columns(lp=pl.col('draft_overall_pick').fill_null(260).log())
def resid(v,X):
    X=np.c_[np.ones(len(v)),X]; b=np.linalg.lstsq(X,v,rcond=None)[0]; return v-X@b
for g,ys in [('WR',['separation OE','separation','targets/route']),('DL',['pressure OE','get-off, rush snaps only'])]:
    w=wide(g)
    spd=[c for c in w.columns if c.startswith('max speed/play')][0]
    w=w.with_columns(retention=pl.col(spd)/pl.col('peakSpeed'))
    for y in ys:
        if y not in w.columns: continue
        d=w.drop_nulls([y,spd,'peakSpeed','combine_weight','forty'])
        base=d.select('combine_weight','combine_height','lp','forty').to_numpy()
        for x in [spd,'retention','peakSpeed','d90','v10rel']:
            r=stats.pearsonr(resid(d[y].to_numpy(),base),resid(d[x].to_numpy(),base))
            print(f'{g} {y:28s} ~ {x:22s} n={d.height} partial r={r[0]:+.2f} p={r[1]:.3f}')
