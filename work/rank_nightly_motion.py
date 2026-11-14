from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
for f in sorted(Path('pipeline/cache/probe').glob('most_*.ecsv')):
    t=Table.read(f,format='ascii.ecsv'); t=t[np.isfinite(np.asarray(t['mjd_start'],dtype=float))]
    mjd=np.asarray(t['mjd_start'],dtype=float); night=np.floor(mjd).astype(int)
    nights=np.unique(night); gaps=np.diff(nights)
    # De-duplicate same-frame repeats and sub-second near-identical timestamps by rounding MJD to 5 decimals (~0.86 s).
    epoch=np.round(mjd,5)
    rows=[]
    for n in nights:
        ix=np.flatnonzero(night==n)
        by={}
        for i in ix:
            by.setdefault(epoch[i],i)
        ix=np.array([by[k] for k in sorted(by)])
        if len(ix)<2: continue
        tm=mjd[ix]; dt=np.diff(tm)
        coord=SkyCoord(np.asarray(t['ra_obj'][ix],dtype=float)*u.deg,np.asarray(t['dec_obj'][ix],dtype=float)*u.deg)
        ds=coord[:-1].separation(coord[1:]).arcsec
        ok=dt>0
        rows.extend(zip(ds[ok],dt[ok],(ds[ok]/dt[ok])))
    rows=np.array(rows,dtype=float) if rows else np.empty((0,3))
    v=np.asarray(t['vmag'],dtype=float); v=v[np.isfinite(v)]
    print(f'{f.stem.split("_")[2]} obsid={len(set(map(str,t["observationid"])))}, nights={len(nights)}, baseline={(mjd.max()-mjd.min()):.2f}d, maxgap={gaps.max() if len(gaps) else 0}d, vmag={v.min():.2f}..{v.max():.2f}, intranight_pairs={len(rows)}, motion median/max={np.median(rows[:,0]) if len(rows) else 0:.4f}/{np.max(rows[:,0]) if len(rows) else 0:.4f} arcsec, rate median/max={np.median(rows[:,2]) if len(rows) else 0:.2f}/{np.max(rows[:,2]) if len(rows) else 0:.2f} arcsec/day')
