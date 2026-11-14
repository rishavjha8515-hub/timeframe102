from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
for f in sorted(Path('pipeline/cache/probe').glob('most_*.ecsv')):
    t=Table.read(f, format='ascii.ecsv')
    times=np.asarray(t['mjd_start'],dtype=float)
    valid=np.isfinite(times)
    times=times[valid]
    nights=np.unique(np.floor(times).astype(int))
    gaps=np.diff(nights)
    vm=np.asarray(t['vmag'],dtype=float); vm=vm[np.isfinite(vm)]
    obs=np.asarray(t['observationid']).astype(str)
    seq=[]
    for oid in np.unique(obs):
        ix=np.flatnonzero((obs==oid)&np.isfinite(np.asarray(t['mjd_start'],dtype=float)))
        ix=ix[np.argsort(np.asarray(t['mjd_start'][ix],dtype=float))]
        # One ephemeris position per distinct actual start time.
        keep=[]
        for i in ix:
            if not keep or float(t['mjd_start'][i]) != float(t['mjd_start'][keep[-1]]): keep.append(i)
        if len(keep)<2: continue
        tm=np.asarray(t['mjd_start'][keep],dtype=float)
        dt=np.diff(tm)
        c=SkyCoord(np.asarray(t['ra_obj'][keep],dtype=float)*u.deg,np.asarray(t['dec_obj'][keep],dtype=float)*u.deg)
        sep=c[:-1].separation(c[1:]).arcsec
        use=dt>0
        if not np.any(use): continue
        seq.append((float(np.sum(sep[use])),float(np.sum(dt[use])),float(np.sum(sep[use])/np.sum(dt[use])),len(keep)))
    seq=np.asarray(seq) if seq else np.empty((0,4))
    start=Time(np.min(times),format='mjd',scale='utc').isot[:10] if len(times) else 'NA'
    end=Time(np.max(times),format='mjd',scale='utc').isot[:10] if len(times) else 'NA'
    baseline=np.max(times)-np.min(times) if len(times) else 0
    print(f'{f.name}\trows={len(t)}\tobs_ids={len(np.unique(obs))}\tnights={len(nights)}\tbaseline={baseline:.2f}d\tUTC={start}..{end}\tmax_gap={np.max(gaps) if len(gaps) else 0}d\tmedian_gap={np.median(gaps) if len(gaps) else 0}d\tvmag={np.min(vm):.2f}..{np.max(vm):.2f}\tseqs_multi_epoch={len(seq)}\tseq_arcsec_med/max={np.median(seq[:,0]) if len(seq) else 0:.4g}/{np.max(seq[:,0]) if len(seq) else 0:.4g}\tseq_rate_arcsec_day_med/max={np.median(seq[:,2]) if len(seq) else 0:.4g}/{np.max(seq[:,2]) if len(seq) else 0:.4g}')
