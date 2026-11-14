from astropy.table import Table
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
f=Path('pipeline/cache/probe/most_20000002_Pallas_large-brightness-risk.ecsv'); t=Table.read(f)
print('mjd_start',np.min(t['mjd_start']),np.max(t['mjd_start']),'nuniq',len(np.unique(t['mjd_start'])))
for oid in np.unique(t['observationid'])[:3]:
 ix=np.where(np.asarray(t['observationid']).astype(str)==str(oid))[0]
 print('\nobsid',oid,'rows',len(ix),'times',np.unique(t['mjd_start'][ix])[:10])
 for i in ix[:5]: print(t['mjd_start'][i],t['ra_obj'][i],t['dec_obj'][i],t['vmag'][i],t['planeid'][i])
# all timeline sorted unique time
m=np.asarray(t['mjd_start'],dtype=float); ra=np.asarray(t['ra_obj'],dtype=float); de=np.asarray(t['dec_obj'],dtype=float)
ix=np.argsort(m); _, first=np.unique(m[ix],return_index=True); ix=ix[first]
c=SkyCoord(ra[ix]*u.deg,de[ix]*u.deg); dt=np.diff(m[ix]); sep=c[:-1].separation(c[1:]).arcsec
print('\nunique times',len(ix),'dt days',np.nanmin(dt),np.nanmedian(dt),np.nanmax(dt),'sep arcsec',np.nanmin(sep),np.nanmedian(sep),np.nanmax(sep),'rates arcsec/day',np.nanmedian(sep/dt),np.nanmax(sep/dt))
print('first 20 trajectory')
for i in ix[:20]: print(t['mjd_start'][i],t['ra_obj'][i],t['dec_obj'][i])
