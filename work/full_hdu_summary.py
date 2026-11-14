from pathlib import Path
from astropy.io import fits
import numpy as np
keys=['EXTNAME','EXTTYPE','BUNIT','DATE-OBS','MJD-OBS','MJD-BEG','MJD-END','OBSID','CTYPE1','CTYPE2','CUNIT1','CUNIT2','RADESYS','CRVAL1','CRVAL2','CRPIX1','CRPIX2','PC1_1','PC1_2','PC2_1','PC2_2','WCSNAMEA','CTYPE1A','CTYPE2A','WCSNAMEW','CTYPE1W','CTYPE2W','CUNIT1W','CUNIT2W','PS1_0W','PS1_1W','PS1_2W','PS2_0W','PS2_1W','PS2_2W','DETECTOR','DETCOORD','TFIELDS','TTYPE1','TTYPE2','TTYPE3','TFORM1','TFORM2','TFORM3','TDIM1','TDIM2','TDIM3']
for p in sorted(Path('pipeline/cache/fits-inspection').glob('*.fits')):
 print('\nFILE',p.name,'bytes',p.stat().st_size)
 with fits.open(p,memmap=False) as h:
  for i,hd in enumerate(h):
   print('HDU',i,hd.name,'shape',None if hd.data is None else hd.data.shape)
   cards={k:hd.header[k] for k in keys if k in hd.header}
   if cards: print(cards)
   if isinstance(hd,fits.BinTableHDU):
    print('TABLE',[(c.name,c.format,c.dim) for c in hd.columns])
   if hd.name=='EPSF':
    for col in ['CWAVE','CBAND']:
     a=np.asarray(hd.data[col],dtype=float); print(col+'_RANGE',float(a.min()),float(a.max()))
    print('EPSF_COLUMN_SHAPE',hd.data['EPSF'].shape[1:])
   if hd.name=='WCS-WAVE':
    a=np.asarray(hd.data['VALUES'][0],dtype=float)
    print('WCS_WAVE_CENTRAL_BANDWIDTH_RANGES',[(float(a[...,j].min()),float(a[...,j].max())) for j in range(2)])
