from pathlib import Path
from astropy.io import fits
for p in sorted(Path('pipeline/cache/fits-inspection').glob('*.fits')):
 print('\n',p.name)
 with fits.open(p,memmap=False) as h:
  hd=h['FLAGS'].header
  for k in hd:
   if k.startswith(('MP_','L2 ','N_','EXT')) or k in {'BUNIT'}: print(k,repr(hd[k]))
  print('flag unique sample',len(set(h['FLAGS'].data.ravel().tolist())), 'top counts')
  import numpy as np
  vals,cnt=np.unique(h['FLAGS'].data,return_counts=True)
  print(sorted(zip(cnt.tolist(),vals.tolist()),reverse=True)[:12])
