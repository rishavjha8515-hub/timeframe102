from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
files=sorted(Path('pipeline/cache/fits-inspection').glob('*.fits'))
for p in files:
 print('\nFILE',p.name)
 with fits.open(p,memmap=False) as h:
  image=h['IMAGE']
  keys=[k for k in image.header if any(s in k for s in ('WCS','CTYPE','CUNIT','CRVAL','CRPIX','CD1_','CD2_','PC1_','PC2_','PS1_','PS2_','PV1_','PV2_','SIP'))]
  print('IMAGE_WCS_CARDS', {k:image.header[k] for k in keys})
  wave=h['WCS-WAVE'].data['VALUES'][0]
  print('WCS_WAVE_CENTRAL_UM',float(np.min(wave[...,0])),float(np.max(wave[...,0])))
  print('WCS_WAVE_BANDWIDTH_UM',float(np.min(wave[...,1])),float(np.max(wave[...,1])))
  epsf=h['EPSF'].data
  print('EPSF_ROWS',len(epsf),'EPSF_SHAPE',epsf['EPSF'].shape,'CWAVE',float(np.min(epsf['CWAVE'])),float(np.max(epsf['CWAVE'])),'CBAND',float(np.min(epsf['CBAND'])),float(np.max(epsf['CBAND'])))
  print('EPSF_OTHER_COLUMNS',h['EPSF'].columns.names)
