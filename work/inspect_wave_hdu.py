from pathlib import Path
from astropy.io import fits
for p in sorted(Path('pipeline/cache/fits-inspection').glob('*.fits')):
 print('\nFILE',p.name)
 with fits.open(p,memmap=False) as hdus:
  for i,h in enumerate(hdus):
   print('HDU',i,'name=',h.name,'shape=',None if h.data is None else h.data.shape,'dtype=',None if h.data is None else h.data.dtype)
   if h.name=='WCS-WAVE':
    print('WCS-WAVE_HEADER_START')
    print(h.header)
    print('WCS-WAVE_DATA',repr(h.data)[:12000])
    print('WCS-WAVE_COLUMNS',getattr(getattr(h,'columns',None),'names',None))
