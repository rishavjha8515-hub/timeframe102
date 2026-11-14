from astropy.table import Table
from pathlib import Path
for f in sorted(Path('pipeline/cache/probe').glob('most_*.ecsv')):
 t=Table.read(f, format='ascii.ecsv')
 print('\n', f.name)
 for c in ['observationid','start_date','end_date','mjd_start','mjd_end','vmag','ra_obj','dec_obj','instrument','uri']:
  x=t[c][0]
  print(c,repr(x), 'unit=',getattr(t[c],'unit',None))
