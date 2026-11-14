from astropy.table import Table
from pathlib import Path
for f in sorted(Path('pipeline/cache/probe').glob('most_*.ecsv')):
 t=Table.read(f, format='ascii.ecsv')
 print('\n', f.name, 'rows=', len(t))
 print('columns=', t.colnames)
 print(t[:2])
