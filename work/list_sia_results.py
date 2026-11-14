from pathlib import Path
from astropy.table import Table
for f in sorted(Path('pipeline/cache/probe').glob('sia_*_r*deg_mr10000.ecsv')):
 t=Table.read(f,format='ascii.ecsv')
 print(f.name, len(t),len(t.colnames),','.join(t.colnames))
