from pathlib import Path
from astropy.table import Table
for p in sorted(Path('pipeline/cache/probe').glob('most_*.ecsv')):
 t=Table.read(p)
 print(p.name.split('_')[2], 'rows',len(t),'target_name',sorted(set(map(str,t['target_name']))),'target_moving', sorted(set(map(str,t['target_moving']))))
