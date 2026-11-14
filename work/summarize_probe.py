from pathlib import Path
from astropy.table import Table
import numpy as np
for f in sorted(Path('pipeline/cache/probe').glob('*_r*deg_mr10000.ecsv')):
    if not f.name.startswith('sia_'):
        continue
    t=Table.read(f, format='ascii.ecsv')
    ids=set(str(x) for x in t['obs_id'] if str(x) not in ('', 'None', 'masked'))
    def bounds(k):
        a=np.asarray(np.ma.asarray(t[k]).compressed(), dtype=float)
        return f'{a.min():.6f}..{a.max():.6f}' if len(a) else 'none'
    print(f'{f.name}\trows={len(t)}\tunique_obs_id={len(ids)}\tt_min={bounds("t_min")}\tt_max={bounds("t_max")}\tem_min={bounds("em_min")}\tem_max={bounds("em_max")}')

