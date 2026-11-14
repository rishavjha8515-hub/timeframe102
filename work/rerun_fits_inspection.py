from pathlib import Path
from pipeline.fetch_and_inspect import inspect_fits
for path in sorted(Path('pipeline/cache/fits-inspection').glob('*.fits')):
    inspect_fits(path)
