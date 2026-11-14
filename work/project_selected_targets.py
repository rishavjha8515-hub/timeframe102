from pathlib import Path
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
selections=[
 ('Pallas','most_20000002_Pallas_large-brightness-risk.ecsv','level2_2025W18_2B_0055_1D4_spx_l2b-v27-2026-197.fits','first'),
 ('Pallas','most_20000002_Pallas_large-brightness-risk.ecsv','level2_2026W29_2A_0444_1D6_spx_l2b-v27-2026-277.fits','last'),
 ('Hygiea','most_20000010_Hygiea_large-brightness-risk.ecsv','level2_2025W38_2C_0242_1D4_spx_l2b-v27-2026-237.fits','first'),
]
for name,table_file,fits_file,which in selections:
 t=Table.read(Path('pipeline/cache/probe')/table_file)
 order=sorted(range(len(t)),key=lambda i:float(t['mjd_start'][i]))
 # unique URI rows
 seen=set(); rows=[]
 for i in order:
  uri=str(t['uri'][i])
  if uri not in seen: rows.append(i);seen.add(uri)
 i=rows[0] if which=='first' else rows[-1]
 ra,dec=float(t['ra_obj'][i]),float(t['dec_obj'][i])
 p=Path('pipeline/cache/fits-inspection')/fits_file
 with fits.open(p,memmap=False) as h:
  wc=WCS(h['IMAGE'].header, fobj=h).celestial
  x,y=wc.world_to_pixel(SkyCoord(ra*u.deg,dec*u.deg))
  x,y=float(x),float(y)
  ix,iy=int(round(x)),int(round(y))
  inside=0<=x<h['IMAGE'].data.shape[1] and 0<=y<h['IMAGE'].data.shape[0]
  flags=int(h['FLAGS'].data[iy,ix]) if inside else None
  var=float(h['VARIANCE'].data[iy,ix]) if inside else None
  flux=float(h['IMAGE'].data[iy,ix]) if inside else None
  print(name,which,'mjd=',float(t['mjd_start'][i]),'ra/dec=',ra,dec,'xy0=',x,y,'inside=',inside,'nearest=',ix,iy,'flags=',flags,'var=',var,'flux=',flux)

