import inspect
from astroquery.ipac.irsa.most import Most
print(inspect.getsource(Most.query_object))
print(inspect.getsource(Most._validate_input))
