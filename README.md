# mylib

A collection of useful functions written in the Python Programming Language. The library is primarily focused on groundwater modelling.

Currently the collection includes:

1. `heat_transport` : Heat transport equations to estimate the temperature at depth dependent on a groundwater flux value.
2. `hydro_funcs` : Functions to calculate useful hydrological properties.
3. `signal` : Functions for working with temperature time series.


## To install

```bash
git clone https://github.com/robinkeegan/mylib
cd mylib
python setup.py install
```

## Future Directions and Todo
1. Implement a more robust amplitude extraction algorithm
2. Add some tests
