# GeostatInversion

This package provides methods for doing inverse analysis with parameter fields that are modeled using geostatistics.
Currently, two methods are implemented.
One is the principal component geostatistical approach (PCGA) of [Kitanidis](http://dx.doi.org/10.1002/2013WR014630) & [Lee](http://dx.doi.org/10.1002/2014WR015483).
The other utilizes a randomized geostatistical approach (RGA) that builds on PCGA (a reference for this method is forthcoming).


Two versions of PCGA are implemented in this package

- `pcgadirect`, which uses full matrices and direct solvers during iterations
- `pcgalsqr`, which uses low rank representations of the matrices combined with iterative solvers during iterations

The RGA method, `rga`, can use either of these approaches using the keyword argument. That is, by doing `rga(...; pcgafunc=GeostatInversion.pcgadirect)` or `rga(...; pcgafunc=GeostatInversion.pcgalsqr)`.