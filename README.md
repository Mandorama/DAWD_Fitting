# DAWD_Fitting
An experimental code to fit single DA White dwarf spectras that I use in my Master's.

The model grid included comes from Koester, 2010 and can be found [here](https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=koester2).

Two python files are present: Fitting.py is the main code used to fit DA spectra by comparing its normalized flux with given models by using χ². SpectrumClasses contains mainly classes with methods to manipulate spectra.

## Example usage

The code can be controled through the command line. The expression bellow fits a single spectra and plots the χ² surface.

```
python3 Fitting.py -s ./Spectra_Examples/1237656495115010166.spec -p
```
The codes assumes BP-RP photometry to distinguish between multiple minima, but can also use the colors from SDSS:
```
python3 Fitting.py -s ./Spectra_Examples/1237679317478211623.spec -c u-g
```
Using -d, one can fit all .spec files in a directory:
```
python3 Fitting.py -d ./Spectra_Examples
```

The code is far from finished, I hope to extend it to more spectral classes and improve many of its functionalities. E.g., the values given for the uncertainties, still far from ideal.
