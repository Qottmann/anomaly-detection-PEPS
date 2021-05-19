# anomaly-detection-PEPS
## Deep anomaly detection for mapping out phase diagrams with PEPS  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4770558.svg)](https://doi.org/10.5281/zenodo.4770558)  

These are the complete notebooks to reproduce the plots in "Unsupervised mapping of phase diagrams of 2D systems from infinite projected entangled-pair states via deep anomaly detection" by Korbinian Kottmann, Philippe Corboz, Maciej Lewenstein and Antonio Acín (arxiv link tba).

The notebooks are nummerically ordered to reproduce Figs. 1-3. 

The data from the simulated PEPS (bond singular values and reduced density matrices) are in `data`

Intermediate results are saved for convenience in `data_results`.

`AD_tools.py` contains some functions that are used for the anomaly detection in all notebooks.

`plots` contains the resulting figures as well as additional images like training convergences.

`CNN_data` contains the parameters of the neural networks involved. Training is _very_ fast here, so there is not really a need for it.
