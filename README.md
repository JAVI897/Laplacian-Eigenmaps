# Laplacian Eigenmaps

Laplacian Eigenmaps is another method for non-linear dimensionality reduction. It was proposed in 2003 by  Mikhail Belkin and Partha Niyogi. LE constructs embeddings based on the properties of the Laplacian matrix. At that time, Laplacian matrix was widely used for clustering problems (Spectral Clustering, for instance), but LE was the first algorithm that used the Laplacian matrix for dimensionality reduction. For a more detailed description on how the algorithm works see;

> Belkin, Niyogi. *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*. Neural Computation 15, 1373–1396 (2003)

Or you can also read my introduction to Laplacian Eigenmaps 😉; [Laplacian Eigenmaps](https://javi897.github.io/Laplacian_eigenmaps/)

In this project it has been implemented a version of LEE. However, this implementation is not computationally optimal, this repository is primarily for research purposes.

### Requirements

- numpy

### Usage

The algorithm is implemented as a python class called LE.

###### Parameters

**X**: {array-like, sparse matrix}, shape (n_samples, n_features). Data matrix.

**dim**: number of components to extract.

**k**: number of neighbours. Only used if graph = 'k-nearest'.

**eps**: epsilon hyperparameter. Only used if graph = 'eps'.



