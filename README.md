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

| Parameter         | Description                                                  |
| ----------------- | :----------------------------------------------------------- |
| **X**             | {array-like, sparse matrix}, shape (n_samples, n_features). Data matrix |
| **dim**           | number of components to extract                              |
| **graph**         | if set to 'k-nearest', two points are neighbours if one is the k nearest point of the other. If set to 'eps', two points are neighbours if their distance is less than epsilon |
| **eps**           | epsilon hyperparameter. Only used if graph = 'eps'. If is set to None, then epsilon is computed to be the minimum one which guarantees G to be connected |
| **k**             | number of neighbours. Only used if graph = 'k-nearest'       |
| **weights**       | if set to 'heat kernel', the similarity between two points is computed using the heat kernel approach. If set to 'rbf' the similarity between two points is computed using the gaussian kernel approach. If set to 'simple', the weight between two points is 1 if they are connected and 0 otherwise. |
| **sigma**         | coefficient for gaussian kernel or heat kernel               |
| **laplacian**     | if set to 'unnormalized', eigenvectors are obtained by solving the generalized eigenvalue problem Ly = λDy where L is the unnormalized laplacian matrix. If set to 'random', eigenvectors are obtained by decomposing the Random Walk Normalized Laplacian matrix. If set to 'symmetrized', eigenvectors are obtained by decomposing the Symmetrized Normalized Laplacian |
| **opt_eps_jumps** | increasing factor for epsilon                                |



