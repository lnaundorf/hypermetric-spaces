# hypermetric-spaces

This collection of python scripts can perform the following tasks:

## Counterexamples for hypermetric inequalities


This functionality is provided by the script ``k_hypermetric_test_iterative.py``.
The script needs three arguments:

* dimension of space d. The matrices used for the hypermetric inequality lie in the space R^{d \times d} or in the space Sym R^{d \times d} of symmetric matrices depending on the chosen norm
* degree of hypermetric inequality k
* name of norm. Possible values are "diff", "abs", "nuclear", "operator" and "euclidean"

More information on the specific norms can be obtained by invoking the script with no arguments.


## Generators for the orthogonal groups of degree 2 and 3

This functionality is provided by the script ``orthogonal_groups.py``.
The constructor of the ``OrthogonalGroup`` class needs to be called with the dimension (either 2 or 3) and the number of steps for each dimension.

## Implementation of some convex bodies

The ``convex_bodies.py`` script partially implements zonotopes and the generalized Schur-Horn orbitope of a zonotope. It is also possible to compute the support function of these special convex bodies.

## Matrix generation

The class ``MatrixGenerator`` in ``matrix_generation.py`` allows the creation of some random square matrices with a given entry range and further properties such as symmetrc matrices and diagonal matrices or only integer entries. These random matrices are used as perturbation matrices for the hypermetric inequalities.