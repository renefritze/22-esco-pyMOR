---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"slideshow": {"slide_type": "slide"}}

<center><img src="img/pymor_logo.png" width="70%"></center>

# Model Order Reduction with pyMOR -- an Interactive Crash Course

## ESCO 2022

## René Fritze

+++ {"slideshow": {"slide_type": "slide"}}

<div class="container">

<div>

# Get the slides

## <https://rene.fritze.me/22-esco-pyMOR>

README has a link to execute the presentation on mybinder.org

</div>

<div>
<img src="qr_self.png" />
</div>
</div>

+++ {"slideshow": {"slide_type": "slide"}}

<div class="container">

<div>

# pyMOR School 2022

## August 22-26th

## Magdeburg, Germany

## <https://school.pymor.org/>

- 5 day course on pyMOR
- no participation fee
- registration open now

</div>

<div>
<img src="qr_school_website.png" />
</div>
</div>

+++ {"slideshow": {"slide_type": "slide"}}

<div class="container">

<div>

# Documentation

<https://docs.pymor.org>

- (interactive) [Tutorials](https://docs.pymor.org/2021-2-0/tutorials.html)
- [Design](https://docs.pymor.org/2021-2-0/technical_overview.html) overview
- [API](https://docs.pymor.org/2021-2-0/autoapi/index.html) reference

</div>

<div>
<img src="qr_docs.png" />
</div>
</div>

+++ {"slideshow": {"slide_type": "slide"}}

# What is pyMOR?

+++ {"slideshow": {"slide_type": "fragment"}}

- a software library for writing **M**odel **O**rder **R**eduction applications

+++ {"slideshow": {"slide_type": "fragment"}}

- in the **py**thon programming language.

+++ {"slideshow": {"slide_type": "fragment"}}

- BSD-licensed, fork us on [Github](https://github.com/pymor/pymor).

+++ {"slideshow": {"slide_type": "fragment"}}

- Started 2012, 22k lines of code, 7k commits.

+++ {"slideshow": {"slide_type": "slide"}}

### MOR??

## Have

a (parameterized) PDE/LTI system

## Need to solve

1. for lots of parameters/inputs
2. really fast for a single parameter

+++ {"slideshow": {"slide_type": "slide"}}

### MOR??

## Solution

Build a lower dimensional approximation for the solution space,
solve there instead.

+++ {"slideshow": {"slide_type": "slide"}}

### Design Goal 1

## One library for algorithm development *and* large-scale applications

+++ {"slideshow": {"slide_type": "fragment"}}

- Small NumPy/SciPy-based discretization toolkit for easy prototyping.
- `VectorArray`, `Operator`, `Model` interfaces for seamless integration with high-performance PDE solvers.

+++ {"slideshow": {"slide_type": "slide"}}

### Design Goal 2

## Unified view on MOR

+++ {"slideshow": {"slide_type": "fragment"}}

- Implement RB and system-theoretic methods in one common language.

+++ {"slideshow": {"slide_type": "slide"}}

## Implemented Algorithms

- Gram-Schmidt, POD, HAPOD.
- Greedy basis generation with different extension algorithms.
- Automatic (Petrov-)Galerkin projection of arbitrarily nested affine combinations of operators.
- Interpolation of arbitrary (nonlinear) operators, EI-Greedy, DEIM.
- A posteriori error estimation.

+++ {"slideshow": {"slide_type": "slide"}}

## Implemented Algorithms

- System theory methods: balanced truncation, IRKA, ...
- Iterative linear solvers, eigenvalue computation, Newton algorithm, time-stepping algorithms.
- Non-intrusive MOR using artificial neural networks.

+++ {"slideshow": {"slide_type": "slide"}}

## PDE Solvers: Official Support

- [deal.II](https://dealii.org)
- [FEniCS](https://fenicsproject.org)
- [NGSolve](https://ngsolve.org)

+++ {"slideshow": {"slide_type": "slide"}}

## PDE Solvers: Used with

- [DUNE](https://dune-project.org)
- [preCICE](https://precice.org)
- [BEST](https://www.itwm.fraunhofer.de/en/departments/sms/products-services/best-battery-electrochemistry-simulation-tool.html)
- [GridLOD](https://github.com/fredrikhellman/gridlod)
- file I/O, e.g. [COMSOL](https://comsol.com)
- ...

+++ {"slideshow": {"slide_type": "slide"}}

## pyMOR Development

### Main Developers

<table><tr>
<td><img src="img/balicki.png"></td>
<td><img src="img/fritze.jpg"></td>
<td><img src="img/mlinaric.jpeg"></td>
<td><img src="img/rave.jpg"></td>
<td><img src="img/schindler.jpg"></td>
</tr>
</table>

+++ {"slideshow": {"slide_type": "slide"}}

### Contributions

- everyone can/should(!) contribute
- everyone can become main developer
- not just code contributions are valued!

+++ {"slideshow": {"slide_type": "slide"}}

## Minimal pyMOR installation

  ```
  python -m pip install pymor
  ```

+++ {"slideshow": {"slide_type": "slide"}}

## Minimal pyMOR installation

  ```
  conda install -c conda-forge pymor
  ```

+++ {"slideshow": {"slide_type": "slide"}}

## Hello pyMOR

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import pymor
pymor.config
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
from pymor.basic import *
print_defaults()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Subpackages of the pymor Package

|                                |                                                      |
| :-                             | :-                                                   |
| **`pymor.algorithms`**         | generic algorithms                                   |
| **`pymor.analyticalproblems`** | problem descriptions for use with discretizers       |
| `pymor.bindings`               | bindings to external solvers                         |
| `pymor.core`                   | base classes/caching/defaults/logging                |
| **`pymor.discretizers`**       | create `Models` from `analyticalproblems`            |
| **`pymor.models`**             | `Model` interface/implementations                    |

+++ {"slideshow": {"slide_type": "slide"}}

|                                |                                                      |
| :-                             | :-                                                   |
| **`pymor.operators`**          | `Operator` interface/constructions                   |
| `pymor.parallel`               | `WorkerPools` for parallelization                    |
| **`pymor.parameters`**         | parameter support/`ParameterFunctionals`             |
| **`pymor.reductors`**          | most MOR algorithms (rest in `pymor.algorithms`)     |
| `pymor.tools`                  | non MOR-specific support code (pprint/floatcmp, ...) |
| **`pymor.vectorarrays`**       | `VectorArray` interface/implementations              |

+++ {"slideshow": {"slide_type": "slide"}}

# Reduced Basis Methods with pyMOR

+++ {"slideshow": {"slide_type": "slide"}}

## Building the Full Order Model (FOM)

+++ {"slideshow": {"slide_type": "slide"}}

### The Thermal Block Problem

Solve:

\begin{align}
- \nabla \cdot [d(x, \mu) \nabla u(x, \mu)] &= f(x),  & x &\in \Omega,\\
                                  u(x, \mu) &= 0,     & x &\in \partial\Omega,
\end{align}

where

\begin{align}
d(x, \mu) &= \sum_{q=1}^Q \mathbb{1}_{\Omega_q}, \\
f(x)      &= 1.
\end{align}

satisfying $\overline{\Omega} = \overline{\dot{\bigcup}_{i=1}^{Q} \Omega_q}$.

+++ {"slideshow": {"slide_type": "slide"}}

### Setting up an analytical description of the thermal block problem

The thermal block problem already comes with pyMOR:

```{code-cell} ipython3
from pymor.basic import *
p = thermal_block_problem([2,2])
```

+++ {"slideshow": {"slide_type": "fragment"}}

Our problem is parameterized:

```{code-cell} ipython3
p.parameters
```

+++ {"slideshow": {"slide_type": "slide"}}

### Looking at the definition

We can easily look at the definition of `p` by printing its `repr`:

```{code-cell} ipython3
p
```

+++ {"slideshow": {"slide_type": "slide"}}

### Building a discrete model

We use the builtin discretizer `discretize_stationary_cg` to compute a finite-element discretization of the problem:

```{code-cell} ipython3
from pymor.basic import *
fom, data = discretize_stationary_cg(p, diameter=1/100)
```

+++ {"slideshow": {"slide_type": "fragment"}}

`fom` is a `Model`. It has the same `Parameters` as `p`:

```{code-cell} ipython3
fom.parameters
```

+++ {"slideshow": {"slide_type": "slide"}}

### Looking at the model

`fom` inherits its structure from `p`:

```{code-cell} ipython3
fom
```

+++ {"slideshow": {"slide_type": "slide"}}

### Note

> Using an `analyticalproblem` and a `discretizer` is just one way
  to build the FOM.

> Everything that follows works the same for a FOM built using an external PDE solver.

+++ {"slideshow": {"slide_type": "slide"}}

### Solving the FOM

Remember the FOM's parameters:

```{code-cell} ipython3
fom.parameters
```

+++ {"slideshow": {"slide_type": "fragment"}}

To `solve` the FOM, we need to specify values for those parameters:

```{code-cell} ipython3
U = fom.solve({'diffusion': [1., 0.01, 0.1, 1]})
```

+++ {"slideshow": {"slide_type": "fragment"}}

`U` is a `VectorArray`, an ordered collection of vectors of the same dimension:

```{code-cell} ipython3
U
```

+++ {"slideshow": {"slide_type": "fragment"}}

> There is not the notion of a single vector in pyMOR! Don't try to get hold of one!

+++ {"slideshow": {"slide_type": "slide"}}

### Some words about VectorArrays

Each `VectorArray` has a length, giving you the number of vectors:

```{code-cell} ipython3
len(U)
```

+++ {"slideshow": {"slide_type": "fragment"}}

Its dimension gives you the *uniform* size of each vector in the array:

```{code-cell} ipython3
U.dim
```

+++ {"slideshow": {"slide_type": "slide"}}

### Supported Operations

|                  |                                                        |
| :-               | :-                                                     |
| `+`/`-`/`*`      | elementwise addition/subtraction/scalar multiplication |
| `inner`          | matrix of inner products between all vectors           |
| `pairwise_inner` | list of pairwise inner products                        |
| `lincomb`        | linear combination of the vectors in the array         |
| `scal`           | in-place scalar multiplication                         |
| `axpy`           | in-place BLAS axpy operation                           |
| `dofs`           | return some degrees of freedom as NumPy array          |
| `norm`           | list of norms                                          |
| `append`         | append vectors from another array                      |

+++ {"slideshow": {"slide_type": "slide"}}

### Playing a bit with VectorArrays

> All `VectorArrays` are created by their `VectorSpace`

```{code-cell} ipython3
V = fom.solution_space.empty()
```

Let's accumulate some solutions:

```{code-cell} ipython3
for mu in p.parameter_space.sample_randomly(10):
    V.append(fom.solve(mu))
```

+++ {"slideshow": {"slide_type": "slide"}}

### Indexing

We can index a `VectorArray` using numbers, sequences of numbers, or slices, e.g.:

```{code-cell} ipython3
V_indexed = V[3:6]
```

Indexing **always** create a view on the original array:

```{code-cell} ipython3
print(V_indexed.is_view)
V_indexed *= 0
V.norm()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Looking at the solution

We can use the `visualize` method to plot the solution:

```{code-cell} ipython3
fom.visualize(U)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Looking at the solution

An array with multiple elements is visualized as a time-series:

```{code-cell} ipython3
fom.visualize(V)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Is the solution really a solution?

We compute the residual:

```{code-cell} ipython3
mu = fom.parameters.parse([1., 0.01, 0.1, 1])
U = fom.solve(mu)
(fom.operator.apply(U, mu=mu) - fom.rhs.as_vector(mu)).norm()
```

> If you implement a `Model`, make sure that `solve` really returns solutions with zero residual!

+++ {"slideshow": {"slide_type": "slide"}}

## Reducing the FOM

+++ {"slideshow": {"slide_type": "slide"}}

### Building an approximation space

As before, we compute some random solution **snapshots** of the FOM, which will
span our **reduced** approximation space:

```{code-cell} ipython3
snapshots = fom.solution_space.empty()
for mu in p.parameter_space.sample_randomly(10):
    snapshots.append(fom.solve(mu))
```

+++ {"slideshow": {"slide_type": "slide"}}

### Building an approximation space

It's a good idea to orthonormalize the basis:

```{code-cell} ipython3
basis = gram_schmidt(snapshots)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Projecting the Model

In pyMOR, ROMs are built using a `Reductor`. Let's pick the most basic `Reductor`
available for a `StationaryModel`:

```{code-cell} ipython3
reductor = StationaryRBReductor(fom, basis)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Projecting the Model

Every reductor has a `reduce` method, which builds the ROM:

```{code-cell} ipython3
rom = reductor.reduce()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Comparing ROM and FOM

```{code-cell} ipython3
fom
```

+++ {"slideshow": {"slide_type": "slide"}}

### Comparing ROM and FOM

```{code-cell} ipython3
rom
```

+++ {"slideshow": {"slide_type": "slide"}}

### Solving the ROM

To solve the ROM, we just use `solve` again,

```{code-cell} ipython3
mu = fom.parameters.parse([1., 0.01, 0.1, 1])
u_rom = rom.solve(mu)
```

+++ {"slideshow": {"slide_type": "fragment"}}

to get the reduced coefficients:

```{code-cell} ipython3
u_rom
```

+++ {"slideshow": {"slide_type": "fragment"}}

A high-dimensional representation is obtained from the `reductor`:

```{code-cell} ipython3
U_rom = reductor.reconstruct(u_rom)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Computing the MOR error

Let's compute the error:

```{code-cell} ipython3
U = fom.solve(mu)
ERR = U - U_rom
ERR.norm() / U.norm()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Computing the MOR error

and look at it:

```{code-cell} ipython3
fom.visualize(ERR)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Certified Reduced Basis Method

Let's use use a more sophisticated `reductor` which assembles an efficient
upper bound for the MOR error:

```{code-cell} ipython3
reductor = CoerciveRBReductor(
   fom,
   product=fom.h1_0_semi_product,
   coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Certified Reduced Basis Method

and build a basis using a greedy search over the parameter space:

```{code-cell} ipython3
training_set = p.parameter_space.sample_uniformly(4)
print(training_set[0])
```

+++ {"slideshow": {"slide_type": "slide"}}

### Certified Reduced Basis Method

```{code-cell} ipython3
greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=20)
print(greedy_data.keys())
rom = greedy_data['rom']
```

+++ {"slideshow": {"slide_type": "slide"}}

### Testing the ROM

Let's compute the error again:

```{code-cell} ipython3
mu = p.parameter_space.sample_randomly()
U = fom.solve(mu)
u_rom = rom.solve(mu)
ERR = U - reductor.reconstruct(u_rom)
ERR.norm(fom.h1_0_semi_product)
```

+++ {"slideshow": {"slide_type": "fragment"}}

and compare it with the estimated error:

```{code-cell} ipython3
rom.estimate_error(mu)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Is it faster?

Finally, we compute some timings:

```{code-cell} ipython3
from time import perf_counter
mus = p.parameter_space.sample_randomly(10)
tic = perf_counter()
for mu in mus:
    fom.solve(mu)
t_fom = perf_counter() - tic
tic = perf_counter()
for mu in mus:
    rom.solve(mu)
t_rom = perf_counter() - tic
```

+++ {"slideshow": {"slide_type": "slide"}}

### Yes, it is

```{code-cell} ipython3
print(f'Speedup: {t_fom/t_rom}')
```

+++ {"slideshow": {"slide_type": "slide"}}

### Some things to try

- Plot the MOR error vs. the dimension of the reduced space.

- Plot the speedup vs. the dimension of the reduced space.

- Compute the maximum/minimum efficiency of the error estimator over the parameter space.

- Try different numbers of subdomains.

+++ {"slideshow": {"slide_type": "slide"}}

# System-Theoretic Methods with pyMOR

+++ {"slideshow": {"slide_type": "slide"}}

## Building the FOM

+++ {"slideshow": {"slide_type": "fragment"}}

### Linear Time-Invariant (LTI) System

\begin{align}
  \dot{x}(t) & = A x(t) + B u(t), \quad x(0) = 0, \\
  y(t) & = C x(t),
\end{align}

- $u(t) \in \mathbb{R}^m$ is the input,
- $x(t) \in \mathbb{R}^n$ is the state,
- $y(t) \in \mathbb{R}^p$ is the output.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
import numpy as np
from pymor.basic import *
n = 100
np.random.seed(0)
A = np.random.randn(n, n)
A -= (np.linalg.eigvals(A).real.max() + 0.1) * np.eye(n)
B = np.random.randn(n, 1)
C = np.random.randn(1, n)
lti = LTIModel.from_matrices(A, B, C)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
lti
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(lti)
```

+++ {"slideshow": {"slide_type": "slide"}}

### FOM Poles

```{code-cell} ipython3
import matplotlib.pyplot as plt
poles = lti.poles()
_ = plt.plot(poles.real, poles.imag, '.')
```

+++ {"slideshow": {"slide_type": "slide"}}

### FOM Hankel Singular Values

```{code-cell} ipython3
_ = plt.semilogy(range(1, n + 1), lti.hsv(), '.-')
```

+++ {"slideshow": {"slide_type": "slide"}}

### FOM Bode Plot

```{code-cell} ipython3
w = np.logspace(-4, 4, 200)
_ = lti.bode_plot(w)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Balanced Truncation

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
bt = BTReductor(lti)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rom_bt = bt.reduce(10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
rom_bt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(rom_bt)
```

```{code-cell} ipython3
rom_bt.to_matrices()
```

+++ {"slideshow": {"slide_type": "slide"}}

### ROM poles

```{code-cell} ipython3
poles = lti.poles()
poles_rom = rom_bt.poles()
_ = plt.plot(poles.real, poles.imag, '.')
_ = plt.plot(poles_rom.real, poles_rom.imag, 'x')
```

+++ {"slideshow": {"slide_type": "slide"}}

### ROM Hankel Singular Values

```{code-cell} ipython3
_ = plt.semilogy(range(1, n + 1), lti.hsv(), '.-')
_ = plt.semilogy(range(1, 11), rom_bt.hsv(), '.-')
```

+++ {"slideshow": {"slide_type": "slide"}}

### ROM Bode Plot

```{code-cell} ipython3
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(6, 8), tight_layout=True)
_ = lti.bode_plot(w, ax=ax)
_ = rom_bt.bode_plot(w, ax=ax, linestyle='--')
```

+++ {"slideshow": {"slide_type": "slide"}}

### Error Magnitude Plot

```{code-cell} ipython3
err_bt = lti - rom_bt
_ = err_bt.mag_plot(w)
```

+++ {"slideshow": {"slide_type": "fragment"}}

### Relative Errors

```{code-cell} ipython3
print(err_bt.hinf_norm() / lti.hinf_norm())
print(err_bt.h2_norm() / lti.h2_norm())
```

+++ {"slideshow": {"slide_type": "slide"}}

<div class="container">

<div>

# Get the slides

## <https://rene.fritze.me/22-esco-pyMOR>

README has a link to execute the presentation on mybinder.org

</div>

<div>
<img src="qr_self.png" />
</div>
</div>
