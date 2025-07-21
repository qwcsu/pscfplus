# PSCF+ -- Polymer Self-Consistent Field Theory (C++/CUDA) with Improved and Extended Capabilities
PSCF+ is a software package for solving the polymer self-consistent field (SCF) theory in continuum. 
It is based on the nice GPU framework of [PSCF](https://github.com/dmorse/pscfpp) 
(which is only for the "standard" or known as the Edwards-Helfand model, *i.e.*,
incompressible melts of continuous Gaussian chains with the Dirac δ-function interactions, commonly used in
polymer field theories) originally developed by Prof. David Morse and co-workers, but is improved with better
numerical methods, less GPU memory usage and more flexible algorithms, and is extended to various discrete-chain models. 
Similar to the C++/CUDA version of PSCF, PSCF+ is written primarily in C++ with
GPU accelerated code in CUDA.

Same as the C++/CUDA version of PSCF, PSCF+ is applicable to mixtures containing arbitrary acyclic copolymers,
and preserves all of the nice features already implemented in the former, including the use of 
[cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) 
on GPU, and the use of Anderson mixing (which is performed on GPU) combined with a variable-cell scheme to simultaneously solve
the nonlinear SCF equations and find the bulk periodicity for the ordered phases formed by block copolymer self-assembly 
(which speeds-up the calculation by about one order of magnitude). Their differences and advantages of the latter include:

- PSCF is only applicable to the "standard" model, while PSCF+ can also be applied to various discrete-chain
  models with finite-range non-bonded interactions commonly used in molecular simulations, thus providing the
  mean-field reference results for such simulations; see 
  [Models.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/Models.pdf) for more details.

- For the continuous-Gaussian-chain model, PSCF+ uses the Richardson-extrapolated pseudo-spectral
  methods (denoted by REPS-*K* with *K*=0,1,2,3,4) to solve the modified diffusion equations (which is the crux of
  SCF calculations of such models), while PSCF uses only REPS-1. A larger *K*-value gives more accurate result (thus reducing the GPU memory usage) at larger computational cost; see [REPS.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/REPS.pdf) for more details.

- In SCF calculations the (one-end-integrated) forward and backward propagators 
  $q$ and $q^\dagger$ of each block
  usually take the largest memory usage, but the GPU memory is rather limited. While in PSCF the size of
  these propagators is $n_{\bold r}n_s$, where $n_{\bold r}$ denotes the number of grid points in real space and $n_s$ the number of
  contour discretization points on a continuous Gaussian chain (or the number of segments on a discrete
  chain), in PSCF+ the "slice" algorithm proposed by Li and Qiang can be used to reduce the size of $q$ to
  $\mathbf{n_{\bold {r}}\sqrt{n_s}}$
  and that of $q^\dagger$ to just $n_{\bold r}$, thus greatly reducing the GPU memory usage at the cost of computing $q$
  twice; see [SavMem.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/SavMem.pdf) for more details.

- While propagators for different blocks having the same segment type, block length and initial condition are
  calculated and stored seperately in PSCF, PSCF+ automatically avoids such redundant calculations and GPU memory usage
  based on the chain architecture. One example is the forward propagators (starting from a free chain end) of
  the side chains of a bottlebrush polymer, which are usually all the same; calculation and storage of such propagators are therefore independent of the number of side chains in PSCF+.

- For 3D spatially periodic ordered phases such as those formed by block copolymer self-assembly, while
  PSCF uses fast Fourier transforms (FFTs) between a uniform grid in the real space and that in the reciprocal
  space, for the Pmmm supergroup PSCF+ uses discrete cosine transforms instead of FFTs to take advantage
  of the (partial) symmetry of an ordered phase to reduce the number of grid points, thus both speeding up the
  calculation and reducing the GPU memory usage; see 
  [Qiang and Li, Macromolecules 53, 9943 (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.0c01974) for more
  details.

- The approach used by PSCF to solve the SCF equations (for an incompressible system) does not allow any
  athermal species (having no Flory-Huggins-type interactions with all other species) in the system. This
  problem is solved in PSCF+; see [SlvSCF.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/SlvSCF.pdf) for more details.

- Since SCF equations are highly nonlinear, having a good initial guess is very important in practice as it
  determines not only which final solution (corresponding to a phase in block copolymer self-assembly) can be
  obtained but also how many iteration steps the solver (*e.g.*, the Anderson mixing) takes to converge these
  equations. PSCF+ uses automated calculation along a path (ACAP), where the converged solution at a neighboring point is taken as the initial guess at the current point in the parameter space. While this is similar
  to the "SWEEP" command in PSCF, the key for ACAP to be successful and efficient is that it automatically
  adjusts the step size along the path connecting the two points. Most importantly, in PSCF+, ACAP is further combined with the
  phase-boundary calculation between two specified phases, making the construction of phase diagrams very
  efficient. See [ACAP.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/ACAP.pdf) for more details.

## Contributors
- Qiang (David) Wang
- Juntong He


