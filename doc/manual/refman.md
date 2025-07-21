# PSCF+: Polymer Self-Consistent Field Theory (C++/CUDA) with Improved and Extended Capabilities

### Content

- [1. Overview](#1)
- [2. Installation](#2)
  - [2.1. System Requirements](#21)
  
  - [2.2. Source Code](#22)
  
  - [2.3. Environment Variables](#23)
  
  - [2.4. Compilation](#24)
- [3. User Guide](#3)
  - [3.1. Invoking an Executable](#31)
  - [3.2. Parameter Files](#32)
  - [3.3. Command Files](#33)
  - [3.4. Free-Energy Density and Its Components](#34)

## <p id="1"></p>

## 1. Overview

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
  these propagators is $n_\bold{r}n_s$, where $n_\bold{r}$ denotes the number of grid points in real space and $n_s$ the number of
  contour discretization points on a continuous Gaussian chain (or the number of segments on a discrete
  chain), in PSCF+ the "slice" algorithm proposed by Li and Qiang can be used to reduce the size of $q$ to
  $n_\bold{r}\sqrt{n_s}$
  and that of $q^\dagger$ to just $n_\bold{r}$, thus greatly reducing the GPU memory usage at the cost of computing $q$
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

## <a id="2"></a>2. Installation

### <a id="21"></a>2.1. System Requirements

The PSCF+ package provides programs that are designed to run on a desktop, laptop or cluster with an NVIDIA
GPU. PSCF+ is distributed only as source code, and must be compiled by the user. All source code is written in
ANSI 2011 C++ language standard with CUDA. Compilation of PSCF+ is controlled by a system of Unix makefiles
and a series of shell scripts. In order to compile all of the programs in the PSCF+ package, the system on which the
code is compiled must have:

- a NVIDIA graphics card,
- a C++ compiler (g++),
- a CUDA compiler (nvcc),
- a git version control client,
- a python interpreter,
- the GNU Scientific Library (GSL),
- the cuFFT GPU-accelerated fast Fourier transform library,
- JsonCpp, the C++ library that allows manipulating JSON values.

A git client is needed to obtain (clone) the source code, which is maintained in a git repository on the github.com
server. A python interpreter is needed during compilation (but not during execution) because the build system that
compiles the PSCF+ source code uses a few python scripts that are provided with the package. The GNU scientific
library is used by several programs within the package for linear algebra operations. The cuFFT library, which is
used extensively in SCF calculations here, is provided with recent versions of the CUDA development environment. 
The JsonCpp library is required for reading the command files (see [Command Files](#CommandFiles)) and writing 
the system free energy and its components (see the `"SinglePhaseSCF"` and `"ACAP"` blocks in [Command Files](#CommandFiles)).

### <a id="22"></a>2.2. Source Code

The source code for PSCF+ is hosted on the [github](https://github.com/) server project `csu/pscfplus`, and can be obtained by using a
git version-control manager to clone the public [PSCF+ git repository](https://github.com/qwcsu/pscfplus). Instructions below assume that a git client has already been
installed on your computer.

To obtain a working copy of the PSCF+ git repository, you should first change directory (`cd`) to the one you
want to contain the `pscfplus/` directory. From there, then enter the command

    git clone --recursive https://github.com/qwcsu/pscfplus.git

This should create a complete, working copy of the PSCF+ source code in a new subdirectory named `pscfplus/` under
the directory from which you ran the above command.
Hereafter, we assume that the root directory of the PSCF+ working copy is named `pscfplus/`. Any path that does not explicitly begin with the `pscfplus/` prefix should be interpreted as a relative path with respect to this directory.

Since PSCF+ is an extension of PSCF, it reuses most of the source code and adopts the [directory structure](https://dmorse.github.io/pscfpp-man/developer_source_page.html) and [build system](https://dmorse.github.io/pscfpp-man/developer_build_page.html) of PSCF with the following exceptions:

- `src/pspg/` contains code for mathematical utility written in CUDA, instead of the implementation of
  SCF calculation for the "standard"  model accelerated by GPU as in PSCF.
- `src/pgc` contains all code implementing SCF calculation for the continuous-Gaussian-chain model, using namespce `Pscf::Pspg::Continuous`.
- `src/pgd` contains all code implementing SCF calculation for discrete-chain models, using namespce `Pscf::Pspg::Discrete`.
- PSCF+ no longer contains source code used by the `pscf_fd` 1D finite element program (`src/fd1d`) and the `pscf_pc` C++ CPU programs (`src/pspc`) in PSCF.

With the above differences, the build directory becomes

        BLD_DIR/
            makefile
            config.mk
            configure
            util/
            pscf/
            pspg/
            pgc/
            pgd/

### <a id="23"></a>2.3. Environment Variables

To compile PSCF+ in a Unix environment, before compiling any code the user should modify the following Unix
environment variables:

- Add the `pscfplus/bin/` directory to the Unix `$PATH` shell environment variable (the shell command search
  path). By default, executable file created by the PSCF+ build system is installed in the `pscfplus/bin/` directory. The directory in which these files are located must be added to the user's `$PATH` variable in order to
  allow the Unix shell to find the executable file when it is invoked by name in a command executed from any
  other directory.

- Add the `pscfplus/lib/python` directory to the `$PYTHONPATH` environment variable (the python module
  search path). The `pscfplus/scripts/python` directory contains a python script that is used by the build
  system during compilation to generate information about dependencies among C++ files. This directory must
  be added to the `$PYTHONPATH` variable in order to allow the python interpreter to find this file.

To make these changes using a bash shell, add some variant of the following lines to the `.profile` or `.bash_profile` file in your user home directory:

    > PSCFPLUS_DIR=${HOME}/pscfplus
    > export PATH=${PATH}:/${PSCFPLUS_DIR}/bin    
    > export PYTHONPATH=${PYTHONPATH}:/${PSCFPLUS_DIR}/scripts/python

The value of `PSCFPLUS_DIR` should be set to the path to the PSCF+ root directory (*i.e.*, the root of the directory
tree created by cloning the PSCF+ git repository). In the above fragment, as an example, it is assumed that this is a
subdirectory named `pscfplus/` within the user's home directory. After adding an appropriate variant of these lines to `.profile` or `.bash_profile`, re-login, and then enter `echo $PATH` and `echo $PYTHONPATH` to make sure that these variables have been set correctly.

### <a id="24"></a>2.4. Compilation

Below are the instructions for compiling the PSCF+ program with examples. It is assumed that you have cloned the
PSCF+ git repository and installed all required dependencies, and that the root directory of your working copy is named `pscfplus/`.

- **Set environment variables:** Modify the user's `$PATH` and `$PYTHONPATH` Unix environment variables, as
  discussed [here](#EnvironmentVariables).

- **Navigate to root directory:** Change directory (`cd`) to the `pscfplus/` root directory.

- **Setup:** Invoke the `setup` script from the `pscfplus/` root directory using the command
  
        > ./setup
  
    to setup the build system.

- **Change directory to the build directory:** Change directory (`cd`) to the `pscfplus/bld` subdirectory.

- **Compile the PSCF+ program for a given model system:** From `pscflus/bld`, enter
  
        > bash compile.sh [-B CHN] [-N NBP] [-C] [-D] [-K K]
  
    This will generate a large number of intermediate object (`.o`), dependency (`.d`) and library (`.a`) files in
  subdirectories of the `pscfplus/bld` directory, and install the executables in the `pscfplus/bin` directory.
  The options in the above command are as follows:
  
  - `-B CHN`: Specifying the model of chain connectivity; see 
    [Models.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/Models.pdf) for details. By default the continuous-Gaussian-chain model is used; otherwise `CHN` can be
    
          DGC: discrete-Gaussian-chain model
          FJC: freely-jointed-chain model
  
  - `-N NBP`: Specifying the form of non-bonded pair potential;
    see [Models.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/Models.pdf) for details. By default the Dirac δ-function potential is used; otherwise `NBP` can be
    
          G: Gaussian potential
          DPD: dissipative particle dynamics potential
          SS: soft-sphere potential
  
  - `-C`: Specifying a compressible system; see [Models.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/Models.pdf) for details. By default the system is incompressible.
  
  - `-D`: Specifying the use of discrete cosine transforms between the real and reciprocal space; see [Qiang and Li, Macromolecules 53, 9943 (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.0c01974) for
  details. By default the fast Fourier transforms are used.
  
  - `-K K`: Specifying the *K*-value of the REPS-*K* method; see [REPS.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/REPS.pdf) for details. Note that this is used only for the continuous-Gaussian-chain model. By default the REPS-1 method is used; otherwise *K* can be 2, 3 and 4.
    
   Examples:
  
  - **Compilation for the "standard" model:** To compile PSCF+ for calculations of the “standard” model (*i.e.*,
    incompressible melts of continuous Gaussian chains with the Dirac δ-function repulsion) using the REPS-1 method and fast Fourier transforms (same as used in PSCF), simply use the following command:
    
          bash compile.sh
  
  - **Compilation for the DPDC model:** To compile PSCF+ for calculations of the DPDC model (*i.e.*, compressible
    melts of discrete Gaussian chains with the dissipative particle dynamics potential) using fast Fourier
    transforms, use the following command:
    
          bash compile.sh -B DGC -C -N DPD
    
   Finally, to get a list of the aboved options, use the following command:
    
        > bash compile.sh -h

## <a id="3"></a>3. User Guide

### <a id="31"></a>3.1. Invoking an Executable

#### 3.1.1. Calculation of a single (ordered) phase

Here is an example of command-line usage of PSCF+ program for calculation of a single ordered phase:

    pg [-e] -d D -c caseID

In the above, `pg` is the name of executable, `-e` activates echoing of the parameter file to standard output, 
dimensionality `D` of the phase (which can be 1, 2 and 3) is passed to the program as argument of the `-d` command-line option (such implementation is due to Prof. David Morse), and the calculation case ID is specified via `caseID` as argument of the `-c`.

Single-phase SCF calculation requires two input files:

- a parameter file: `caseID.prm`
- a command file: `caseID.cmd`

under the working directory. For example, If one specifies the `caseID` to be `LAM`, then the parameter and command files have to be named as `LAM.prm` and `LAM.cmd`, respectively.

When the program is executed, the parameter file is read first, which is used to initialize the state of the program and
allocate memory. The command file is read and interpreted after the parameter file. The command file is in JSON
format and contains a list of commands that are interpreted and executed in sequence, which controls the program
flow after initialization. The content and format of these files are explained in detail elsewhere (see
[Parameter Files](#ParameterFiles) and [Command Files](#CommandFiles)).

#### 3.1.2. Calculation of the boundary between two phases

Here is an example of command-line usage of PSCF+ to calculate the boundry between two specified phases (where they
have the same Helmholtz free-energy density) using the Ridders' method:

    pg [-e] -d D1,D2 -c caseID

In the above, dimensionalities of the two phases, `D1` and `D2`, are passed to the program as arguments of the `-d`
command-line option; use `0` for dimensionality of the disordered phase.

Two-phase SCF calculation requires three input files:

- two parameter files: `caseID.prm1` and `caseID.prm2`, which can only differ in the lines for `unitCell`, `mesh` 
and `groupName` explained in 3.2 below?.
- a command file: `caseID.cmd`

under the working directory. For example, If one specifies the `caseID` to be `LAM-HEX`, then the two parameter files have to be named as `LAM-HEX.prm1` and `LAM-HEX.prm2`, respectively, for the two phases having dimensionalities D1 and D2, and the command file has to be named as `LAM-HEX.cmd`.

### <a id="32"></a>3.2. Parameter Files

The structure of parameter file is adapted from the C++/CUDA version of PSCF, which contains one System block as
shown below.

    System{
        Mixture{
            nMonomer    ...
            monomers    ...    ...
                        ...    ...
            nPolymer    ...
            Polymer{
                nBlock  ...
                nVertex ...
                blocks  ...    ...    ...    ...
                        ...    ...    ...    ...
                phi     ...
            }
            DPolymer{
                nBond  ...
                nVertex ...
                bonds   ...    ...    ...    ...
                        ...    ...    ...    ...
                phi     ...
            }
            [ns         ...]
        }
        Interaction{
            chiN        ...    ...    ...
                        ...    ...    ...
            [N/kappa    ...]
            sigma       ...
        }
        unitCell        ...    ...    ...
        mesh    ...
        groupName       ...
        AmIterator{
            maxItr      ...
            epsilon     ...
            maxHist     ...
            isMinimized ...
        }
    }

The sub-blocks and required parameters (represented by `...` above) are explained as follows:

- **Mixture:** Description of molecular components (each is considered as a block copolymer in general with each
  block being a linear homopolymer) and composition in the system (which is considered as a mixture in
  general).
  
  - **nMonomer:** Number of monomer (segment) types in the system; this includes solvent.
  
  - **monomers:** Description of each segment type in a seperate line (thus a total of `nMonomer` lines). The
    first parameter in each line is a unique integer index starting from 0 for the segment type, and the
    second parameter specifies its effective bond length (note that only the ratios of these bond lengths matter,
    so one can just set the effective bond length of one segment type to be $b=1$).
  
  - **nPolymer:** Number of molecular components in the system.
  
  - **Polymer** (only used for the continuous-Gaussian-chain model)**:** Description of each molecular component
    in a seperate sub-block (thus a total of `nPolymer` sub-blocks), which includes its chain architecture
    (specified by `nBlock`, `nVertex`, and `blocks` as explained below) and its overall volume fraction `phi` (for canonical-ensemble calculation) or dimensionless chain chemical potential `mu` (for grand-canonical-ensemble calculation) in the system.
    
    - **nBlock:** Number of blocks of this molecular component.
    - **nVertex:** Number of vertices of this molecular component. A vertex is either a joint (where at
      least two blocks meet) or a free chain end.
    - **blocks:** Description of each block in a seperate line (thus a total of `nBlock` lines). The first
      parameter in each line is a unique integer index starting from 0 for the block, the second
      parameter specifies its segment type, the next two parameters are the indices of the two vertices
      it connects, and the last parameter specifies its length (which is just proportional to the block volume fraction).
    - **ensemble:** Choice of ensemble. `closed` is for canonical ensemble and `open` is for grand canonical ensemble.  
  
  - **DPolymer** (only used for discrete-chain models)**:** Description of each molecular component in a
    seperate sub-block (thus a total of `nPolymer` sub-blocks), which includes its chain architecture
    (specified by `nBond`, `nVertex`, and `bonds` as explained below) and its overall volume fraction `phi` (for canonical-ensemble calculation) or dimensionless chain chemical potential `mu` (for grand-canonical-ensemble calculation) in the system.
    
    - **nBond:** Number of v-bonds (including both block bonds and joint bonds; see [Models.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/Models.pdf) for details) of this molecular component.
    - **nVertex:** Number of vertices of this molecular component. A vertex here is either a joint (which is
      connected by at least two v-bonds) or a free chain end (which is connected by only one v-bond).
    - **bonds:** Description of each v-bond in a seperate line (thus a total of `nBond` lines). The first
      parameter in each line is a unique integer index starting from 0 for the v-bond, the second and the
      third parameters are the indices of the two vertices it connects, the next two parameters specify
      the types of these vertices (segments), and the last parameter is its number of segments (0 for a
      joint bond).
  
  - **ns:** Total number of discretization steps along a block contour of length 1. This line is used only for the
    continuous-Gaussian-chain model, and is omitted for discrete-chain models.

- **Interaction:** Description of non-bonded interactions in the system.
  
  - **chiN:** The (generalized) Flory-Huggins <font face="Times New Roman">χ</font> parameter for each pair of different segment types, multiplied by the total number of segments (or chain length) $N$ of a specified molecular component, in a separate line. The first two parameters in each line are the segment-type indices, and the third one is the corresponding value of <font face="Times New Roman">χ</font>$N$. By default, the value between segments of the same type is 0, and thus not needed.
  - **N/kappa:** The value of $N$/<font face="Times New Roman">κ</font>, where <font face="Times New Roman">κ</font> is the generalized Helfand compressibility parameter (see [Models.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/Models.pdf) for details). This line is used only for compressible systems and omitted for incompressible systems.
  - **sigma**: Interaction range of the non-bonded potential in units of $R_{g}\equiv\sqrt{N/6}b$, which is 0 for the Dirac δ-function interaction.

- **unitCell:** The first parameter in this line is the lattice system of the unit cell,  
  which can be `Lamellar`, `Square`, `Rhombic`, `Hexagonal`,  `Rectangular`, `Oblique`, `Monoclinic`, `Orthorhombic`, `Rhombohedral`, `Tetragonal` and `Cubic`, followed by a list of real numbers needed to describe the unit cell.

- **mesh:** Description of the mesh size used for spatial discretization, given by D integer numbers with D being
  the dimensionality of the system.

- **groupName:** Name of the crystallographic space group, which must be the same as the name of the file where the hard-coded symmetry operators are stored; these files are found under the `pscfplus/data/groups/` subdirectory.

- **AmIterator:** Parameters required by Anderson mixing for iteratively solving the SCF equations; see [Matsen,
  Eur. Phys. J. E 53, 361 (2009)](https://link.springer.com/article/10.1140/epje/i2009-10534-3) 
  and [Arora et. al., J. Chem. Phys. 146, 244902 (2017)](https://pubs.aip.org/aip/jcp/article/146/24/244902/992694/Accelerating-self-consistent-field-theory-of-block) for details.
  
  - **maxItr:** Maximum number of iterations.
  - **epsilon:** Convergence criterion for solving the SCF equations; see parameter section of the  [PSCF user manual](https://pscf.readthedocs.io/en/latest/param.html#example) for details.
  - **maxHist:** A positive integer for the maximum size of the history matrix used in Anderson mixing.
  - **isMinimized**: 1 for finding the bulk periodicty of the ordered phase (where the numbers for `unitCell` above are used as the initial guess), and 0 for performing the calculation at fixed `unitCell`.

Below are two examples of the parameter file:

- Example for SCF calculations of the BCC phase formed by the "standard" model of compositionally and conformationally asymmetric diblock copolymer A-B in a canonical ensemble, where the free-energy density will be minimized to find its bulk period.
  
        System{
          Mixture{
            nMonomer                               2
            monomers                               0  A   2.0
                                                   1  B   1.0
            nPolymer                               1
            Polymer{
              nBlock                                 2
              nVertex                                3
              blocks                                 0  0  0  1    0.25
                                                     1  0  1  2    0.75
              ensemble                          Closed
              phi                   1.0
            }
            sigma                   0.0
            ns                      256
          }
          Interaction
             chiN  1   0   24.0
          }
          unitCell cubic     7.4477956126
          mesh      32   32   32
          groupName         I_m_-3_m
          AmIterator{
           maxItr 1000
           epsilon 1e-6
           maxHist 20
           isFlexible 1
          }
        }

- Example for SCF calculations of the σ phase formed by the DPDC model of conformationally
  symmetric diblock copolymer binary blend A<sub>1</sub>-B<sub>1</sub>/A<sub>2</sub>-B<sub>2</sub> in a grand-canonical ensemble.
  
         System{
          Mixture{
            nMonomer                               2
            monomers                               0  A   1.0
                                                   1  B   1.0
            nPolymer                               2
            DPolymer{
              nBond                                  3
              nVertex                                4
              bonds                                  0  0  1  0  0  3
                                                     1  2  3  1  1  7
                                                     2  1  2  0  1  0
              ensemble                          Open
              mu                    8.0
            }
            DPolymer{
              nBond                                  3
              nVertex                                4
              bonds                                  0  0  1  0  0  5
                                                     1  2  3  1  1  5
                                                     2  1  2  0  1  0
              ensemble                          Open
              mu                    1.0
            }
            sigma                   0.8944271909999159
            N/kappa                 157.0796326794896619
          }
          Interaction
             chiN  1   0   24.0
          }
          unitCell tetragonal 2.8e+01   1.5e+01
          mesh      32   32   32
          groupName         I_m_-3_m
          AmIterator{
           maxItr 1000
           epsilon 1e-6
           maxHist 20
           isFlexible 1
          }
        }

### <a id="33"></a>3.3. Command Files

The command file contains a sequence of commands that are read and executed in serial. The commands are
organized into a JSON file ([here](https://www.json.org/json-en.html) gives an introduction to JSON). Below is an example of a command file for single-phase calculation.

    [
        {
            "FieldIO":
            {
                "IO": "read",
                "Type": "omega",
                "Format": "basis",
                "Directory": "out/omega/"
            }
        },
        {
            "SinglePhaseSCF":
            {
                "OutputDirectory": "out/"
            }
        },
        {
            "FieldIO":
            {
                "IO": "write",
                "Type": "omega",
                "Format": "basis",
                "Directory": "out/omega/"
            }
        },
        {
            "FieldIO":
            {
                "IO": "write",
                "Type": "omega",
                "Format": "real",
                "Directory": "out/omega/"
            }
        },
        {
            "FieldIO":
            {
                "IO": "write",
                "Type": "phi",
                "Format": "real",
                "Directory": "out/phi/"
            }
        }
    ]

All commands are put in a pair of square brackets, and they are divided into different blocks. The following explains the usage of each command block.

- To read or write a (volume-fraction or conjugate) field file in a specified format, use `"FieldIO"`
  block.
  
        {
            "FieldIO":
            {
                "IO": "read",
                "Type": "omega",
                "Format": "basis",
                "Directory": "in/"
            }
        }
  
    `"IO"` can be either `"read"` or `"write"`, for reading from or writing to a file, respectively. `"Type"` specifies the
  field, which can be either `"omega"` for conjugate field or `"phi"` for volume-fraction field. `"Format"` specifies
  the format of the field, which can be either `"basis"` for the basis format, `"real"` for the real-space-grid
  format, or `"reciprocal"` for the reciprocal-space-grid format; see 
  [PSCF documentation](https://dmorse.github.io/pscfpp-man/) for the explanation
  of these formats. `"Directory"` specifies the directory of the file; use `./` for the same directory as the command file.
  Finally, the filename is specified by the case ID, field type and format as `caseId_type.format`; for example, to read a conjugate field in the basis format as input of your SCF calculation with the case ID `BCC`, the filename must be
  `BCC_omega.basis`.

- To perform SCF calculation of a single phase with given initial guess (the conjugate field of which should be
  read before the calculation), use the `"SinglePhaseSCF"` block.
  
        {
            "SinglePhaseSCF":
            {
                "OutputDirectory": "out/"
            }
        }
  
    `"OutputDirectory"` specifies the directory of the output file for the system free energy and its
  components; again, use `./` for the same directory as the command file. This output filename is `caseId_fe.json`; for example, with the case ID `sigma`, the name of the output file is `sigma_fe.json`.

- If the initial guess is given in terms of the volume-fraction fields (*i.e.*, the morphology of a phase), 
  one needs to convert it into the conjugate fields using the "PhiToWBasis" block. 
  
        {
            "PhiToWBasis":
            {
                "InputDirectory": "in/phi/"
                "OutputDirectory": "out/omega/"
            }
      
        }
  
    `"InputDirectory"` specifies the directory of the input file of the volume-fraction fields, which is in the format of `"real"`; the filename has to be `caseID_phi.real`. `"OutputDirectory"` specifies the directory of the output file of the conjugate fields, which is in the format of `"basis"`; the filename has to be `caseID_omega.basis`. The conjugate fields are obtained via the SCF equations excluding the Lagrange multiplier contribution.

- To perform automated calculation along a path (ACAP; see [ACAP.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/ACAP.pdf) for details), use the block `"ACAP"`.
  
        [
            {
                "FieldIO": {
                    "IO" : "read",
                    "Type": "omega",
                    "Format": "basis",
                    "Directory": "in/"
                }
            },
            {
                "ACAP":{
                    "Variable": ["chiN", 0, 1],      
                    "InitialValue": 16,
                    "FinalValue": 15.5,
                    "InitialStep": 0.1,
                    "SmallestStep": 0.001,
                    "LargestStep": 0.5,
                    "StepScale": 1.1,  
                    "OutputDirectory": "out/",
                    "IntermediateOuput":
                    [
                        {
                            "OutputPoints": [15.4, 15.6, 15.8]
                        },
                        {
                            "Field" : "omega",
                            "Format": "basis", 
                            "OutputDirectory": "out/omega/"
                        },
                        {
                            "Field" : "phi",
                            "Format": "real", 
                            "OutputDirectory": "out/phi/"
                        }
                    ]
                }
            },
            {
                "FieldIO": {
                    "IO" : "write",
                    "Type": "omega",
                    "Format": "basis",
                    "Directory": "out/omega/"
                }
            },
            {
                "FieldIO": {
                    "IO" : "write",
                    "Type": "phi",
                    "Format": "real",
                    "Directory": "out/phi/"
                }
            }
        ]

    `"Variable"` specifies the paramter whose value is varied along the path; this so far can be either `"chiN"` as 
    in the above example (see the `Interaction` sub-block in [Sec. 3.2](#ParameterFiles)), or `"b"`, the effective bond length of a segment type (see the `Mixture` sub-block in [Sec. 3.2](#ParameterFiles)). If the varying parameter is `"chiN"`, user needs to specify the two segment types as shown in the above example. If it is `"b"`, user needs to specify the corresponding segment type (*e.g.*, `["b", 0]`), which cannot be that having $b=1$. `"InitialValue"` and `"FinalValue"` give the starting and ending parameter values of the path, respectively. `"InitialStep"`, `"SmallestStep"`, and `"LargestStep"` specifies the initial, smallest and largest absolute values of the stepsize, respectively, used for varing the parameter along the path. `"StepScale"` specifies the scaling factor used to vary the stepsize. `"OutputDirectory"` specifies the directory of the output file for the system free energy and its components along the path. `"IntermediateOuput"` is needed when user wants to output field files during `ACAP`. The first block in `"IntermediateOuput"` specifies the parameter values at which the fields are output along the path (the order of these values does not matter, which means `[1.1, 1.2, 1.3]` and `[1.2, 1.3, 1.1]` result in the same intermediate output files). Each of the following blocks specifies the type of the field, its format, and the directory of the output files via `"Field"`, `"Format"`, and `"IntermediateDirectory"`, respectively.

    In addition, users can change all <font face="Times New Roman">χ</font>$N$-values for each step using `"chiN_all"` for `"Variable"` as in the following example; note that all <font face="Times New Roman">χ</font>$N$-values are varied with the same initial and final points and adaptive steps in this case. 
  
        {
            "ACAP":{
                "Variable": ["chiN_all"],      
                "InitialValue": 16,
                "FinalValue": 15.5,
                "InitialStep": 0.1,
                "SmallestStep": 0.001,
                "LargestStep": 0.5,
                "StepScale": 1.1,  
                "OutputDirectory": "out/",
                "IntermediateOuput":
                [
                    {
                        "OutputPoints": [15.4, 15.6, 15.8]
                    },
                    {
                        "Field" : "omega",
                        "Format": "basis", 
                        "OutputDirectory": "out/omega/"
                    },
                    {
                        "Field" : "phi",
                        "Format": "real", 
                        "OutputDirectory": "out/phi/"
                    }
                ]
            }
        }

- To find a boundary point between two specified phases, where they have the same Helmholtz free-energy density, use the "PhaseBoundaryPoints" block as in the following example:
  
        [
            {
                "FieldIO": {
                    "PhaseId": 1,
                    "IO" : "read",
                    "Type": "omega",
                    "Format": "basis",
                    "Directory": "in/1/"
                }
            },
            {
                "FieldIO": {
                    "PhaseId": 2,
                    "IO" : "read",
                    "Type": "omega",
                    "Format": "basis",
                    "Directory": "in/2/"
                }
            },
            {
                "PhaseBoundaryPoints": {
                    "epsilon": 1e-5,
                    "b": [1, 1.0],
                    "InitialGuess(chiN)": [0, 1, 19.1, 19.3]
                }
            }
            {
                "FieldIO": {
                    "PhaseId": 1,
                    "IO" : "write",
                    "Type": "omega",
                    "Format": "basis",
                    "Directory": "out/1/omega/"
                }
            },
            {
                "FieldIO": {
                    "PhaseId": 2,
                    "IO" : "write",
                    "Type": "phi",
                    "Format": "real",
                    "Directory": "out/2/phi/"
                }
            }
        ]

    Here, the initial guess of each phase is read first by the two `"FieldIO"` blocks; different from the above single-phase calculation, `"PhaseId"` is needed in each `"FieldIO"` block, which takes the value of 1 or 2 in accordance to the command-line arguments of `-d`, `D1` and `D2`, respectively (see [Invoking an Executable](#InvokinganExecutable)). In the `"PhaseBoundaryPoints"` block, `"epsilon"` specifies the convergence criterion, which is the absolute difference in the dimensionless Helmholtz free-energy density between the two phases; the next line specifies that the calculation is performed at the constant value for the effective bond length (*i.e.*, `"b"`) of segment type `1`, which is 1.0; in this case, the calculation solves for the <font face="Times New Roman">χ</font>$N$-value between segment types `0` and `1`, which falls in the interval of `[19.1, 19.3]` as shown in third line. Note that this interval is required by the Ridders' method used for the phase-boundary calculation.

### <a id="34"></a>3.4. Free-Energy Density and Its Components

The system parameters, (mean-field) free-energy density and its components (see [FE_Decomp.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/FE_Decomp.pdf) for details) are written into a 
JSON file named `caseId_fe.json`. Below gives an example of the JSON file for lamellae formed by the "standard" model of linear and symmetric diblock copolymers and a detailed explanation of each component in the file, 
including its data structure and meaning:


    [
    	{
    		"ChemicalPotential" : 
    		[
    			[
    				0,
    				5.4414520382370526
    			]
    		],
    		"Chi" : 
    		[
    			[
    				0,
    				1,
    				15
    			]
    		],
    		"HelmholtzFreeEnergy" : 
    		{
    			"EntropyContribution" : 
    			{
    				"BlockComponent" : 
    				[
    					[
    						0,
    						0,
    						0.60431478570038388
    					],
    					[
    						0,
    						1,
    						0.60431478569871278
    					]
    				],
    				"Total" : 1.4171996827232327,
    				"VertexComponent" : 
    				[
    					[
    						0,
    						0,
    						0.089918704246324227
    					],
    					[
    						0,
    						1,
    						0.028732702831544728
    					],
    					[
    						0,
    						2,
    						0.089918704246278097
    					]
    				]
    			},
    			"InternalEnergyContribution" : 
    			{
    				"Compressibility" : 0,
    				"FloryHugginsRepulsion" : 
    				{
    					"Components" : 
    					[
    						[
    							0,
    							1,
    							0,
    							0,
    							2.0121261777529433
    						]
    					],
    					"Total" : 2.0121261777529433
    				},
    				"Total" : 2.0121261777529433
    			},
    			"Total" : 3.429325860476176
    		},
    		"Phi" : 
    		[
    			[
    				0,
    				1
    			]
    		],
    		"SegmentLength" : 
    		[
    			[
    				0,
    				1
    			],
    			[
    				1,
    				1
    			]
    		],
    		"Unitcell" : 
    		[
    			[
    				"P_-1",
    				3.7136946098077317
    			]
    		]
    	}
    ]

- Top-level structure

      [
        {...},
          ...
        {...}
      ]

  The top level is an array (`[]`) containing a series of objects (`{}`), where 
  each object contains the Helmholtz free-energy (in a canonical ensemble) or the grand potential (in a grand-canonical ensemble) per chain of length $N$ and its components. For 
  SCF calculation of a single phase, the array contains only one objet; for ACAP,
  the number of objects depends on the number of steps along the path.

- Object components

  Each oject on the top-level contains the following eight components:

  `"SegmentLength" : [ ..., [m_k,b_k], ... ]` contains the effective bond length 
  of each segment type in a separate array, where `m_k` denotes the index of segment type and 
  `b_k` denotes the corresponding effective bond length.

  `"ChiN" : [ ..., [m_A, m_B, chiN_AB], ... ]`  contains the (generalized) Flory-Huggins <font face="Times New Roman">χ</font> 
  parameter for each pair of different segment types, multiplied by $N$, in a separate array. The first two parameters in each line are the segment-type indices, and the third one is 
  the corresponding value of <font face="Times New Roman">χ</font>$N$. 
 
  `"N/kappa" :  value` contains the value of $N$/<font face="Times New Roman">κ</font>, where <font face="Times New Roman">κ</font> 
  is the generalized Helfand compressibility parameter, This component is used only for compressible systems and omitted for incompressible systems.

  `"sigma" :  value` contains the value of the interaction range of the non-bonded potential in units of $R_{g}\equiv\sqrt{N/6}b$, which is 0 for the Dirac δ-function interaction.

  `"Unitcell" : ["SpaceGroup", [L_x, L_y, L_z]]` contains the unit-cell parameter, where the first element is the space group and the second element is an array of 
  real numbers describing the unit cell.

  `"Phi" : [ ..., [chain_k, phi_k], ... ]` contains the specified (in a canonical ensemble) or calculated (in a grand-canonical ensemble) overall volume fraction of each chain type, where `chain_k` is the index of chain type and `phi_k` is 
  the corresponding overall volume fraction.

  `"ChemicalPotential" : [ ..., [chain_k, mu_k], ... ]` contains the calculated (in a caonical ensemble) or specified (in a grand-canonical ensemble) chemical potential of each chain type, where `chain_k` is the index of chain type and `mu_k` is 
  the corresponding chain chemical potential.

  `"FreeEnergyDensity"` contains three components: `"InternalEnergyContribution"` , 
  `"EntropyContribution"`, and `"Total"`. 

  - The  `"InternalEnergyContribution"` contains the internal-energy contribution (`"Total"`) and its components (`"FloryHugginsRepulsion"`, 
  `"Compressibility"`).

        "InternalEnergyContribution" : 
			  {
			  	"Compressibility" : value,
			  	"FloryHugginsRepulsion" : 
			  	{
			  		"Components" : 
			  		[     ...,
			  			[
			  				chain_c1,
			  				block_b1,
			  				chain_c2,
			  				block_b2,
			  				value
			  			], ...
			  		],
			  		"Total" : value
			  	},
			  	"Total" : value
			  },
      The value of `"Compressibility"` is the internal-energy contribution due to compressibility. The components
      of `"FloryHugginsRepulsion"` include the internal-energy contributions due to the Flory-Huggins-type repulsion between 
      each pair of blocks having different segment types (`"Components"`) and their total (`"Total"`). 



  - The  `"EntropyContribution"` contains the entropy contribution (`"Total"`) and its components (`"VertexComponent"`, 
  `"BlockComponent"`).

            "EntropyContribution" : 
	      		{
	      			"BlockComponent" : 
	      			[     ...
	      				[
	      					c,
	      					b,
	      					value
	      				], ...
	      			],
	      			"Total" : 1.4171996827232327,
	      			"VertexComponent" : 
	      			[     ...
	      				[
	      					c,
	      					v,
	      					value
	      				], ...
	      			]
	      		}
      The elemets of `"BlockComponent"` are the entropy contributions of each block, where `c` is the chain-type index, `b` is the block index, 
      and `value` is the corresponding entropy contribution. The elements of `"VertexComponent"` are the the entropy contributions of each vertex,
      where `c` is the chain-type index and `v` is the vertex index.

An advantage of using a JSON file to store the data of free-energy density and its components is that the data can be accessed like elements of an array, which
can make the data analysis quite convenient. For example, one can write the following Python script to read a sequence of JSON files and extract the internal-energy 
contribution to the free-energy density due to the Flory-Huggins-type repulsion for varying <font face="Times New Roman">χ</font>$N$:

```     python 
import json
import os
from pathlib import Path

# Configure paths (update these)
JSON_DIR = "path/to/json_files"  # Directory with your 100 JSON files
OUTPUT_DIR = "extracted_data"    # Where to save results

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each JSON file
for filename in os.listdir(JSON_DIR):
    if filename.endswith('.json'):
        file_path = os.path.join(JSON_DIR, filename)
        
        try:
            # Load JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract first simulation object
            sim = data[0]
            
            # 1. Extract Chi parameters
            chi_data = []
            for entry in sim["Chi"]:
                chi_data.append({
                    "m1": entry[0],
                    "m2": entry[1],
                    "chiN": entry[2]
                })
            
            # 2. Extract VertexComponent
            vertex_data = []
            for entry in sim["HelmholtzFreeEnergy"]["InternalEnergyContribution"]["FloryHugginsRepulsion"]["Total"]:
                vertex_data.append({
                    "FloryHugginsRepulsion": entry[0]
                })
      
```
