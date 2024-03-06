# PSCF+ -- Polymer Self-Consistent Field Theory (C++/CUDA) with Improved and Extended Capabilities
PSCF+ is a software package for solving the polymer self-consistent field (SCF) theory in continuum. It is based on the nice GPU framework of [PSCF](https://github.com/dmorse/pscfpp) (which is only for the "standard" or known as the Edwards-Helfand model, *i.e.*, incompressible melts of continuous Gaussian chains with the Dirac $\delta$-function interactions, commonly used in polymer field theories) originally developed by Prof. David Morse and co-workers, but is improved with better numerical methods, less GPU memory usage and more flexible algorithms, and is extended to various discrete chain models. Similar to the C++/CUDA version of PSCF, PSCF+ described here is written primarily in C++, with GPU accelerated code in CUDA.

PSCF+ is free, open-source software. It is distributed under the terms of the GNU General Public License (GPL) as published by the Free Software Foundation, either version 3 of the License or (at your option) any later version. PSCF+ is distributed without any warranty, without even the implied warranty of merchantability or fitness for a particular purpose. See the LICENSE file or the [gnu web page](https://www.gnu.org/licenses/) for details.


## Overview
Similar to the C++/CUDA version of PSCF, PSCF+ is applicable to mixtures containing arbitrary acyclic copolymers, and preserves all of the nice features already implemented in the former, including the use of [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) on GPUs, the use of Anderson mixing (which is performed on GPUs) combined with a variable-cell scheme to simultaneously solve the nonlinear SCF equations and find the bulk periodicity for the ordered phases formed by block copolymer self-assembly (which speeds-up the calculation by an order of magnitude), and the extensive documentation produced by [Doxygen](https://www.doxygen.nl/). Their differences and expected advantages of the latter include:
- PSCF is only applicable to the "standard" model, while PSCF+ can also be applied to various discrete chain models (with finite-range non-bonded interactions) as commonly used in molecular simulations, thus providing the mean-field reference results for such simulations; see [Models.pdf](https://github.com/qwcsu/PSCFplus/blob/master/doc/notes/Models.pdf) for more details.
- For the "standard" model, PSCF+ uses the Richardson-extrapolated pseudo-spectral methods (denoted by REPS-*K* with *K*=0,1,2,3,4) to solve the modified diffusion equations (which is the crux of SCF calculations), while PSCF uses only REPS-1. A larger *K*-value gives more accurate result at the cost of larger computational cost; see [REPS.pdf](https://github.com/qwcsu/PSCFplus/blob/master/doc/notes/REPS.pdf) for more details.
- For 3D spatially periodic ordered phases such as those formed by block copolymer self-assembly, while PSCF uses fast Fourier transforms (FFTs) between a uniform grid in the real space and that in the reciprocal space, for *Pmmm* supergroup PSCF+ uses the discrete cosine transform of type II instead of the FFT to take advantage of the (partial) symmetry of an ordered phase to reduce the number of grid points, thus both speeding up the calculations and reducing the memory usage; see [*Y. Qiang and W. Li*, **Macromolecules 53**, 9943 (2020)](https://pubs.acs.org/doi/10.1021/acs.macromol.0c01974) for more details.
- In SCF calculations the (one-end-integrated) forward and backward propagators $q$ and $q^{\dagger}$ of each block usually take the largest memory usage, but the GPU memory is often limited. While in PSCF the size of these propagators is $MN_s$, where $M$ denoting the number of grid points in real space and $N_s$ the number of contour-discretization points on a continuous chain (or the number of segments on a discrete chain), in PSCF+ the "slice" algorithm proposed by Li and Qiang can be used to reduce the size of $q$ to $M\sqrt{N_s}$ and that of $q^{\dagger}$ to just $M$, thus greatly reducing the GPU memory usage, at the cost of computing $q$ twice; see [SavMem.pdf](https://github.com/qwcsu/PSCFplus/blob/master/doc/notes/SavMem.pdf) for more details.
- Since SCF equations are highly nonlinear, having a good initial guess is very important in practice as it determines not only which final solution (corresponding to a phase in block copolymer self-assembly) can be obtained but also how many iteration steps the solver (*e.g.*, the Anderson mixing) takes to converge these equations. PSCF+ uses automated calculation along a path (ACAP), where the converged solution at a neighboring point is taken as the initial guess at the current point in the parameter space. While this is similar to the "SWEEP" command in PSCF, the key for ACAP to be successful and efficient is that it automatically adjusts the step size along the path connecting the two points, instead of using a fixed step size as in the "SWEEP" command. In PSCF+, ACAP is further combined with the phase-boundary calculation between two specified phases, making the construction of phase diagrams in 2D very efficient. See [ACAP.pdf](https://github.com/qwcsu/PSCFplus/blob/master/doc/notes/ACAP.pdf) for more details.
- The approach used by PSCF to solve the SCF equations (for an incompressible system) does not allow any athermal species in the system, which has no non-bonded interactions with all other species. This problem is solved in PSCF+; see [SlvSCF.pdf](https://github.com/qwcsu/PSCFplus/blob/master/doc/notes/SlvSCF.pdf) for more details. 

## Getting the source code
The PSCF+ source code is maintained in the GitHub repository [https://github.com/qwcsu/pscfplus](https://github.com/qwcsu/pscfplus).
The source code may be obtained by using a git version-control-system client to clone the repository. To do so on a machine with a git client, use the command:
````
git clone --recursive https://github.com/qwcsu/pscfplus.git
````
Note that using the ```--recursive``` option is necessary to clone several git submodules that are maintained in separate repositories. This command will create a new directory called ```pscf+/``` that contains all the source code and associated documentation, including all the required git submodules.

## Dependencies
Same as PSCF, the PSCF+ source code is written using C++ as the primary language, with CUDA used for GPU acceleration. PSCF+ is only provided in source file format; all programs must be compiled from source. This package was developed on LINUX operating systems using standard UNIX utilities.
The PSCF+ package depends on the [GNU scientific library](https://www.gnu.org/software/gsl/) (GSL), and can only run on a computer with an appropriate NVIDIA GPU. To compile or run these programs, the system must also have an NVIDIA CUDA development environment that provides the [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) fast Fourier transform library.

## Compiling
Short instructions for compiling, after cloning the git repository and installing all of the required dependencies, are given below:
- Add the ```pscf+/bin``` directory to your Linux command search ```$PATH``` environment variable.
- Add the ```pscf+/lib/python``` directory to your ```$PYTHONPATH``` environment variable.
- Change directory (```cd```) to the ```pscf+/``` directory.
- Run the ```pscf+/setup``` script by entering ```./setup```. 
To compile PSCF+ with chosen chain-connectivity model, non-bonded pair potential, system compressibility, REPS-*K* method, and type of discrete transform, users just need to change directory to ```pscf+/bld``` and use the following command:
  ````
  bash compile.sh [-B CHN] [-N NBP] [-C] [-D] [-K K]  
  ````
  to invoke the compilation script, which installs executable programs in the ```pscf+/bin``` directory. The options in the above command are as follows:
  
  ```CHN``` – Specifying the model of chain connectivity (by default it is the continuous Gaussian chain)  
    - ```DGC```: discrete Gaussian chain
    - ```FJC```: freely jointed chain
    
  ```NBP``` – Specifying the form of non-bonded pair potential (by default it is the Dirac $\delta$-function potential)
    - ```G```: Gaussian potential
    - ```DPD```: dissipative particle dynamics potential
    - ```SS```: soft-sphere potential 
    
  ```-C``` – Specifying a compressible system (by default the system is incompressible)
  
  ```-D``` – Specifying the use of the discrete cosine transform of type II between the real and reciprocal space (by default the fast Fourier transform is used)
  
  ```K``` – Specifying the K-value of the REPS-*K* method (by default the REPS-1 method is used)
  
  For example, to compile the “standard” model (*i.e.*, incompressible melts of continuous Gaussian chains with the Dirac $\delta$-function repulsion) using REPS-1 method and fast Fourier transform (same as used in PSCF), users can simply use the following command:
  
  ````
  bash compile.sh
  ````
  
  To compile the DPDC model (*i.e.*, compressible melts of discrete Gaussian chains with the dissipative particle dynamics potential) using fast Fourier transform, users can use the following command:
  
  ````
  bash compile.sh -B DGC -C -N DPD
  ````
  
  To get the list of the aboved options, users can use

  ````
  bash compile.sh -h
  ````
  
## Programs and command line usage
Similar to PSCF, ```pgDd``` are the executable programs for ```D``` = ```1```, ```2```, or ```3``` dimensional periodic structures for the chosen model system. Each of these programs reads a parameter file and a command file. The parameter file is a fixed-format file that contains parameters required to initialize the program. Note that the parameter files of different model systems are slightly different. The command file is a more flexible script containing a sequence of commands that are read and executed sequentially to specify a sequence of computational steps. The command file usually specifies the name of a file that contains an initial guess for the conjugate fields and names of files to which the final conjugate fields and volume-fraction fields should be written.  
The usual command line syntax for executing any pscf+ program is:
````
pgDd [-e] -p param -c command 
````
where the ```-e``` command line option causes the program to echo the parameter file to the standard output as it is read, ```param``` denotes the path to a parameter file, and  ```command``` denotes the path to a command file. For example, one might enter

````
pg3d -p param -c command
``````
to execute the program for three dimensional periodic structures with given command and param file.

## Examples
The directory ```pscf+/examples``` contains a set of examples of simple calculations of various models. Each example directory contains a parameter file (named ```param```), a command file (named ```command```), and an input conjugate-field file. 
Top level subdirectories of ```pscf+/examples``` contain examples for different model systems. Subdirectory ```examples/STD_4_DCT``` contains examples for the “standard” model obtained using REPS-4 method and DCT. Subdirectory ```examples/DPDC``` contains examples for the DPDC model. Subdirectory ```examples/FJC_G_INC``` contains examples for incompressible melts of freely jointed chains with the Gaussian potential. 


## Contributors
- Qiang (David) Wang
- Juntong He


