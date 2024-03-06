#ifndef PSPG_GPU_TYPES_H
#define PSPG_GPU_TYPES_H

#include <cufft.h>
#include <stdio.h>
#include <iostream>

namespace Pscf {
namespace Pspg {

typedef cufftDoubleReal cudaReal;
typedef cufftDoubleComplex cudaComplex;
typedef cufftDoubleReal hostReal;
typedef cufftDoubleComplex hostComplex;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if(abort) exit(code);
    }
}

}
}
#endif