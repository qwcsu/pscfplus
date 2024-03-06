#ifndef PSPG_LINEAR_ALGEBRA_H
#define PSPG_LINEAR_ALGEBRA_H

#include "GpuTypes.h"

namespace Pscf {
namespace Pspg {

/** \ingroup Pspg_Math_Module 
* @{
*/


__global__ void subtractUniform(cudaReal* result, cudaReal rhs, int size);

__global__ void addUniform(cudaReal* result, cudaReal rhs, int size);

__global__ void pointWiseSubtract(cudaReal* result, const cudaReal* rhs, int size);

__global__ void pointWiseSubtractFloat(cudaReal* result, const float rhs, int size);

__global__ void pointWiseBinarySubtract(const cudaReal* a, const cudaReal* b, cudaReal* result, int size);

__global__ void pointWiseAdd(cudaReal* result, const cudaReal* rhs, int size);

__global__ void pointWiseBinaryAdd(const cudaReal* a, const cudaReal* b, cudaReal* result, int size);

__global__ void pointWiseAddScale(cudaReal* result, const cudaReal* rhs, double scale, int size);

__global__ void pointWiseAddScale2(cudaReal* result, const cudaReal* rhs, const cudaReal* rhs2, double scale, int size);

__global__ void inPlacePointwiseMul(cudaReal* a, const cudaReal* b, int size);

__global__ void pointwiseMul(const cudaReal* a, const cudaReal* b, cudaReal* result, int size);

__global__ void assignUniformReal(cudaReal* result, cudaReal uniform, int size);

__global__ void assignReal(cudaReal* result, const cudaReal* rhs, int size);

__global__ void assignExp(cudaReal* exp, const cudaReal* w, double constant, int size);

__global__ void scaleReal(cudaReal* result, double scale, int size);

__global__ void cudaComplexMulAdd(cudaReal* result, 
                                  const cudaComplex* c1, 
                                  const cudaComplex* c2,
                                  const cudaReal scale,
                                  int size);

__global__ 
void cudaComplexAdd(cudaComplex* result, 
                    const cudaComplex* c1,
                    int size);

__global__ void mStressHelperIncmp(cudaReal* result, 
                                   const cudaReal* c,
                                   const cudaReal* dksq,
                                   const cudaReal* dbu0K,
                                   int paramN,
                                   int kSize,
                                   int rSize); 

__global__ 
void mStressHelper(cudaReal* result, 
                   const cudaReal* c,
                   const cudaComplex* k1, 
                   const cudaReal* dksq,
                   const cudaReal* dbu0,
                   cudaReal kappaN,
                   int paramN,
                   int kSize,
                   int rSize);  

__global__ 
void sVHelper(cudaReal *result,
              const cudaReal *rhoJ,
              int nx);

__global__ 
void sBlockHelper(cudaReal *result,
                  const cudaReal *rhoJ,
                  const cudaReal *q,
                  int nx);

__global__ void sSHelper(cudaReal *result,
                         const cudaReal *rhoS,
                         const cudaReal *w,
                         const cudaReal Q,
                         int nx);

/** @} */

}
}
#endif
