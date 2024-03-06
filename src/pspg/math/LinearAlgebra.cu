#ifndef PSPG_LINEAR_ALGEBRA_CU
#define PSPG_LINEAR_ALGEBRA_CU

#include "LinearAlgebra.h"

namespace Pscf
{
   namespace Pspg
   {

      __global__ void subtractUniform(cudaReal *result, cudaReal rhs, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] -= rhs;
         }
      }

      __global__ void addUniform(cudaReal *result, cudaReal rhs, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] += rhs;
         }
      }

      __global__ void pointWiseSubtract(cudaReal *result, const cudaReal *rhs, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] -= rhs[i];
         }
      }

      __global__ void pointWiseSubtractFloat(cudaReal *result, const float rhs, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] -= rhs;
         }
      }

      __global__ void pointWiseBinarySubtract(const cudaReal *a, const cudaReal *b, cudaReal *result, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] = a[i] - b[i];
         }
      }

      __global__ void pointWiseAdd(cudaReal *result, const cudaReal *rhs, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] += rhs[i];
         }
      }

      __global__ void pointWiseBinaryAdd(const cudaReal *a, const cudaReal *b, cudaReal *result, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] = a[i] + b[i];
         }
      }

      __global__ void pointWiseAddScale(cudaReal *result, const cudaReal *rhs, double scale, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] += scale * rhs[i];
         }
      }

      __global__ void pointWiseAddScale2(cudaReal *result, const cudaReal *rhs, const cudaReal *rhs2, double scale, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] += scale * rhs[i] * rhs2[i];
         }
      }

      __global__ void inPlacePointwiseMul(cudaReal *a, const cudaReal *b, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            a[i] *= b[i];
         }
      }

      __global__ void pointwiseMul(const cudaReal *a, const cudaReal *b, cudaReal *result, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] = a[i] * b[i];
         }
      }

      __global__ void assignUniformReal(cudaReal *result, cudaReal uniform, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] = uniform;
         }
      }

      __global__ void assignReal(cudaReal *result, const cudaReal *rhs, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] = rhs[i];
         }
      }

      __global__ void assignExp(cudaReal *out, const cudaReal *w, double constant, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            out[i] = exp(-w[i] * constant);
         }
      }

      __global__ void scaleReal(cudaReal *result, double scale, int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;

         for (int i = startID; i < size; i += nThreads)
         {
            result[i] *= scale;
         }
      }

      __global__ void cudaComplexMulAdd(cudaReal *result,
                                        const cudaComplex *c1,
                                        const cudaComplex *c2,
                                        const cudaReal scale,
                                        int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            result[i] += scale * cuCmul(c1[i], cuConj(c2[i])).x;
         }
      }

      __global__ void cudaComplexAdd(cudaComplex *result,
                                     const cudaComplex *c1,
                                     int size)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < size; i += nThreads)
         {
            if (i == 0)
            {
               result[i].x += (c1[i].x - 1.0);
               result[i].y += c1[i].y;
            }
            else
            {
               result[i].x += c1[i].x;
               result[i].y += c1[i].y;
            }
         }
      }

      __global__ void mStressHelperIncmp(cudaReal *result,
                                         const cudaReal *c,
                                         const cudaReal *dksq,
                                         const cudaReal *dbu0K,
                                         int paramN,
                                         int kSize,
                                         int rSize)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < kSize; i += nThreads)
         {

            result[i] = dbu0K[i] * dksq[paramN * rSize + i] * c[i];
         }
      }

      __global__ void mStressHelper(cudaReal *result,
                                    const cudaReal *c,
                                    const cudaComplex *k1,
                                    const cudaReal *dksq,
                                    const cudaReal *dbu0,
                                    cudaReal kappaN,
                                    int paramN,
                                    int kSize,
                                    int rSize)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < kSize; i += nThreads)
         {

            result[i] = dbu0[i] * dksq[paramN * rSize + i] * (kappaN * (k1[i].x * k1[i].x - k1[i].y * k1[i].y) + c[i]);
         }
      }

      __global__ void sVHelper(cudaReal *result,
                               const cudaReal *rhoJ,
                               int nx)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < nx; i += nThreads)
         {
            result[i] = rhoJ[i]*log(abs(rhoJ[i]));
         }
      }

      __global__ void sSHelper(cudaReal *result,
                               const cudaReal *rhoS,
                               const cudaReal *w,
                               const cudaReal Q,
                               int nx)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < nx; i += nThreads)
         {
            result[i] = rhoS[i]*log(Q)+rhoS[i]*w[i];
         }
      }

      __global__ void sBlockHelper(cudaReal *result,
                                   const cudaReal *rhoJ,
                                   const cudaReal *q,
                                   int nx)
      {
         int nThreads = blockDim.x * gridDim.x;
         int startID = blockIdx.x * blockDim.x + threadIdx.x;
         for (int i = startID; i < nx; i += nThreads)
         {
            result[i] = rhoJ[i]*log(abs(q[i]));
         }
      }

   }
}
#endif
