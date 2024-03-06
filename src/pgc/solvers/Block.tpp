#ifndef PGC_BLOCK_TPP
#define PGC_BLOCK_TPP
// #define double float

/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include "Block.h"
#include <pspg/math/GpuHeaders.h>
#include <pscf/mesh/Mesh.h>
#include <pscf/mesh/MeshIterator.h>
#include <pscf/crystal/shiftToMinimum.h>
#include <util/containers/FMatrix.h> // member template
#include <util/containers/DArray.h>  // member template
#include <util/containers/FArray.h>  // member template
#include <ctime>

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {
            static __global__ void mulDelKsq(cudaReal *result, const cudaComplex *q1,
                                             const cudaComplex *q2, const cudaReal *delKsq,
                                             int paramN, int kSize, int rSize)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < kSize; i += nThreads)
                {

                    result[i] = cuCmul(q1[i], cuConj(q2[i])).x * delKsq[paramN * rSize + i];
                }
            }

            static __global__ void mulKsqFCT(cudaReal *result,
                                             cudaReal *qkSum,
                                             cudaReal *delKsq,
                                             int paramN,
                                             int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    result[i] = qkSum[i] * delKsq[paramN * size + i];
                }
            }

            static __global__ void mulKsqFFT(cudaReal *result,
                                             cudaReal *qkSum,
                                             const cudaReal *delKsq,
                                             int paramN, int kSize, int rSize)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < kSize; i += nThreads)
                {
                    result[i] = qkSum[i] * delKsq[paramN * rSize + i];
                    // printf("%f * %f = %f\n", qkSum[i], delKsq[paramN * rSize + i], result[i]);
                }
            }

            static __global__ void equalize(const cudaReal *a, double *result, int size)
            {
                // try to add elements of array here itself

                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    result[i] = a[i];
                }
            }

            static __global__ void pointwiseMulUnroll2(const cudaReal *a, const cudaReal *b, cudaReal *result, int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
                cudaReal localResult[2];
                for (int i = startID; i < size; i += nThreads * 2)
                {
                    localResult[0] = a[i] * b[i];
                    localResult[1] = a[i + 1] * b[i + 1];
                    result[i] = localResult[0];
                    result[i + 1] = localResult[1];
                    // result[i] = a[i] * b[i];
                    // result[i + 1] = a[i + 1] * b[i + 1];
                }
            }

            static __global__ void pointwiseMulCombi(cudaReal *a, const cudaReal *b, cudaReal *c, const cudaReal *d, const cudaReal *e, int size)
            {
                // c = a * b
                // a = d * e
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                cudaReal tempA;
                for (int i = startID; i < size; i += nThreads)
                {
                    tempA = a[i];
                    c[i] = tempA * b[i];
                    a[i] = d[i] * e[i];
                }
            }

            static __global__ void pointwiseMulSameStart(const cudaReal *a,
                                                         const cudaReal *expW,
                                                         const cudaReal *expW2,
                                                         cudaReal *q1, cudaReal *q2,
                                                         int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                cudaReal input;
                for (int i = startID; i < size; i += nThreads)
                {
                    input = a[i];
                    q1[i] = expW[i] * input;
                    q2[i] = expW2[i] * input;
                }
            }

            static __global__ void pointwiseMulTwinned(const cudaReal *qr1,
                                                       const cudaReal *qr2,
                                                       const cudaReal *expW,
                                                       cudaReal *q1, cudaReal *q2,
                                                       int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                cudaReal scale;
                for (int i = startID; i < size; i += nThreads)
                {
                    scale = expW[i];
                    q1[i] = qr1[i] * scale;
                    q2[i] = qr2[i] * scale;
                }
            }

            static __global__ void scaleComplexTwinned(cudaComplex *qk1,
                                                       cudaComplex *qk2,
                                                       const cudaReal *expksq1,
                                                       const cudaReal *expksq2, int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    qk1[i].x *= expksq1[i];
                    qk1[i].y *= expksq1[i];
                    qk2[i].x *= expksq2[i];
                    qk2[i].y *= expksq2[i];
                }
            }

            static __global__ void scaleComplex(cudaComplex *a, cudaReal *scale, int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    a[i].x *= scale[i];
                    a[i].y *= scale[i];
                }
            }

            static __global__ void scaleRealPointwise(cudaReal *a, cudaReal *scale, int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    a[i] *= scale[i];
                }
            }

            static __global__ void assignExp(cudaReal *expW, const cudaReal *w, int size, double cDs)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    expW[i] = exp(-w[i] * cDs);
                }
            }

#if REPS == 0
            static __global__ void richardsonExp_0(cudaReal *qNew,
                                                   const cudaReal *q1,
                                                   int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    qNew[i] = q1[i];
                }
            }
#endif
#if REPS == 1
            static __global__ void richardsonExp_1(cudaReal *qNew,
                                                   const cudaReal *q1,
                                                   const cudaReal *q2,
                                                   int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    qNew[i] = (4.0 * q2[i] - q1[i]) / 3.0;
                }
            }
#endif
#if REPS == 2
            static __global__ void richardsonExp_2(cudaReal *qNew,
                                                   const cudaReal *q1,
                                                   const cudaReal *q2,
                                                   const cudaReal *q3,
                                                   int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    qNew[i] = (64 * q3[i] - 20 * q2[i] + q1[i]) / 45.0;
                }
            }
#endif
#if REPS == 3
            static __global__ void richardsonExp_3(cudaReal *qNew,
                                                   const cudaReal *q1,
                                                   const cudaReal *q2,
                                                   const cudaReal *q3,
                                                   const cudaReal *q4,
                                                   int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    qNew[i] = (4096 * q4[i] - 1344 * q3[i] + 84 * q2[i] - q1[i]) / 2835.0;
                }
            }
#endif
#if REPS == 4
            static __global__ void richardsonExp_4(cudaReal *qNew,
                                                   const cudaReal *q1,
                                                   const cudaReal *q2,
                                                   const cudaReal *q3,
                                                   const cudaReal *q4,
                                                   const cudaReal *q5,
                                                   int size)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = startID; i < size; i += nThreads)
                {
                    qNew[i] = (1048576 * q5[i] - 348160 * q4[i] + 22848 * q3[i] - 340 * q2[i] + q1[i]) / 722925.0;
                }
            }
#endif

            using namespace Util;

            static __global__ void multiplyScaleQQ(cudaReal *result,
                                                   const cudaReal *p1,
                                                   const cudaReal *p2,
                                                   int size, double scale)
            {

                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;

                for (int i = startID; i < size; i += nThreads)
                {
                    result[i] += scale * p1[i] * p2[i];
                }
            }
            static __global__ void multiplyComplex(cudaReal *result,
                                                   const cudaComplex *p1,
                                                   const cudaComplex *p2,
                                                   int size)
            {

                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;

                for (int i = startID; i < size; i += nThreads)
                {
                    result[i] = cuCmul(p1[i], cuConj(p2[i])).x;
                }
            }
            static __global__ void multiplyReal(cudaReal *result,
                                                const cudaReal *p1,
                                                const cudaReal *p2,
                                                int size)
            {

                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;

                for (int i = startID; i < size; i += nThreads)
                {
                    result[i] = p1[i] * p2[i];
                }
            }

            static __global__ void scaleReal(cudaReal *result, int size, double scale)
            {
                int nThreads = blockDim.x * gridDim.x;
                int startID = blockIdx.x * blockDim.x + threadIdx.x;

                for (int i = startID; i < size; i += nThreads)
                {
                    result[i] *= scale;
                }
            }
            /*
             * Constructor.
             */
            template <int D>
            Block<D>::Block()
                : meshPtr_(0),
                  ds_(0.0),
                  ns_(0)
            {
                propagator(0).setBlock(*this);
                propagator(1).setBlock(*this);
            }

            /*
             * Destructor.
             */
            template <int D>
            Block<D>::~Block()
            {
                delete[] temp_;
                cudaFree(d_temp_);

#if DFT == 0
                qF.deallocate();
                qsF.deallocate();
                qtmpF.deallocate();
                qstmpF.deallocate();
#endif
#if DFT == 1
                cudaFree(qtmp);
                cudaFree(qstmp);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                delete[] expKsq_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                delete[] expKsq2_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                delete[] expKsq3_host;
#endif
#if REPS == 4 || REPS == 3
                delete[] expKsq4_host;
#endif
#if REPS == 4
                delete[] expKsq5_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                expKsq_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                expKsq2_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                expKsq3_.deallocate();
#endif
#if REPS == 4 || REPS == 3
                expKsq4_.deallocate();
#endif
#if REPS == 4
                expKsq5_.deallocate();
#endif

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                expW_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                expW2_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                expW3_.deallocate();
#endif
#if REPS == 4 || REPS == 3
                expW4_.deallocate();
#endif
#if REPS == 4
                expW5_.deallocate();
#endif
                delete[] ic_;
                cudaFree(qkSum_);
            }

            template <int D>
            void Block<D>::setDiscretization(double ds, const Mesh<D> &mesh)
            {
                UTIL_CHECK(mesh.size() > 1)
                UTIL_CHECK(ds > 0.0)

                // Set association to mesh
                meshPtr_ = &mesh;

                int ns_tmp = floor(length() / ds + 0.5) + 1;
               
                int n = 1;
                ns_ = 0;
#if REPS == 1 
                while (ns_ < ns_tmp || ns_ % 2 != 0)
                {
                    ns_ += n*(n+1);
                    ++n;
                }     
#endif
#if REPS == 2 
                while (ns_ < ns_tmp || ns_ % 4 != 0)
                {
                    ns_ += n*(n+1);
                    ++n;
                }     
#endif
#if REPS == 3
                while (ns_ < ns_tmp || ns_ % 8 != 0)
                {
                    ns_ += n*(n+1);
                    ++n;
                }     
#endif
#if REPS == 4
                while (ns_ < ns_tmp || ns_ % 16 != 0)
                {
                    ns_ += n*(n+1);
                    ++n;
                }     
#endif
                ns_ += 1; 

                // Set contour length discretization
                ds_ = length() / double(ns_ - 1);
                // Allocate work arrays
                size_ph_ = meshPtr_->size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size_ph_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

#if DFT == 0
                // Compute Fourier space kMeshDimensions_
                for (int i = 0; i < D; ++i)
                {
                    if (i < D - 1)
                    {
                        kMeshDimensions_[i] = meshPtr_->dimensions()[i];
                    }
                    else
                    {
                        kMeshDimensions_[i] = meshPtr_->dimensions()[i] / 2 + 1;
                    }
                }
                kSize_ = 1;
                for (int i = 0; i < D; ++i)
                {
                    kSize_ *= kMeshDimensions_[i];
                }
                // Allocate work arrays
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                expKsq_.allocate(kMeshDimensions_);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                expKsq2_.allocate(kMeshDimensions_);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                expKsq3_.allocate(kMeshDimensions_);
#endif
#if REPS == 4 || REPS == 3
                expKsq4_.allocate(kMeshDimensions_);
#endif
#if REPS == 4
                expKsq5_.allocate(kMeshDimensions_);
#endif
#endif

#if DFT == 1
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                expKsq_.allocate(size_ph_);

#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                expKsq2_.allocate(size_ph_);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                expKsq3_.allocate(size_ph_);
#endif
#if REPS == 4 || REPS == 3
                expKsq4_.allocate(size_ph_);
#endif
#if REPS == 4
                expKsq5_.allocate(size_ph_);
#endif
                cudaMalloc((void **)&qtmp, size_ph_ * sizeof(cudaReal));
                cudaMalloc((void **)&qstmp, size_ph_ * sizeof(cudaReal));
#endif

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                expW_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                expW2_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                expW3_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3
                expW4_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4
                expW5_.allocate(meshPtr_->dimensions());
#endif

                qr_.allocate(meshPtr_->dimensions());
#if DFT == 0
                qF.allocate(meshPtr_->dimensions());
                qsF.allocate(meshPtr_->dimensions());
                qk_.allocate(meshPtr_->dimensions());
                qtmpF.allocate(meshPtr_->dimensions());
                qstmpF.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                q1_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                q2_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                q3_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3
                q4_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4
                q5_.allocate(meshPtr_->dimensions());
#endif

                propagator(0).allocate(ns_, mesh);
                propagator(1).allocate(ns_, mesh);

                cField().allocate(meshPtr_->dimensions());

                cudaMalloc((void **)&d_temp_,
                           NUMBER_OF_BLOCKS * sizeof(cudaReal));
                temp_ = new cudaReal[NUMBER_OF_BLOCKS];

#if DFT == 0
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                expKsq_host = new cudaReal[kSize_];
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                expKsq2_host = new cudaReal[kSize_];
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                expKsq3_host = new cudaReal[kSize_];
#endif
#if REPS == 4 || REPS == 3
                expKsq4_host = new cudaReal[kSize_];
#endif
#if REPS == 4
                expKsq5_host = new cudaReal[kSize_];
#endif
#endif

#if DFT == 1
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                expKsq_host = new cudaReal[size_ph_];
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                expKsq2_host = new cudaReal[size_ph_];
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                expKsq3_host = new cudaReal[size_ph_];
#endif
#if REPS == 4 || REPS == 3
                expKsq4_host = new cudaReal[size_ph_];
#endif
#if REPS == 4
                expKsq5_host = new cudaReal[size_ph_];
#endif
                dqk_c = new cudaReal[6 * size_ph_];
                cudaMalloc((void **)&dqk_,
                           6 * size_ph_ * sizeof(cudaReal));
#endif
                ic_ = new double[ns_];

                for (int i = 0; i < ns_; ++i)
                {
                    ic_[i] = 0.0;
                }

                int round;
#if REPS == 0
                round = (ns_ - 1);
                for (int i = 0; i < round; ++i)
                {
                    ic_[i] += (0.5 * ds_);
                    ic_[i + 1] += (0.5 * ds_);
                }
#endif
#if REPS == 1
                round = (ns_ - 1) / 2;
                for (int i = 0; i < round; ++i)
                {
                    ic_[2 * i] += (ds_ / 3.0);
                    ic_[2 * i + 1] += (4.0 * ds_ / 3.0);
                    ic_[2 * i + 2] += (ds_ / 3.0);
                }

#endif
#if REPS == 2
                round = (ns_ - 1) / 4;
                for (int i = 0; i < round; ++i)
                {
                    ic_[4 * i] += (14.0 * ds_ / 45.0);
                    ic_[4 * i + 1] += (64.0 * ds_ / 45.0);
                    ic_[4 * i + 2] += (24.0 * ds_ / 45.0);
                    ic_[4 * i + 3] += (64.0 * ds_ / 45.0);
                    ic_[4 * i + 4] += (14.0 * ds_ / 45.0);
                }
#endif
#if REPS == 3
                round = (ns_ - 1) / 8;
                for (int i = 0; i < round; ++i)
                {
                    ic_[8 * i] += (868.0 * ds_ / 2835.0);
                    ic_[8 * i + 1] += (4096.0 * ds_ / 2835.0);
                    ic_[8 * i + 2] += (1408.0 * ds_ / 2835.0);
                    ic_[8 * i + 3] += (4096.0 * ds_ / 2835.0);
                    ic_[8 * i + 4] += (1744.0 * ds_ / 2835.0);
                    ic_[8 * i + 5] += (4096.0 * ds_ / 2835.0);
                    ic_[8 * i + 6] += (1408.0 * ds_ / 2835.0);
                    ic_[8 * i + 7] += (4096.0 * ds_ / 2835.0);
                    ic_[8 * i + 8] += (868.0 * ds_ / 2835.0);
                }
#endif
#if REPS == 4
                round = (ns_ - 1) / 16;
                for (int i = 0; i < round; ++i)
                {
                    ic_[16 * i] += (220472.0 * ds_ / 722925.0);
                    ic_[16 * i + 1] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 2] += (352256.0 * ds_ / 722925.0);
                    ic_[16 * i + 3] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 4] += (443648.0 * ds_ / 722925.0);
                    ic_[16 * i + 5] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 6] += (352256.0 * ds_ / 722925.0);
                    ic_[16 * i + 7] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 8] += (440928.0 * ds_ / 722925.0);
                    ic_[16 * i + 9] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 10] += (352256.0 * ds_ / 722925.0);
                    ic_[16 * i + 11] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 12] += (443648.0 * ds_ / 722925.0);
                    ic_[16 * i + 13] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 14] += (352256.0 * ds_ / 722925.0);
                    ic_[16 * i + 15] += (1048576.0 * ds_ / 722925.0);
                    ic_[16 * i + 16] += (220472.0 * ds_ / 722925.0);
                }
#endif
#if DFT == 0
                cudaMalloc((void **)&qkSum_, kSize_ * sizeof(cudaReal));
#endif
#if DFT == 1
                cudaMalloc((void **)&qkSum_, size_ph_ * sizeof(cudaReal));
#endif
            }

            /*
             * Setup data that depend on the unit cell parameters.
             */
            template <int D>
            void
            Block<D>::setupUnitCell(const UnitCell<D> &unitCell,
                                    const WaveList<D> &wavelist)
            {
                nParams_ = unitCell.nParameter();
#if DFT == 0
                MeshIterator<D> iter;
                iter.setDimensions(kMeshDimensions_);
                IntVec<D> G, Gmin;
                double Gsq;
                double factor = -kuhn() * kuhn() * ds_;
                int kSize = 1;
                for (int i = 0; i < D; ++i)
                {
                    kSize *= kMeshDimensions_[i];
                }

                int idx;
                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    idx = iter.rank();
                    Gsq = unitCell.ksq(wavelist.minImage(iter.rank()));
                    // std::cout << wavelist.minImage(iter.rank()) << "\n";
                    // exit(1);
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                    expKsq_host[idx] = exp(Gsq * factor);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                    expKsq2_host[idx] = exp(0.5 * Gsq * factor);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                    expKsq3_host[idx] = exp(0.25 * Gsq * factor);
#endif
#if REPS == 4 || REPS == 3
                    expKsq4_host[idx] = exp(0.125 * Gsq * factor);
#endif
#if REPS == 4
                    expKsq5_host[idx] = exp(0.0625 * Gsq * factor);
#endif
                }

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                cudaMemcpy(expKsq_.cDField(), expKsq_host,
                           kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                cudaMemcpy(expKsq2_.cDField(), expKsq2_host,
                           kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                cudaMemcpy(expKsq3_.cDField(), expKsq3_host,
                           kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3
                cudaMemcpy(expKsq4_.cDField(), expKsq4_host,
                           kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4
                cudaMemcpy(expKsq5_.cDField(), expKsq5_host,
                           kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#endif

#if DFT == 1
                MeshIterator<D> iter;
                IntVec<D> G, Gmin, m, p;
                m = meshPtr_->dimensions();
                iter.setDimensions(m);
                double Gsq;
                double factor = -kuhn() * kuhn() * ds_;

                int idx;
                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    idx = iter.rank();
                    p = iter.position();
                    Gsq = unitCell.ksq(p);

                    double factor = -1.0 * kuhn() * kuhn() * ds_;
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                    expKsq_host[idx] = exp(Gsq * factor);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                    expKsq2_host[idx] = exp(0.5 * Gsq * factor);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                    expKsq3_host[idx] = exp(0.25 * Gsq * factor);
#endif
#if REPS == 4 || REPS == 3
                    expKsq4_host[idx] = exp(0.125 * Gsq * factor);
#endif
#if REPS == 4
                    expKsq5_host[idx] = exp(0.0625 * Gsq * factor);
#endif
                }

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                cudaMemcpy(expKsq_.cDField(), expKsq_host,
                           size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                cudaMemcpy(expKsq2_.cDField(), expKsq2_host,
                           size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                cudaMemcpy(expKsq3_.cDField(), expKsq3_host,
                           size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3
                cudaMemcpy(expKsq4_.cDField(), expKsq4_host,
                           size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4
                cudaMemcpy(expKsq5_.cDField(), expKsq5_host,
                           size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
                int h, k, l, tmpidx;

                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    idx = iter.rank();
                    G = iter.position();

                    h = idx / (m[1] * m[2]);
                    tmpidx = idx % (m[1] * m[2]);
                    k = tmpidx / m[2];
                    l = tmpidx % m[2];
                    // std::cout << h << " " << k << " " << l << "\n";
                    for (int n = 0; n < nParams_; ++n)
                    {
                        if (h == 0 && k == 0 && l == 0)
                            dqk_c[idx + n * size_ph_] = unitCell.dksq(G, n);
                        else if (h == 0 && k == 0 && l != 0)
                            dqk_c[idx + n * size_ph_] = 2.0 * unitCell.dksq(G, n);
                        else if (h == 0 && k != 0 && l == 0)
                            dqk_c[idx + n * size_ph_] = 2.0 * unitCell.dksq(G, n);
                        else if (h != 0 && k == 0 && l == 0)
                            dqk_c[idx + n * size_ph_] = 2.0 * unitCell.dksq(G, n);
                        else if (h != 0 && k != 0 && l == 0)
                            dqk_c[idx + n * size_ph_] = 4.0 * unitCell.dksq(G, n);
                        else if (h != 0 && k == 0 && l != 0)
                            dqk_c[idx + n * size_ph_] = 4.0 * unitCell.dksq(G, n);
                        else if (h == 0 && k != 0 && l != 0)
                            dqk_c[idx + n * size_ph_] = 4.0 * unitCell.dksq(G, n);
                        else
                            dqk_c[idx + n * size_ph_] = 8.0 * unitCell.dksq(G, n);
                        // std::cout << dqk_c[idx+n*size_ph_] << "\n";
                        // if (idx > 128)
                        //
                    }
                }
                // exit(1);

                cudaMemcpy(dqk_, dqk_c,
                           nParams_ * size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
            }

            template <int D>
            void
            Block<D>::setWavelist(WaveList<D> &wavelist)
            {
                wavelist_ = new WaveList<D>();
                wavelist_ = &wavelist;
            }
            /*
             * Setup the contour length step algorithm.
             */
            template <int D>
            void
            Block<D>::setupSolver(Block<D>::WField const &w)
            {
                UTIL_CHECK(size_ph_ > 0)
                int Discretization_of_each_block = ns() - 1;
                // Discretization of each block should satisfies the following
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size_ph_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
#if REPS == 4
                UTIL_CHECK(Discretization_of_each_block % 16 == 0)
#endif
#if REPS == 3
                UTIL_CHECK(Discretization_of_each_block % 8 == 0)
#endif
#if REPS == 2
                UTIL_CHECK(Discretization_of_each_block % 4 == 0)
#endif
#if REPS == 1
                UTIL_CHECK(Discretization_of_each_block % 2 == 0)
#endif

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0

                assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(expW_.cDField(), w.cDField(), size_ph_, (double)0.5 * ds_);

#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(expW2_.cDField(), w.cDField(), size_ph_, (double)0.25 * ds_);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(expW3_.cDField(), w.cDField(), size_ph_, (double)0.125 * ds_);
#endif
#if REPS == 4 || REPS == 3
                assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(expW4_.cDField(), w.cDField(), size_ph_, (double)0.0625 * ds_);
#endif
#if REPS == 4
                assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(expW5_.cDField(), w.cDField(), size_ph_, (double)0.03125 * ds_);
#endif
            }

#if DFT == 0
            template <int D>
            void Block<D>::setupFFT()
            {
                if (!fft_.isSetup())
                {
                    // std::cout<<"setting up batch fft with the following values"<<std::endl;
                    // std::cout<<mesh().dimensions()<<'\n'<<kMeshDimensions_<<'\n'<<ns_<<'\n';
                    fft_.setup(qr_, qk_);
                }
            }
#endif
#if DFT == 1
            template <int D>
            void Block<D>::setupFCT()
            {
                if (!fct_.isSetup())
                {
                    int mesh[3];
                    if (D == 3)
                    {
                        mesh[0] = meshPtr_->dimension(0);
                        mesh[1] = meshPtr_->dimension(1);
                        mesh[2] = meshPtr_->dimension(2);
                    }
                    else if (D == 2)
                    {
                        mesh[0] = 1;
                        mesh[1] = meshPtr_->dimension(0);
                        mesh[2] = meshPtr_->dimension(1);
                    }
                    else
                    {
                        mesh[0] = 1;
                        mesh[1] = 1;
                        mesh[2] = meshPtr_->dimension(0);
                    }

                    fct_.setup(mesh);
                }
            }
#endif
            /*
             * Propagate solution by one step.
             */
            template <int D>
            void Block<D>::step(const cudaReal *q, cudaReal *qNew)
            {
                UTIL_CHECK(size_ph_ > 0)
                UTIL_CHECK(qr_.capacity() == size_ph_)
                UTIL_CHECK(expW_.capacity() == size_ph_)

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size_ph_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

#if DFT == 0
                // Fourier-space mesh sizes
                int nx = mesh().size();
                int nk = qk_.capacity();
                UTIL_CHECK(expKsq_.capacity() == nk)
                /// Apply pseudo-spectral algorithm
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                // 1-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW_.cDField(), qr_.cDField(), nx);
                fft_.forwardTransform(qr_, qk_);

                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW_.cDField(), q1_.cDField(), nx);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                // 2-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW2_.cDField(), qr_.cDField(), nx);
                fft_.forwardTransform(qr_, qk_);
                ThreadGrid::setThreadsLogical(nk, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq2_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW_.cDField(), qr_.cDField(), nx);

                fft_.forwardTransform(qr_, qk_);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq2_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW2_.cDField(), q2_.cDField(), nx);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                // 4-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW3_.cDField(), qr_.cDField(), nx);
                fft_.forwardTransform(qr_, qk_);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq3_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW2_.cDField(), qr_.cDField(), nx);

                for (int i = 0; i < 2; ++i)
                {
                    fft_.forwardTransform(qr_, qk_);
                    scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq3_.cDField(), nk);
                    fft_.inverseTransform(qk_, qr_);
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW2_.cDField(), qr_.cDField(), nx);
                }

                fft_.forwardTransform(qr_, qk_);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq3_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW3_.cDField(), q3_.cDField(), nx);
#endif
#if REPS == 4 || REPS == 3
                // 8-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW4_.cDField(), qr_.cDField(), nx);
                fft_.forwardTransform(qr_, qk_);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq4_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW3_.cDField(), qr_.cDField(), nx);

                for (int i = 0; i < 6; ++i)
                {
                    fft_.forwardTransform(qr_, qk_);
                    scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq4_.cDField(), nk);
                    fft_.inverseTransform(qk_, qr_);
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW3_.cDField(), qr_.cDField(), nx);
                }

                fft_.forwardTransform(qr_, qk_);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq4_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW4_.cDField(), q4_.cDField(), nx);
#endif
#if REPS == 4
                // 16-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW5_.cDField(), qr_.cDField(), nx);
                fft_.forwardTransform(qr_, qk_);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq5_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW4_.cDField(), qr_.cDField(), nx);

                for (int i = 0; i < 14; ++i)
                {
                    fft_.forwardTransform(qr_, qk_);
                    scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq5_.cDField(), nk);
                    fft_.inverseTransform(qk_, qr_);
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW4_.cDField(), qr_.cDField(), nx);
                }

                fft_.forwardTransform(qr_, qk_);
                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq5_.cDField(), nk);
                fft_.inverseTransform(qk_, qr_);
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW5_.cDField(), q5_.cDField(), nx);
#endif
#endif
#if DFT == 1
                /// Apply pseudo-spectral algorithm
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                // 1-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW_.cDField(), qr_.cDField(), size_ph_);
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW_.cDField(), q1_.cDField(), size_ph_);

#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                // 2-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW2_.cDField(), qr_.cDField(), size_ph_);
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq2_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW_.cDField(), size_ph_);

                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq2_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW2_.cDField(), q2_.cDField(), size_ph_);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                // 4-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW3_.cDField(), qr_.cDField(), size_ph_);
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq3_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW2_.cDField(), qr_.cDField(), size_ph_);

                for (int i = 0; i < 2; ++i)
                {
                    fct_.forwardTransform(qr_.cDField());
                    scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq3_.cDField(), size_ph_);
                    fct_.inverseTransform(qr_.cDField());
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW2_.cDField(), qr_.cDField(), size_ph_);
                }

                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq3_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW3_.cDField(), q3_.cDField(), size_ph_);
#endif
#if REPS == 4 || REPS == 3
                // 8-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW4_.cDField(), qr_.cDField(), size_ph_);
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq4_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW3_.cDField(), qr_.cDField(), size_ph_);

                for (int i = 0; i < 6; ++i)
                {
                    fct_.forwardTransform(qr_.cDField());
                    scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq4_.cDField(), size_ph_);
                    fct_.inverseTransform(qr_.cDField());
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW3_.cDField(), qr_.cDField(), size_ph_);
                }

                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq4_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW4_.cDField(), q4_.cDField(), size_ph_);
#endif
#if REPS == 4
                // 16-step
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, expW5_.cDField(), qr_.cDField(), size_ph_);
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq5_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW4_.cDField(), qr_.cDField(), size_ph_);

                for (int i = 0; i < 14; ++i)
                {
                    fct_.forwardTransform(qr_.cDField());
                    scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq5_.cDField(), size_ph_);
                    fct_.inverseTransform(qr_.cDField());
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW4_.cDField(), qr_.cDField(), size_ph_);
                }

                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expKsq5_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW5_.cDField(), q5_.cDField(), size_ph_);
#endif
#endif
#if REPS == 0
                richardsonExp_0<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qNew,
                                                                         q1_.cDField(),
                                                                         size_ph_);

#endif
#if REPS == 1
                richardsonExp_1<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qNew,
                                                                         q1_.cDField(),
                                                                         q2_.cDField(),
                                                                         size_ph_);
#endif
#if REPS == 2
                richardsonExp_2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qNew,
                                                                         q1_.cDField(),
                                                                         q2_.cDField(),
                                                                         q3_.cDField(),
                                                                         size_ph_);
#endif
#if REPS == 3
                richardsonExp_3<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qNew,
                                                                         q1_.cDField(),
                                                                         q2_.cDField(),
                                                                         q3_.cDField(),
                                                                         q4_.cDField(),
                                                                         size_ph_);
#endif

#if REPS == 4
                richardsonExp_4<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qNew,
                                                                         q1_.cDField(),
                                                                         q2_.cDField(),
                                                                         q3_.cDField(),
                                                                         q4_.cDField(),
                                                                         q5_.cDField(),
                                                                         size_ph_);
#endif
                // cudaReal qc[2], qcNew[2];
                // cudaMemcpy(qc, q, sizeof(cudaReal)*2, cudaMemcpyDeviceToHost);
                // cudaMemcpy(qcNew, qNew, sizeof(cudaReal)*2, cudaMemcpyDeviceToHost);
                // std::cout << qc[0] << " -> " << qcNew[0] << "\n";
                // std::cout << qc[1] << " -> " << qcNew[1] << "\n\n";
            }

            /*
             * Compute stresses..
             */
            template <int D>
            void Block<D>::computeStress(double prefactor)
            {
                FArray<double, 6> dQ;
                int i;
                for (i = 0; i < 6; ++i)
                {
                    dQ[i] = 0.0;
                    stress_[i] = 0.0;
                }

                for (i = 0; i < nParams_; ++i)
                {
                    inc_[i] = inc_[i] * kuhn() * kuhn();
                    dQ[i] -= inc_[i];
                }
                for (i = 0; i < nParams_; ++i)
                    stress_[i] -= (dQ[i] * prefactor);
            }

            template <int D>
            void Block<D>::computeInt(cudaReal *q, cudaReal *qs, int ic)
            {
                // Preconditions
                UTIL_CHECK(size_ph_ > 0)
                UTIL_CHECK(ns_ > 0)
                UTIL_CHECK(ds_ > 0)
                UTIL_CHECK(propagator(0).isAllocated())
                UTIL_CHECK(propagator(1).isAllocated())

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size_ph_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
                // Initialize cField to zero at all points
                // cField()[i] = 0.0;
#if DFT == 0
                if (ic == 0)
                {
                    for (int i = 0; i < nParams_; ++i)
                        inc_[i] = 0.0;

                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), 0.0, size_ph_);
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qkSum_, 0.0, kSize_);
                }
#endif
#if DFT == 1

                if (ic == 0)
                {
                    for (int i = 0; i < nParams_; ++i)
                        inc_[i] = 0.0;

                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), 0.0, size_ph_);
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qkSum_, 0.0, size_ph_);
                }
#endif

                multiplyScaleQQ<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), q, qs, size_ph_, ic_[ic]);

#if DFT == 0
                assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qF.cDField(), q, size_ph_);
                assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsF.cDField(), qs, size_ph_);

                fft_.forwardTransform(qF, qtmpF);
                fft_.forwardTransform(qsF, qstmpF);
                multiplyComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qkSum_, qtmpF.cDField(), qstmpF.cDField(), kSize_);
                cudaReal *tmp;
                cudaMalloc((void **)&tmp, kSize_ * sizeof(cudaReal));
                // assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), 0.0, size_ph_);
                for (int i = 0; i < nParams_; ++i)
                {
                    mulKsqFFT<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qkSum_, wavelist_->dkSq(), i, kSize_, size_ph_);
                    // double sum = 0.0;
                    // cudaReal *tmp_c;
                    // tmp_c = new cudaReal[kSize_];
                    // cudaMemcpy(tmp_c, cField().cDField(), sizeof(cudaReal)*kSize_, cudaMemcpyDeviceToHost);
                    // std::cout << tmp_c[0] << "\n";
                    // for (int j = 0; j < 64; ++j)
                    //     std::cout << tmp_c[j] << "\n";
                    // inc_[i] += ic_[ic]*sum;
                    inc_[i] += ic_[ic] * gpuSum(tmp, kSize_);
                    // std::cout << inc_[i] << "\n";
                    // inc_[i] += ic_[ic]*reductionH(tmp, kSize_);
                    // std::cout << reductionH(tmp, kSize_)<< "\n\n\n";
                    // delete [] tmp_c;
                }
                cudaFree(tmp);
                // exit(1);

#endif
#if DFT == 1
                assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qtmp, q, size_ph_);
                assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qstmp, qs, size_ph_);
                fct_.forwardTransform(qtmp);
                fct_.forwardTransform(qstmp);
                // cudaMemset(qkSum_, 0, size_ph_*sizeof(cudaReal));
                multiplyReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qkSum_, qtmp, qstmp, size_ph_);
                cudaReal *tmp;
                cudaMalloc((void **)&tmp, size_ph_ * sizeof(cudaReal));
                for (int i = 0; i < nParams_; ++i)
                {
                    mulKsqFCT<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qkSum_, dqk_, i, size_ph_);
                    double sum = 0.0;
                    cudaReal *tmp_c;
                    tmp_c = new cudaReal[size_ph_];
                    cudaMemcpy(tmp_c, tmp, sizeof(cudaReal) * size_ph_, cudaMemcpyDeviceToHost);
                    for (int j = 0; j < size_ph_; ++j)
                        sum += tmp_c[j];
                    inc_[i] += ic_[ic] * sum;
                    delete[] tmp_c;
                }
                cudaFree(tmp);

#endif
            }

        }
    }
}
#endif
