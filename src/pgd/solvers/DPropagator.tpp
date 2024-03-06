#ifndef D_PROPAGATOR_TPP
#define D_PROPAGATOR_TPP

#include "DPropagator.h"
#include "pgd/solvers/Bond.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <thrust/count.h>
#include <pspg/math/GpuHeaders.h>
#include <pscf/mesh/Mesh.h>

namespace Pscf
{
    namespace Pspg
    {
        static __global__ void scaleReal(cudaReal *a, cudaReal *scale, int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                a[i] *= scale[i];
            }
        }

        static __global__ void inPlacePointwiseDiv(cudaReal *a, const cudaReal *b, int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                a[i] /= b[i];
            }
        }

        namespace Discrete
        {

            using namespace Util;

            template <int D>
            DPropagator<D>::DPropagator()
                : bondPtr_(0),
                  meshPtr_(0),
                  N_(0),
                  isAllocated_(false)
            {
            }

            template <int D>
            DPropagator<D>::~DPropagator()
            {
                cudaFree(qFields_d);
            }

            template <int D>
            void DPropagator<D>::allocate(int N, const Mesh<D> &mesh)
            {
                N_ = N;
                meshPtr_ = &mesh;

                cudaMalloc((void **)&qFields_d, sizeof(cudaReal) * mesh.size() * N);

                isAllocated_ = true;
            }

            template <int D>
            void DPropagator<D>::computeHead()
            {
                int nx = meshPtr_->size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d, 1.0, nx);

                const cudaReal *qt;

                for (int is = 0; is < nSource(); ++is)
                {
                    if (!source(is).isSolved())
                    {
                        UTIL_THROW("Source not solved in computeHead");
                    }

                    qt = source(is).tail();

                    inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d, qt, nx);
                }
            }

            template <int D>
            void DPropagator<D>::solve()
            {
                UTIL_CHECK(isAllocated())
                int nx = meshPtr_->size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                computeHead();

                bond().setupFFT();

                if (bond().bondtype() == 0)
                {
                    bond().step(qFields_d,
                                qFields_d + nx);
                    // cudaReal a[10];
                    // cudaMemcpy(a, qFields_d, 10*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                    // for (int i = 0; i < 10; ++i)
                    //     std::cout << a[i]<< "->";
                    // cudaMemcpy(a, qFields_d + meshPtr_->size(), 10*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                    // for (int i = 0; i < 10; ++i)
                    //     std::cout << a[i] << "\n";
                    // std::cout << "\n";
                }
                else
                {
                    int currentIdx;
                    scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d, bond().expW().cDField(), nx);

                    for (int iStep = 0; iStep < N_ - 1; ++iStep)
                    {
                        currentIdx = iStep * meshPtr_->size();
                        bond().step(qFields_d + currentIdx,
                                    qFields_d + currentIdx + nx);

                        // cudaReal a[10];
                        // cudaMemcpy(a, qFields_d + currentIdx, 10*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                        // for (int i = 0; i < 10; ++i)
                        //     std::cout << a[i]<< "->";
                        // cudaMemcpy(a, qFields_d + currentIdx + meshPtr_->size(), 10*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                        // for (int i = 0; i < 10; ++i)
                        //     std::cout << a[i] << "\n";
                        // std::cout << "\n";
                    }
                    // exit(1);
                }

                setIsSolved(true);
            }

            template <int D>
            cudaReal DPropagator<D>::computeQ()
            {

                if (!isSolved())
                {
                    UTIL_THROW("Propagator is not solved.");
                }

                const cudaReal *qh = head();
                const cudaReal *qt = partner().tail();

                int nx = meshPtr_->size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                cudaReal Q;
                cudaReal *tmp;
                cudaMalloc((void **)&tmp, nx * sizeof(cudaReal));

                assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qh, nx);

                inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qt, nx);

                inPlacePointwiseDiv<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, bond().expW().cDField(), nx);

                // cudaReal *d_tmp;
                // cudaMalloc(&d_tmp, sizeof(cudaReal));
                // cudaMemset(d_tmp, 0, sizeof(cudaReal));
                // sumArray<<<(nx + 32 -1)/32, 32>>>
                // (d_tmp, tmp, nx);
                // cudaMemcpy(&Q, d_tmp, sizeof(cudaReal), cudaMemcpyDeviceToHost);
                // cudaFree(d_tmp);
                Q = gpuSum(tmp, nx);
                Q /= double(nx);
                cudaFree(tmp);
                // exit(1);
                return Q;
            }

        }
    }
}

#endif