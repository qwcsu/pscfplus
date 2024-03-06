#ifndef PG_PROPAGATOR_TPP
#define PG_PROPAGATOR_TPP

/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include "Propagator.h"
#include "Block.h"
// #include <thrust/reduce.h>
#include "device_launch_parameters.h"
#include <cuda.h>
// #include <device_functions.h>
#include <thrust/count.h>
#include <pspg/math/GpuHeaders.h>
#include <pscf/mesh/Mesh.h>
// #include <Windows.h>

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            using namespace Util;

            /*
             * Constructor.
             */
            template <int D>
            Propagator<D>::Propagator()
                : blockPtr_(0),
                  meshPtr_(0),
                  ns_(0),
                  isAllocated_(false)
            {
            }

            /*
             * Destructor.
             */
            template <int D>
            Propagator<D>::~Propagator()
            {
                delete[] temp_;
                cudaFree(d_temp_);
                cudaFree(qFields_d);
                cudaFree(q_cpy);
            }

            template <int D>
            void Propagator<D>::allocate(int ns, const Mesh<D> &mesh)
            {
                ns_ = ns;
                meshPtr_ = &mesh;
                int nx = 1;
                for (int i = 0; i < D; ++i)
                {
                    nx *= meshPtr_->dimensions()[i];
                }
                nc_ = std::floor((std::sqrt(1 + 8 * ns_) - 1) * 0.5);

                // std::cout << nc_ << "\n";

                // cudaMalloc((void**)&qFields_d, sizeof(cudaReal)* nx * ns);
                cudaMalloc((void **)&qFields_d, sizeof(cudaReal) * nx * (nc_ + 1));
                cudaMalloc((void **)&q_cpy, nx* (nc_ + 1) * sizeof(cudaReal));
                cudaMalloc((void **)&d_temp_, ThreadGrid::nBlocks() * sizeof(cudaReal));
                temp_ = new cudaReal[ThreadGrid::nBlocks()];

                isAllocated_ = true;

                ThreadGrid::setThreadsLogical(mesh.size());
            }

            /*
             * Compute initial head QField from final tail QFields of sources.
             */
            template <int D>
            void Propagator<D>::computeHead()
            {

                // Reference to head of this propagator
                // QField& qh = qFields_[0];

                // Initialize qh field to 1.0 at all grid points
                int nx = 1;
                for (int i = 0; i < D; ++i)
                {
                    nx *= meshPtr_->dimensions()[i];
                }
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
                // qh[ix] = 1.0;
                // qFields_d points to the first float in gpu memory
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d, 1.0, nx);

                // Pointwise multiply tail QFields of all sources
                // this could be slow with many sources. Should launch 1 kernel for the whole
                // function of computeHead
                const cudaReal *qt;
                for (int is = 0; is < nSource(); ++is)
                {
                    if (!source(is).isSolved())
                    {
                        UTIL_THROW("Source not solved in computeHead");
                    }
                    // need to modify tail to give the total_size - mesh_size pointer
                    qt = source(is).qtail();

                    // qh[ix] *= qt[ix];
                    inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d, qt, nx);
                }
            }

            template <int D>
            void Propagator<D>::solveForward()
            {
                UTIL_CHECK(isAllocated())
                int nx = 1;
                for (int i = 0; i < D; ++i)
                {
                    nx *= meshPtr_->dimensions()[i];
                }

                computeHead();

#if DFT == 0
                block().setupFFT();
#endif
#if DFT == 1
                block().setupFCT();
#endif
                // std::cout << "nc=" << nc_ << "\n";

                int cl[nc_ + 1];
                cl[0] = ns_ - nc_ * (nc_ + 1) * 0.5;
                // std::cout << cl[0] << "!\n";

                for (int i = 1; i <= nc_; ++i)
                    cl[i] = nc_ - i + 1;

                cudaReal *tmp;
                cudaMalloc((void **)&tmp, sizeof(cudaReal) * nx);

                if (cl[0] == 2 || cl[0] == 0)
                {
                    // std::cout << "(" << 0
                    //           << ", "<< "tmp"
                    //           << "\n";
                    // std::cout << "(" << "tmp"
                    //           << ", "<< "1"
                    //           << "\n";
                    block().step(qFields_d,
                                 tmp);
                    block().step(tmp,
                                 qFields_d + nx);
                }
                else
                {
                    for (int j = 0; j < cl[0] - 1; ++j)
                    {
                        // std::cout << "(" << j
                        //           << ", "<< j+1
                        //           << "\n";
                        block().step(qFields_d + j * nx,
                                     qFields_d + (j + 1) * nx);
                    }

                    // std::cout << "(" << cl[0]-1
                    //           << ", "<< 1
                    //           << "\n";
                    // std::cout << "\n";
                    block().step(qFields_d + (cl[0] - 1) * nx,
                                 qFields_d + nx);
                }
                // std::cout << "\n";
                for (int i = 1; i < nc_ - 1; ++i)
                {
                    for (int j = 0; j < cl[i] - 1; ++j)
                    {
                        // std::cout << "(" << i+j
                        //           << ", "<< i+j+1
                        //           << "\n";
                        block().step(qFields_d + (i + j) * nx,
                                     qFields_d + (i + j + 1) * nx);
                    }
                    // std::cout << "(" << i+cl[i]-1
                    //           << ", "<< i+1
                    //           << "\n";
                    block().step(qFields_d + (i + cl[i] - 1) * nx,
                                 qFields_d + (i + 1) * nx);
                    // std::cout << "\n";
                }
                // std::cout << "(" << nc_-1
                //           << ", "<< "tmp"
                //           << "\n";
                block().step(qFields_d + (nc_ - 1) * nx,
                             tmp);
                // std::cout << "(" << "tmp"
                //           << ", "<< nc_
                //           << "\n";
                block().step(tmp,
                             qFields_d + (nc_)*nx);
                setIsSolved(true);

                // cudaReal a[10];
                // // cudaMalloc((void**)&a, sizeof(cudaReal)*10);
                // cudaMemcpy(a, qFields_d+nx, sizeof(cudaReal)*10, cudaMemcpyDeviceToHost);
                // for(int i = 0; i < 10; ++i)
                //     std::cout << a[i] << "\n";
                // cudaFree(a);
                // exit(1);

                cudaFree(tmp);
            }

            template <int D>
            void Propagator<D>::solveBackward(cudaReal *q, int n)
            {
                UTIL_CHECK(isAllocated())
                int nx = 1;
                for (int i = 0; i < D; ++i)
                {
                    nx *= meshPtr_->dimensions()[i];
                }
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                computeHead();
                // if (n == 0)
                // {
                //     Q_ = intQ(q, qFields_d + nc_ * nx);
                // }
                Q_ = intQ(q + nc_ * nx, qFields_d);

#if DFT == 0
                block().setupFFT();
#endif
#if DFT == 1
                block().setupFCT();
#endif

                // std::cout << ns_ << "\n";

                int cb[nc_ + 1];
                cb[nc_] = ns_ - nc_ * (nc_ + 1) * 0.5;

                for (int i = 1; i <= nc_; ++i)
                    cb[i - 1] = i;

                // for (int i = 0; i <= nc_; ++i)
                //     std::cout << cb[i] << "\n";

                int icount = 0;
                // std::cout << "int(" << nc_ << "," << "0" <<")\n";
                // std::cout << "q*(" << "0" << "," << nc_ <<")\n";
                block().computeInt(q + nc_ * nx, qFields_d, icount);
                
                cudaMemcpy(q_cpy, q, sizeof(cudaReal) * (nc_+1) * nx, cudaMemcpyDeviceToDevice);
                ++icount;
                block().step(qFields_d, qFields_d + nc_ * nx);

                for (int i = 1; i < nc_; ++i)
                {
                    for (int j = 0; j < cb[i] - 1; ++j)
                    {
                        // std::cout << "q (" << nc_-i+j << "," << nc_-i+j+1 <<")\n";
                        // std::cout << "q*(" << nc_-j << "," << nc_-j-1 <<")\n";
                        block().step(q_cpy + (nc_ - i + j) * nx, q_cpy + (nc_ - i + j + 1) * nx);
                        block().step(qFields_d + (nc_ - j) * nx, qFields_d + (nc_ - j - 1) * nx);
                    }
                    for (int j = 0; j < cb[i]; ++j)
                    {
                        // std::cout << "int(" << nc_-j << "," << nc_-j <<")\n";
                        block().computeInt(q_cpy + (nc_ - j) * nx, qFields_d + (nc_ - j) * nx, icount);
                        ++icount;
                    }
                    // std::cout << "q*(" << nc_-cb[i]+1 << "," << nc_ <<")\n";
                    // std::cout << "\n";
                    block().step(qFields_d + (nc_ - cb[i] + 1) * nx, qFields_d + nc_ * nx);
                }
                for (int j = 0; j < cb[nc_] - 1; ++j)
                {
                    // std::cout << "q (" << j << "," << j+1 <<")\n";
                    // std::cout << "q*(" << nc_-j << "," << nc_-j-1 <<")\n";
                    block().step(q_cpy + j * nx, q_cpy + (j + 1) * nx);
                    block().step(qFields_d + (nc_ - j) * nx, qFields_d + (nc_ - j - 1) * nx);
                }

                for (int j = 0; j < cb[nc_]; ++j)
                {
                    // std::cout << "int(" << j << "," << nc_-cb[nc_]+j+1 <<")\n";
                    block().computeInt(q_cpy + j * nx, qFields_d + (nc_ - cb[nc_] + j + 1) * nx, icount);
                    ++icount;
                }
                // std::cout << icount << "\n";
                assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d + nc_ * nx, qFields_d + (nc_ - cb[nc_] + 1) * nx, nx);

                setIsSolved(true);
            }

            template <int D>
            double Propagator<D>::intQ(cudaReal *q, cudaReal *qs)
            {
                int nx = 1;
                for (int i = 0; i < D; ++i)
                {
                    nx *= meshPtr_->dimensions()[i];
                }
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                double Q;
                Q = gpuInnerProduct(q, qs, nx);

                Q /= double(nx);
                // std::cout << "Q=" << Q << "\n";exit(1);
                return Q;
            }

        }
    }
}
#endif
