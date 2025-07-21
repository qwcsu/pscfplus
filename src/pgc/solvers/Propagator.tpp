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
                if (this->directionFlag() == 0)
                    cudaMalloc((void **)&qFields_d, sizeof(cudaReal) * nx * (nc_ + 1));
                else
                    cudaMalloc((void **)&qFields_d, sizeof(cudaReal) * nx);

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
                    // need to modify tail to give the total_size - mesh_size pointer
                    if (!source(is).isSolved())
                    {
                        UTIL_THROW("Source not solved in computeHead");
                    }

                    if (source(is).directionFlag()==0)
                        qt = source(is).qtail();
                    else
                        qt = source(is).head();
                    // qh[ix] *= qt[ix];
                    inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d, qt, nx);
                }
#if DFT == 0
                block().setupFFT();
#endif
#if DFT == 1
                block().setupFCT();
#endif
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

                int cl[nc_ + 1];
                cl[0] = ns_ - nc_ * (nc_ + 1) * 0.5;
                

                for (int i = 1; i <= nc_; ++i)
                    cl[i] = nc_ - i + 1;

                cudaReal *tmp;
                cudaMalloc((void **)&tmp, sizeof(cudaReal) * nx);
                // std::cout << block().fft().isSetup() << "\n";
                

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
            void Propagator<D>::solveBackward(cudaReal *q, bool isReused)
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
                
                Q_ = intQ(q + nc_ * nx, qFields_d);
                
#if DFT == 0
                block().setupFFT();
#endif
#if DFT == 1
                block().setupFCT();
#endif
                cudaReal *q_tmp, *qd_tmp;
                cudaReal *rtmp;
                cudaMalloc((void **)&q_tmp, sizeof(cudaReal) * nx);
                cudaMalloc((void **)&qd_tmp, sizeof(cudaReal) * nx);
                cudaMemcpy(q_tmp, q + nc_ * nx, sizeof(cudaReal) * nx, cudaMemcpyDeviceToDevice);
                if (isReused)
                {
                    cudaMalloc((void **)&rtmp, sizeof(cudaReal)*(nc_+1)*nx);
                    cudaMemcpy(rtmp, q, sizeof(cudaReal)*(nc_+1)*nx, cudaMemcpyDeviceToDevice);
                }

                int cb[nc_ + 1];
                cb[nc_] = ns_ - nc_ * (nc_ + 1) * 0.5;

                for (int i = 1; i <= nc_; ++i)
                    cb[i - 1] = i;

                int icount = 0;
                block().computeInt(q + nc_ * nx, qFields_d, icount);
                ++icount;
                // block().step(qFields_d, qd_tmp);
                // cudaMemcpy(qFields_d, qd_tmp, sizeof(cudaReal) * nx, cudaMemcpyDeviceToDevice);
                for (int i = 1; i < nc_; ++i)
                {
                    for (int j = 0; j < cb[i] - 1; ++j)
                    {
                        // std::cout << "q (" << nc_-i+j << "," << nc_-i+j+1 <<")\n";
                        // std::cout << "q*(" << nc_-j << "," << nc_-j-1 <<")\n";
                        block().step(q + (nc_ - i + j) * nx, q + (nc_ - i + j + 1) * nx);
                    }
                    // std::cout << "\n";
                    for (int j = 0; j < cb[i]; ++j)
                    {
                        // std::cout << "int(" << nc_-j << "," << nc_-j <<")\n";
                        block().step(qFields_d, qd_tmp);
                        cudaMemcpy(qFields_d, qd_tmp, sizeof(cudaReal) * nx, cudaMemcpyDeviceToDevice);
                        block().computeInt(q + (nc_ - j) * nx, qFields_d, icount);
                        ++icount;
                        // double a[2];
                        // cudaMemcpy(a, qFields_d, sizeof(cudaReal) * 2, cudaMemcpyDeviceToHost);
                        // std::cout << a[0] <<"\n";
                        // std::cout << a[1] <<"\n";
                    }
                    // std::cout << "q*(" << nc_-cb[i]+1 << "," << nc_ <<")\n";
                    // std::cout << "\n";
                }
                // exit(1);
                for (int j = 0; j < cb[nc_] - 1; ++j)
                {
                    // std::cout << "q (" << j << "," << j+1 <<")\n";
                    // std::cout << "q*(" << nc_-j << "," << nc_-j-1 <<")\n";
                    block().step(q + j * nx, q + (j + 1) * nx);
                    
                }
                for (int j = cb[nc_]-1; j >= 0; --j)
                {
                    // std::cout << "int(" << j << "," << nc_-cb[nc_]+j+1 <<")\n";
                    block().step(qFields_d, qd_tmp);
                    cudaMemcpy(qFields_d, qd_tmp, sizeof(cudaReal) * nx, cudaMemcpyDeviceToDevice);
                    block().computeInt(q + j * nx, qFields_d, icount);
                    ++icount;
                }
                
                
                // std::cout << icount << "\n";
                // assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d + nc_ * nx, qFields_d + (nc_ - cb[nc_] + 1) * nx, nx);
                // std::cout << block().id() << "\n";
                setIsSolved(true);
                if (isReused)
                {
                    cudaMemcpy(q, rtmp, sizeof(cudaReal)*(nc_+1)*nx, cudaMemcpyDeviceToDevice);
                    cudaFree(rtmp);
                }
                else
                {
                    cudaMemcpy(q + nc_ * nx, q_tmp, sizeof(cudaReal) * nx, cudaMemcpyDeviceToDevice);
                }
                
                cudaFree(q_tmp);
                cudaFree(qd_tmp);
                // exit(1);
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
                // std::cout << "Q=" << Q << "\n";
                return Q;
            }

        }
    }
}
#endif
