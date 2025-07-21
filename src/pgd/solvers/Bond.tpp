#ifndef BOND_TPP
#define BOND_TPP

#include "Bond.h"

namespace Pscf
{
    namespace Pspg
    {
        static __global__ void ExpW(cudaReal *expW, const cudaReal *w,
                                    int size, double cDs)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                expW[i] = exp(-w[i] * cDs);
                // printf("w = %f, cDs = %f\n", w[i], cDs);
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

        static __global__ void scaleReal(cudaReal *a, cudaReal *scale, int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                a[i] *= scale[i];
            }
        }

        static __global__ void scaleRealconst(cudaReal *a, cudaReal scale, int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                a[i] *= scale;
            }
        }

        static __global__ void pointwiseEqual(const cudaReal *a, cudaReal *b, int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                b[i] = a[i];
            }
        }

        static __global__ void pointwiseTriple(cudaReal *result,
                                               const cudaReal *w,
                                               const cudaReal *q1,
                                               const cudaReal *q2,
                                               int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                result[i] += w[i] * q1[i] * q2[i];
            }
        }

        static __global__ void makedk(cudaReal *dk,
                                      const cudaReal *dksq,
                                      cudaReal *ksq,
                                      int n,
                                      int kSize,
                                      int rSize)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < kSize; i += nThreads)
            {
                if (i != 0)
                    dk[i + n * rSize] = 0.5 * dksq[i + n * rSize] / sqrt(ksq[i]);
                if (i == 0)
                    dk[i + n * rSize] = 0.0;
            }
        }

        namespace Discrete
        {

            using namespace Util;

            template <int D>
            Bond<D>::Bond()
                : meshPtr_(0),
                  kMeshDimensions_(0)
            {
                propagator(0).setBond(*this);
                propagator(1).setBond(*this);
            }

            template <int D>
            Bond<D>::~Bond()
            {
                delete[] expKsq_host;
                delete[] dexpKsq_host;
                delete[] dk_host;
                expKsq_.deallocate();
                dexpKsq_.deallocate();
                qr_.deallocate();
                qk_.deallocate();
                qk1_.deallocate();
                qsum_.deallocate();

                if (bondtype() == 1)
                {
                    cField().deallocate();
                    expW_.deallocate();
                    expWp_.deallocate();
                }
            }

            template <int D>
            void Bond<D>::setDiscretization(const Mesh<D> &mesh, const UnitCell<D> &unitCell)
            {
                UTIL_CHECK(mesh.size() > 1)

                meshPtr_ = &mesh;

                rSize_ = mesh.size();

                nParams_ = unitCell.nParameter();

                for (int i = 0; i < D; ++i)
                {
                    if (i < D - 1)
                    {
                        kMeshDimensions_[i] = mesh.dimensions()[i];
                    }
                    else
                    {
                        kMeshDimensions_[i] = mesh.dimensions()[i] / 2 + 1;
                    }
                }

                kSize_ = 1;

                for (int i = 0; i < D; ++i)
                {
                    kSize_ *= kMeshDimensions_[i];
                }

                expKsq_host = new cudaReal[kSize_];
                expKsq_.allocate(kMeshDimensions_);

                dexpKsq_host = new cudaReal[kSize_];
                dexpKsq_.allocate(kMeshDimensions_);
                dk_host = new cudaReal[nParams_ * kSize_];

                qr_.allocate(mesh.dimensions());
                qk_.allocate(mesh.dimensions());
                qk1_.allocate(mesh.dimensions());
                cudaMalloc((void **)&kSq_d, rSize_ * sizeof(cudaReal));
                qsum_.allocate(kSize_);

                if (bondtype() == 1)
                {
                    propagator(0).allocate(length(), mesh);
                    propagator(1).allocate(length(), mesh);

                    cField().allocate(mesh.dimensions());
                    expW_.allocate(mesh.dimensions());
                    expWp_.allocate(mesh.dimensions());
                }
                else
                {
                    propagator(0).allocate(2, mesh);
                    propagator(1).allocate(2, mesh);
                }
            }

            template <int D>
            void Bond<D>::setupUnitCell(const UnitCell<D> &unitCell,
                                        const WaveList<D> &waveList,
                                        const int N)
            {
                nParams_ = unitCell.nParameter();
                MeshIterator<D> iter;
                iter.setDimensions(kMeshDimensions_);
                IntVec<D> G;
                double Gsq;
#if (CHN == 1)
                double factor = kuhn() * sqrt(6.0 / double(N));
                // double factor = kuhn();
#endif
#if (CHN == 2)
                double factor = -kuhn() * kuhn() / double(N);
#endif
                int i;
                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    i = iter.rank();
                    G = waveList.minImage(i); 
                    Gsq = unitCell.ksq(G);
                    // std::cout << Gsq << std::endl;
                    
#if (CHN == 1)
                    double B = factor * sqrt(Gsq);
                    if (Gsq != 0)
                    {
                        expKsq_host[i] = sin(B) / B;
                        dexpKsq_host[i] = 0.5 * factor * (B * cos(B) - sin(B)) / (B * B * sqrt(Gsq));
                        // std::cout << expKsq_host[i] << "\n";
                    }
                    else
                    {
                        expKsq_host[i] = 1.0;
                        dexpKsq_host[i] = 0.0;
                    }
#endif
#if (CHN == 2)
                    expKsq_host[i] = exp(factor * Gsq);
                    // std::cout << expKsq_host[i] << "\n";
                    if (Gsq != 0)
                        dexpKsq_host[i] = factor * exp(factor * Gsq);
                    else
                        dexpKsq_host[i] = 0.0;
#endif
                }
                // exit(1);
                cudaMemcpy(expKsq_.cDField(),
                           expKsq_host,
                           kSize_ * sizeof(cudaReal),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(dexpKsq_.cDField(),
                           dexpKsq_host,
                           kSize_ * sizeof(cudaReal),
                           cudaMemcpyHostToDevice);
            }

            template <int D>
            void Bond<D>::setupSolver(Bond<D>::WField const &w, int N)
            {
                int nx = mesh().size();
                UTIL_CHECK(nx > 0)

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                double ds = 1.0 / double(N);
                // std::cout << "nx = " << nx << "\n";
                ExpW<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(expW_.cDField(), w.cDField(), nx, ds);
                ExpW<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(expWp_.cDField(), w.cDField(), nx, -ds);
            }

            template <int D>
            void Bond<D>::computeConcentration(double prefactor)
            {
                int nx = mesh().size();
                UTIL_CHECK(nx > 0)

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                int N = length();

                UTIL_CHECK(N > 0)

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), 0.0, nx);

                Pscf::Pspg::Discrete::DPropagator<D> const &p0 = propagator(0);
                Pscf::Pspg::Discrete::DPropagator<D> const &p1 = propagator(1);

                for (int s = 0; s < N; ++s)
                {
                    pointwiseTriple<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), expWp_.cDField(),
                                                                             p0.q(s), p1.q(N - 1 - s), nx);
                }

                scaleRealconst<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), (double)(prefactor), nx);
            }

            template <int D>
            void Bond<D>::setupFFT()
            {
                if (!fft_.isSetup())
                {
                    fft_.setup(qr_, qk_);
                }
            }

            template <int D>
            void Bond<D>::step(const cudaReal *q, cudaReal *qNew)
            {
                int nx = mesh().size();
                int nk = qk_.capacity();
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
                UTIL_CHECK(nx > 0)
                UTIL_CHECK(qr_.capacity() == nx)
                UTIL_CHECK(expKsq_.capacity() == nk)
                if (bondtype() == 1)
                {
                    pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, qr_.cDField(), nx);

                    fft_.forwardTransform(qr_, qk_);

                    scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq_.cDField(), kSize_);
        
                    fft_.inverseTransform(qk_, qr_);

                    scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), expW().cDField(), nx);

                    pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), qNew, nx);
                }
                else
                {
                    pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(q, qr_.cDField(), nx);

                    fft_.forwardTransform(qr_, qk_);

                    scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qk_.cDField(), expKsq_.cDField(), kSize_);

                    fft_.inverseTransform(qk_, qr_);

                    pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qr_.cDField(), qNew, nx);
                }
            }

            template <int D>
            void Bond<D>::computeStress(WaveList<D> &wavelist, double prefactor)
            {
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(rSize_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                int i;
                for (i = 0; i < 6; ++i)
                {
                    stress_[i] = 0.0;
                }

                if (bondtype())
                {
                    for (int n = 0; n < nParams_; ++n)
                    {
                        assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), 0.0, kSize_);
                        for (int s = 0; s < length() - 1; ++s)
                        {
                            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(propagator(0).q(s), qr_.cDField(), meshPtr_->size());
                            fft_.forwardTransform(qr_, qk_);
                            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(propagator(1).q(length() - s - 2), qr_.cDField(), meshPtr_->size());
                            fft_.forwardTransform(qr_, qk1_);
                            cudaComplexMulAdd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), qk_.cDField(), qk1_.cDField(), 1.0, kSize_);
                        }
                        scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), wavelist.dkSq() + n * rSize_, kSize_);
                        scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), dexpKsq_.cDField(), kSize_);

                        cudaReal *tmp, s = 0.0;
                        tmp = new cudaReal[kSize_];
                        cudaMemcpy(tmp, qsum_.cDField(), kSize_ * sizeof(cudaReal), cudaMemcpyDeviceToHost);
                        for (int j = 0; j < kSize_; ++j)
                            s += tmp[j];
                        stress_[n] = s * prefactor;
                        delete[] tmp;
                    }
                }
                else
                {
                    for (int n = 0; n < nParams_; ++n)
                    {
                        assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), 0.0, kSize_);
                        pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(propagator(0).q(0), qr_.cDField(), meshPtr_->size());
                        fft_.forwardTransform(qr_, qk_);
                        pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(propagator(1).q(0), qr_.cDField(), meshPtr_->size());
                        fft_.forwardTransform(qr_, qk1_);
                        cudaComplexMulAdd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), qk_.cDField(), qk1_.cDField(), 1.0, kSize_);
                        scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), wavelist.dkSq() + n * rSize_, kSize_);
                        scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qsum_.cDField(), dexpKsq_.cDField(), kSize_);

                        cudaReal *tmp, s = 0.0;
                        tmp = new cudaReal[kSize_];
                        cudaMemcpy(tmp, qsum_.cDField(), kSize_ * sizeof(cudaReal), cudaMemcpyDeviceToHost);
                        for (int j = 0; j < kSize_; ++j)
                            s += tmp[j];
                        stress_[n] = s * prefactor;
                        delete[] tmp;
                    }
                }
            }
        }
    }
}

#endif