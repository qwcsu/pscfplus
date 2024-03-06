#ifndef D_MIXTURE_TPP
#define D_MIXTURE_TPP
// #define double float
/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include "DMixture.h"
#include <pspg/math/GpuHeaders.h>

#include <cmath>

namespace Pscf
{
    namespace Pspg
    {

        static __global__ void accumulateConc(cudaReal *result, double uniform, const cudaReal *cField, int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                result[i] += (double)uniform * cField[i];
            }
        }

        namespace Discrete
        {
            template <int D>
            DMixture<D>::DMixture()
            {
                setClassName("DMixture");
            }

            template <int D>
            void DMixture<D>::readParameters(std::istream &in)
            {
                DMixtureTmpl<Pscf::Pspg::Discrete::DPolymer<D>, Pscf::Pspg::Solvent<D>>::readParameters(in);

                read(in, "sigma", sigma_);
#if CMP == 1
                read(in, "kappa", kappa_);
#endif

                int total_nBond = 0;
                for (int i = 0; i < nPolymer(); ++i)
                {
                    total_nBond += polymer(i).nBond();
                }

                UTIL_CHECK(nMonomer() > 0)
                UTIL_CHECK(nPolymer() + nSolvent() > 0)
            }

            template <int D>
            void DMixture<D>::setInteraction(ChiInteraction &interaction)
            {
                interaction_ = &interaction;
            }

            template <int D>
            void DMixture<D>::setMesh(Mesh<D> const &mesh, UnitCell<D> &unitCell)
            {
                UTIL_CHECK(nMonomer() > 0)
                UTIL_CHECK(nPolymer() + nSolvent() > 0)

                meshPtr_ = &mesh;

                // Set discretization for all bonds
                for (int i = 0; i < nPolymer(); ++i)
                {
                    for (int j = 0; j < polymer(i).nBond(); ++j)
                    {
                        polymer(i).bond(j).setDiscretization(mesh, unitCell);
                    }
                }

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

                nParams_ = unitCell.nParameter();

                dbu0K_.allocate(kMeshDimensions_);

                cFieldsKGrid_.allocate(nMonomer());

                for (int i = 0; i < nMonomer(); ++i)
                {
                    cFieldsKGrid_[i].allocate(mesh.dimensions());
                }

                cudaMalloc((void **)&Chiphi_,
                           kSize_ * sizeof(cudaReal));
                cudaMalloc((void **)&Kappaphi1_,
                           kSize_ * sizeof(cudaComplex));
                cudaMalloc((void **)&Kappaphi2_,
                           kSize_ * sizeof(cudaComplex));
                cudaMalloc((void **)&mStress_,
                           kSize_ * sizeof(cudaReal));
            }

            template <int D>
            void DMixture<D>::setBasis(Basis<D> &basis)
            {
                basisPtr_ = &basis;
                int ns = basis.nStar();
                dbu0_.allocate(ns);
            }

            template <int D>
            void DMixture<D>::setU0(UnitCell<D> &unitCell, WaveList<D> &wavelist)
            {
                int ns = basisPtr_->nStar();
                bu0_.allocate(ns);
                int hkltmp[D];
                double q;

                bu0_host = new cudaReal[ns];
                // dbu0_host = new cudaReal[ns];

                for (int i = 0; i < ns; ++i)
                {
                    q = 0.0;
                    if (!basisPtr_->star(i).cancel)
                    {
                        for (int j = 0; j < D; ++j)
                        {
                            hkltmp[j] = basisPtr_->star(i).waveBz[j];
                            // std::cout<< basis.star(i).waveBz[j] << "  ";
                        }
                        // std::cout << std::endl;

                        IntVec<D> hkl(hkltmp);

                        q = unitCell.ksq(hkl);
                        q = sqrt(q);
                        q *= sigma_;
                        // std::cout<< q << std::endl;

                        if (q != 0)
                        {
#if NBP == 0
                            bu0_host[i] = exp(-0.5 * q * q);
#endif
#if NBP == 1
                            bu0_host[i] = 60.0 * (2 * q + q * cos(q) - 3 * sin(q)) / pow(q, 5.0);
#endif
#if NBP == 2
                            bu0_host[i] = 3 * (sin(q) - q * cos(q)) / (q * q * q);
#endif
                        }
                        else
                        {
#if NBP == 3 || NBP == 2 || NBP == 1 || NBP == 0
                            bu0_host[i] = 1.0;
#endif
                        }
                    }
                    else
                    {
#if NBP == 3 || NBP == 2 || NBP == 1 || NBP == 0
                        bu0_host[i] = 0.0;
#endif
                    }
                }

                cudaMemcpy(bu0_.cDField(), bu0_host, ns * sizeof(cudaReal), cudaMemcpyHostToDevice);

                delete[] bu0_host;
            }

            template <int D>
            void DMixture<D>::setKpN(double kpN)
            {
                kappa_ = kpN;
            }

            template <int D>
            void DMixture<D>::setupUnitCell(const UnitCell<D> &unitCell,
                                            const WaveList<D> &wavelist)
            {
                for (int i = 0; i < nPolymer(); ++i)
                {
                    polymer(i).setupUnitCell(unitCell, wavelist);
                }

                dbu0_host = new cudaReal[kSize_];
                MeshIterator<D> iter;
                iter.setDimensions(kMeshDimensions_);
                IntVec<D> G;
                double Gsq, q;
                int idx;

                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    idx = iter.rank();
                    G = wavelist.minImage(idx);
                    Gsq = unitCell.ksq(G);
                    if (Gsq != 0)
                    {
                        q = sigma() * sqrt(Gsq);
#if NBP == 0
                        dbu0_host[idx] = -0.5 * sigma() * sigma() * exp(-0.5 * q * q);
#endif

#if NBP == 1
                        dbu0_host[idx] = -30.0 * sigma() * sigma() * ((q * q - 15.0) * sin(q) + 8 * q + 7 * q * cos(q)) / (q * q * q * q * q * q * q);
#endif
#if NBP == 2
                        dbu0_host[idx] = 0.5 * sigma() * sigma() * (9 * q * cos(q) + 3 * (q * q - 3) * sin(q)) / (q * q * q * q * q);

#endif
                    }
                    else
                    {
#if NBP == 2 || NBP == 1 || NBP == 0
                        dbu0_host[idx] = 0.0;
#endif
                    }
                }
                cudaMemcpy(dbu0K_.cDField(), dbu0_host, kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);

                delete[] dbu0_host;
            }

            template <int D>
            void DMixture<D>::compute(DArray<WField> const &wFields,
                                      DArray<CField> &cFields)
            {
                UTIL_CHECK(meshPtr_)
                UTIL_CHECK(mesh().size() > 0)
                UTIL_CHECK(nMonomer() > 0)
                UTIL_CHECK(nPolymer() + nSolvent() > 0)
                UTIL_CHECK(wFields.capacity() == nMonomer())
                UTIL_CHECK(cFields.capacity() == nMonomer())

                int nx = mesh().size();
                int nm = nMonomer();
                int i, j;

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                for (i = 0; i < nm; ++i)
                {
                    UTIL_CHECK(cFields[i].capacity() == nx)
                    UTIL_CHECK(wFields[i].capacity() == nx)
                    // cFields[i][j] = 0.0;
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cFields[i].cDField(), 0.0, nx);
                }

                for (i = 0; i < nPolymer(); ++i)
                {
                    polymer(i).compute(wFields);
                }

                // Accumulate monomer concentration fields
                for (i = 0; i < nPolymer(); ++i)
                {
                    for (j = 0; j < polymer(i).nBond(); ++j)
                    {
                        if (polymer(i).bond(j).bondtype() == 1)
                        {
                            int monomerId = polymer(i).bond(j).monomerId(0);
                            UTIL_CHECK(monomerId >= 0)
                            UTIL_CHECK(monomerId < nm)
                            CField &monomerField = cFields[monomerId];
                            CField &bondField = polymer(i).bond(j).cField();
                            accumulateConc<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(monomerField.cDField(),
                                                                                    polymer(i).phi(),
                                                                                    bondField.cDField(),
                                                                                    nx);
                        }
                    }
                }

                for (int i = 0; i < nm; ++i)
                {
                    polymer(0).bond(0).fft().forwardTransform(cFields[i], cFieldsKGrid_[i]);
                }
            }

            template <int D>
            void DMixture<D>::computeStress(WaveList<D> &wavelist)
            {
                int i, j;
                int rSize = meshPtr_->size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(rSize, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                // Compute stress for each polymer.
                for (i = 0; i < nPolymer(); ++i)
                {
                    polymer(i).computeStress(wavelist);
                }

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(Chiphi_, 0.0, kSize_);
#if CMP == 1
                cudaMemset(Kappaphi1_, 0, kSize_ * sizeof(cudaComplex));
#endif
                for (i = 0; i < nMonomer(); ++i)
                {
                    for (j = 0; j < i; ++j)
                    {
                        if (i != j)
                            cudaComplexMulAdd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(Chiphi_,
                                                                                       cFieldsKGrid_[i].cDField(),
                                                                                       cFieldsKGrid_[j].cDField(),
                                                                                       interaction_->chi(i, j),
                                                                                       kSize_);
                    }
                }
#if CMP == 1
                for (i = 0; i < nMonomer(); ++i)
                {
                    cudaComplexAdd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(Kappaphi1_,
                                                                            cFieldsKGrid_[i].cDField(),
                                                                            kSize_);
                }
#endif
                for (int n = 0; n < nParams_; ++n)
                {
#if CMP == 1
                    mStressHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(mStress_,
                                                                           Chiphi_,
                                                                           Kappaphi1_,
                                                                           wavelist.dkSq(),
                                                                           dbu0K_.cDField(),
                                                                           0.5 * kappa_,
                                                                           n,
                                                                           kSize_,
                                                                           rSize);
#endif
#if CMP == 0
                    mStressHelperIncmp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(mStress_,
                                                                                Chiphi_,
                                                                                wavelist.dkSq(),
                                                                                dbu0K_.cDField(),
                                                                                n,
                                                                                kSize_,
                                                                                rSize);
#endif

                    cudaReal *tmp, ms = 0.0;
                    tmp = new cudaReal[kSize_];
                    cudaMemcpy(tmp, mStress_, kSize_ * sizeof(cudaReal), cudaMemcpyDeviceToHost);
                    for (j = 0; j < kSize_; ++j)
                        ms += tmp[j];
                    stress_[n] = ms;
                    delete[] tmp;
                }
                // exit(1);
                // Accumulate total stress
                for (i = 0; i < nParams_; ++i)
                {
                    for (j = 0; j < nPolymer(); ++j)
                    {
                        stress_[i] -= polymer(j).stress(i);
                    }
                }
            }

        }
    }
}

#endif
