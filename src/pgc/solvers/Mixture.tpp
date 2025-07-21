#ifndef PGC_MIXTURE_TPP
#define PGC_MIXTURE_TPP

/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include "Mixture.h"
#include <pspg/math/GpuHeaders.h>

#include <cmath>

namespace Pscf
{
    namespace Pspg
    {

        // theres a precision mismatch here. need to cast properly.
        static __global__ void accumulateConc(cudaReal *result, double uniform, const cudaReal *cField, int size)
        {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = startID; i < size; i += nThreads)
            {
                result[i] += (double)uniform * cField[i];
            }
        }

        namespace Continuous
        {

            template <int D>
            Mixture<D>::Mixture()
                : vMonomer_(1.0),
                  ds_(-1.0),
                  meshPtr_(0)
            {
                setClassName("Mixture");
            }

            template <int D>
            Mixture<D>::~Mixture() = default;

            template <int D>
            void Mixture<D>::readParameters(std::istream &in)
            {
                MixtureTmpl<Pscf::Pspg::Continuous::Polymer<D>,
                            Pscf::Pspg::Solvent<D>>::readParameters(in);
                vMonomer_ = 1.0; // Default value
                readOptional(in, "vMonomer", vMonomer_);
#if CMP == 1
                read(in, "N/kappa", kappa_);
#endif
                read(in, "sigma", sigma_);
                read(in, "ns", ns_);
                // read(in, "ds", ds_);
                ds_ = 1.0 / ns_;

                int total_nblock = 0;
                for (int i = 0; i < nPolymer(); ++i)
                {
                    total_nblock += polymer(i).nBlock();
                }
                // std::cout << "total nblock = " << total_nblock <<std::endl;

                UTIL_CHECK(nMonomer() > 0)
                UTIL_CHECK(nPolymer() + nSolvent() > 0)
                UTIL_CHECK(ds_ > 0)
            }

            template <int D>
            void Mixture<D>::setInteraction(ChiInteraction &interaction)
            {
                interaction_ = &interaction;
            }

            template <int D>
            void Mixture<D>::setMesh(Mesh<D> const &mesh,
                                     UnitCell<D> &unitCell)
            {
                UTIL_CHECK(nMonomer() > 0)
                UTIL_CHECK(nPolymer() + nSolvent() > 0)
                UTIL_CHECK(ds_ > 0)

                meshPtr_ = &mesh;

                // Set discretization for all blocks
                int i, j, k;
                for (i = 0; i < nPolymer(); ++i)
                {
                    for (j = 0; j < polymer(i).nBlock(); ++j)
                    {
                        polymer(i).block(j).setDiscretization(ds_, mesh);
                        
                    }
                }
                for (i = 0; i < nPolymer(); ++i)
                {
                    for (j = 0; j < polymer(i).nPropagator(); ++j)
                    {
                        int b = polymer(i).propagatorId(j)[0];
                        // int order = polymer(i).propagator(j).order();
                        if (polymer(i).mapping()[j].size() != 0)
                        {
                            polymer(i).propagator(j).allocate(polymer(i).block(b).ns(), mesh);
                            // std::cout << j << " is allocated on block " << b  <<"\n";
                            for (k = 1; k < polymer(i).mapping()[j].size(); ++k)
                            {
                                polymer(i).propagator(polymer(i).mapping()[j][k]).setPropagator(polymer(i).propagator(j), j);
                                // std::cout << polymer(i).mapping()[j][k] << " is refered to "
                                //           << j << "\n";
                            }
                        }        
                    }
                }
                // exit(1);
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

                rSize_ = meshPtr_->size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(rSize_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                nParams_ = unitCell.nParameter();
                cFieldsKGrid_.allocate(nMonomer());
#if DFT == 0
                bu0_.allocate(kMeshDimensions_);
                dbu0_.allocate(kSize_);

                for (int i = 0; i < nMonomer(); ++i)
                {
                    cFieldsKGrid_[i].allocate(mesh.dimensions());
                }

                cudaMalloc((void **)&Chiphi_,
                           kSize_ * sizeof(cudaReal));
                cudaMalloc((void **)&mStress_,
                           kSize_ * sizeof(cudaReal));
#endif
#if DFT == 1
                bu0_.allocate(meshPtr_->dimensions());
                dbu0_.allocate(meshPtr_->size());
                for (int i = 0; i < nMonomer(); ++i)
                {
                    cFieldsKGrid_[i].allocate(rSize_);
                }

                cudaMalloc((void **)&Chiphi_,
                           rSize_ * sizeof(cudaReal));
                cudaMalloc((void **)&mStress_,
                           rSize_ * sizeof(cudaReal));
#endif
                cudaMalloc((void **)&d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
                temp_ = new cudaReal[NUMBER_OF_BLOCKS];
            }

            template <int D>
            void Mixture<D>::setU0(UnitCell<D> &unitCell,
                                   WaveList<D> &wavelist)
            {
#if DFT == 0
                IntVec<D> G;
                double q;
                int idx;

                bu0_host = new cudaReal[kSize_];

                MeshIterator<D> iter;
                iter.setDimensions(kMeshDimensions_);
                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    idx = iter.rank();
                    G = wavelist.minImage(iter.rank());
                    q = sigma() * sqrt(unitCell.ksq(G));
                    if (q != 0)
                    {
#if NBP == 0
                        bu0_host[idx] = exp(-q * q / 2.0);
#endif
#if NBP == 1
                        bu0_host[idx] = 60.0 * (2 * q + q * cos(q) - 3 * sin(q)) / pow(q, 5.0);
#endif
#if NBP == 2
                        bu0_host[idx] = 3 * (sin(q) - q * cos(q)) / (q * q * q);
#endif
                    }
                    else
                    {
#if NBP == 3 || NBP == 2 || NBP == 1 || NBP == 0
                        bu0_host[idx] = 1.0;
#endif
                    }
                }

                cudaMemcpy(bu0_.cDField(), bu0_host, kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);

                delete[] bu0_host;
#endif
#if DFT == 1
                IntVec<D> G;
                double q;
                int idx;

                bu0_host = new cudaReal[rSize_];

                MeshIterator<D> iter;
                iter.setDimensions(meshPtr_->dimensions());
                // std::cout << meshPtr_->dimensions() << "\n";
                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    idx = iter.rank();
                    G = iter.position();
                    q = sigma() * sqrt(unitCell.ksq(G));
                    if (q != 0)
                    {
#if NBP == 0
                        bu0_host[idx] = exp(-q * q / 2.0);
#endif
#if NBP == 1
                        bu0_host[idx] = 60.0 * (2 * q + q * cos(q) - 3 * sin(q)) / pow(q, 5.0);
#endif
#if NBP == 2
                        bu0_host[idx] = 3 * (sin(q) - q * cos(q)) / (q * q * q);
#endif
                    }
                    else
                    {
#if NBP == 3 || NBP == 2 || NBP == 1 || NBP == 0
                        bu0_host[idx] = 1.0;
#endif
                    }
                }
                cudaMemcpy(bu0_.cDField(), bu0_host, rSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);

                delete[] bu0_host;
#endif
            }

            template <int D>
            void Mixture<D>::setupUnitCell(const UnitCell<D> &unitCell,
                                           const WaveList<D> &wavelist)
            {
#if DFT == 0
                nParams_ = unitCell.nParameter();
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
                    G = wavelist.minImage(iter.rank());
                    Gsq = unitCell.ksq(G);
                    if (Gsq != 0)
                    {
                        q = sigma() * sqrt(Gsq);
#if NBP == 0
                        dbu0_host[idx] = -0.5 * sigma() * sigma() * exp(-q * q / 2.0);
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
                cudaMemcpy(dbu0_.cDField(), dbu0_host, kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);

                delete[] dbu0_host;
#endif
#if DFT == 1
                nParams_ = unitCell.nParameter();
                for (int i = 0; i < nPolymer(); ++i)
                {
                    polymer(i).setupUnitCell(unitCell, wavelist);
                }
                dbu0_host = new cudaReal[rSize_];
                MeshIterator<D> iter;
                iter.setDimensions(meshPtr_->dimensions());
                IntVec<D> G;
                double Gsq, q;
                int idx;

                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    idx = iter.rank();
                    G = iter.position();
                    Gsq = unitCell.ksq(G);
                    if (Gsq != 0)
                    {
                        q = sigma() * sqrt(Gsq);
#if NBP == 0
                        dbu0_host[idx] = -0.5 * sigma() * sigma() * exp(-q * q / 2.0);
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
                cudaMemcpy(dbu0_.cDField(), dbu0_host, kSize_ * sizeof(cudaReal), cudaMemcpyHostToDevice);

                delete[] dbu0_host;
#endif
            }

            /*
             * Compute concentrations (but not total free energy).
             */
            template <int D>
            void Mixture<D>::compute(DArray<Mixture<D>::WField> const &wFields,
                                     DArray<Mixture<D>::CField> &cFields)
            {
                UTIL_CHECK(meshPtr_)
                UTIL_CHECK(mesh().size() > 0)
                UTIL_CHECK(nMonomer() > 0)
                UTIL_CHECK(nPolymer() + nSolvent() > 0)
                UTIL_CHECK(wFields.capacity() == nMonomer())
                UTIL_CHECK(cFields.capacity() == nMonomer())

                int nx = 1;
                for (int d = 0; d < D; ++d)
                    nx *= mesh().dimension(d);

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                int nm = nMonomer();
                int i, j;

                // Clear all monomer concentration fields
                for (i = 0; i < nm; ++i)
                {
                    // std::cout << "nx = " << nx << std::endl;
                    // std::cout << "wsize =  " << wFields[i].capacity() << std::endl;
                    UTIL_CHECK(cFields[i].capacity() == nx)
                    UTIL_CHECK(wFields[i].capacity() == nx)
                    // cFields[i][j] = 0.0;
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cFields[i].cDField(), 0.0, nx);
                }

                // Solve MDE for all polymers
                for (i = 0; i < nPolymer(); ++i)
                {
                    polymer(i).compute(wFields);
                }
                double phi_tot = 0.0;
                for (i = 0; i < nPolymer(); ++i)
                {
                    phi_tot += polymer(i).phi();
                }
                for (i = 0; i < nPolymer(); ++i)
                {
                   polymer(i).setPhi(polymer(i).phi()/phi_tot);
                }
                // Accumulate monomer concentration fields                
                for (i = 0; i < nPolymer(); ++i)
                {
                    for (j = 0; j < polymer(i).nBlock(); ++j)
                    {
                        int monomerId = polymer(i).block(j).monomerId();
                        UTIL_CHECK(monomerId >= 0)
                        UTIL_CHECK(monomerId < nm)
                        CField &monomerField = cFields[monomerId];
                        CField &blockField = polymer(i).block(j).cField();
                        // monomerField[k] += polymer(i).phi() * blockField[k];
                        accumulateConc<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(monomerField.cDField(),
                                                                                polymer(i).phi()/polymer(i).length(), 
                                                                                blockField.cDField(), nx);
                    }
                }
                // cudaReal a[32];
                // cudaMemcpy(a, polymer(0).block(0).cField().cDField(), sizeof(cudaReal)*32, cudaMemcpyDeviceToHost);
                // for(int i = 0; i < 32; ++i)
                //     std::cout << a[i] << "\n";
                // cudaFree(a);
                // std::cout << "\n";
                
                // #include <fstream>
                // #include <iomanip>
                // std::ofstream file1("p1");
                // cudaReal *a1, *b1, *a2, *b2;
                // a1 = new cudaReal [nx];
                // b1 = new cudaReal [nx];
                // a2 = new cudaReal [nx];
                // b2 = new cudaReal [nx];
                // cudaMemcpy(a1, polymer(0).block(0).cField().cDField(), sizeof(cudaReal)*nx, cudaMemcpyDeviceToHost);
                // cudaMemcpy(b1, polymer(0).block(1).cField().cDField(), sizeof(cudaReal)*nx, cudaMemcpyDeviceToHost);
                // cudaMemcpy(a2, polymer(1).block(0).cField().cDField(), sizeof(cudaReal)*nx, cudaMemcpyDeviceToHost);
                // cudaMemcpy(b2, polymer(1).block(1).cField().cDField(), sizeof(cudaReal)*nx, cudaMemcpyDeviceToHost);
    
                // for (int z = 0; z < mesh().dimension(2); ++z)
                // {
                //     for (int y = 0; y < mesh().dimension(1); ++y)
                //     {
                //         for (int x = 0; x < mesh().dimension(0); ++x)
                //         {
                //             file1 << std::setw(25) << std::scientific << std::setprecision(12) 
                //                   << a1[z+y*mesh().dimension(2)+x*mesh().dimension(2)*mesh().dimension(1)]*polymer(0).phi()/polymer(0).length();
                //             file1 << std::setw(25) << std::scientific << std::setprecision(12) 
                //                   << b1[z+y*mesh().dimension(2)+x*mesh().dimension(2)*mesh().dimension(1)]*polymer(0).phi()/polymer(0).length();
                //             file1 << "\n";
                //         }
                //     }
                // }
                // file1.close();

                // std::ofstream file2("p2");
    
                // for (int z = 0; z < mesh().dimension(2); ++z)
                // {
                //     for (int y = 0; y < mesh().dimension(1); ++y)
                //     {
                //         for (int x = 0; x < mesh().dimension(0); ++x)
                //         {
                //             file2 << std::setw(25) << std::scientific << std::setprecision(12) 
                //                   << a2[z+y*mesh().dimension(2)+x*mesh().dimension(2)*mesh().dimension(1)]*polymer(1).phi()/polymer(1).length();
                //             file2 << std::setw(25) << std::scientific << std::setprecision(12) 
                //                   << b2[z+y*mesh().dimension(2)+x*mesh().dimension(2)*mesh().dimension(1)]*polymer(1).phi()/polymer(1).length();
                //             file2 << "\n";
                //         }
                //     }
                // }
                // file2.close();
                // exit(1);

                for (int i = 0; i < nm; ++i)
                {
#if DFT == 0
                    polymer(0).block(0).fft().forwardTransform(cFields[i], cFieldsKGrid_[i]);
#endif
                }
            }

            /*
             * Compute Total Stress.
             */
            template <int D>
            void Mixture<D>::computeStress(WaveList<D> &wavelist)
            {
                int i, j;

                // Compute stress for each polymer.
                for (i = 0; i < nPolymer(); ++i)
                {
                    polymer(i).computeStress(wavelist);
                }
                for (i = 0; i < nParams_; ++i)
                {
                    stress_[i] = 0;
                }
#if DFT == 0
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(rSize_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(Chiphi_, 0.0, kSize_);

                for (i = 0; i < nMonomer(); ++i)
                {
                    for (j = 0; j < i; ++j)
                    {
                        if (i != j)
                            cudaComplexMulAdd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(Chiphi_,
                                                                                       cFieldsKGrid_[i].cDField(),
                                                                                       cFieldsKGrid_[j].cDField(),
                                                                                       0.5 * interaction_->chi(i, j),
                                                                                       kSize_);
                    }
                }

                for (int n = 0; n < nParams_; ++n)
                {
                    mStressHelperIncmp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(mStress_,
                                                                                Chiphi_,
                                                                                wavelist.dkSq(),
                                                                                dbu0_.cDField(),
                                                                                n,
                                                                                kSize_,
                                                                                rSize_);

                    cudaReal *tmp, ms = 0.0;
                    tmp = new cudaReal[kSize_];
                    cudaMemcpy(tmp, mStress_, kSize_ * sizeof(cudaReal), cudaMemcpyDeviceToHost);
                    for (j = 0; j < kSize_; ++j)
                        ms += tmp[j];
                    stress_[n] = ms;
                    delete[] tmp;
                }
#endif
                // Accumulate total stress
                for (i = 0; i < nParams_; ++i)
                {
                    for (j = 0; j < nPolymer(); ++j)
                    {
                        stress_[i] += polymer(j).stress(i);
                    }
                }
            }

            template <int D>
            void Mixture<D>::computeBlockEntropy(DArray<RDField<D>> const &wFields,
                                                 Mesh<D> const &mesh)
            {
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(rSize_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                int np = nPolymer();
                int nbTotal = 0;
                for (int p = 0; p < np; ++p)
                {
                    int nb = polymer(p).nBlock();
                    nbTotal += nb;
                }
                
                if (!sBlock_.isAllocated())
                    sBlock_.allocate(nbTotal);

                cudaReal *tmp;
                cudaMalloc ((void **)&tmp, rSize_ * sizeof(cudaReal));

                int idx = 0;
                for (int p = 0; p < np; ++p)
                {
                    int nb = polymer(p).nBlock();
                    double factor;
                    factor = polymer(p).phi()/double(polymer(p).nVertex()*polymer(p).length()*mesh.size());
                    for (int b = 0; b < nb; ++b)
                    {
                        int monomerId = polymer(p).block(b).monomerId();
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                        (tmp, wFields[monomerId].cDField(), rSize_);
                        inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                        (tmp, polymer(p).block(b).cField().cDField(), rSize_);
                        sBlock_[idx] = -(polymer(p).phi()/polymer(p).length()/mesh.size())*gpuSum(tmp, rSize_);

                        int vId0 = polymer(p).block(b).vertexId(0);
                        int vs0 = polymer(p).vertex(vId0).size();
                        for (int s = 0; s < vs0; ++s)
                        {
                            if (polymer(p).vertex(vId0).inPropagatorId(s)[0] == b)
                            {
                                int dir = polymer(p).vertex(vId0).inPropagatorId(s)[1];
                                if (polymer(p).block(b).propagator(dir).isAllocated())
                                {
                                    if (polymer(p).block(b).propagator(dir).directionFlag()==0)
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                         polymer(p).vertexRho(vId0).cDField(), 
                                         polymer(p).block(b).propagator(dir).qtail(), 
                                         rSize_);
                                    }
                                    else
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                         polymer(p).vertexRho(vId0).cDField(), 
                                         polymer(p).block(b).propagator(dir).qhead(), 
                                         rSize_);
                                    }
                                }
                                else
                                {   
                                    if (polymer(p).block(b).propagator(dir).directionFlag()==0)
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                         polymer(p).vertexRho(vId0).cDField(), 
                                         polymer(p).block(b).propagator(dir).ref().qtail(), 
                                         rSize_);
                                    }
                                    else
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                         polymer(p).vertexRho(vId0).cDField(), 
                                         polymer(p).block(b).propagator(dir).ref().qhead(), 
                                         rSize_);
                                    }
                                }
                                sBlock_[idx] -= factor*gpuSum(tmp, rSize_);
                                break;
                            }
                        }

                        int vId1 = polymer(p).block(b).vertexId(1);
                        int vs1 = polymer(p).vertex(vId1).size();
                        for (int s = 0; s < vs1; ++s)
                        {
                            if (polymer(p).vertex(vId1).inPropagatorId(s)[0] == b)
                            {
                                int dir = polymer(p).vertex(vId1).inPropagatorId(s)[1];
                                if (polymer(p).block(b).propagator(dir).isAllocated())
                                {
                                    if (polymer(p).block(b).propagator(dir).directionFlag()==0)
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                        polymer(p).vertexRho(vId1).cDField(), 
                                        polymer(p).block(b).propagator(dir).qtail(), 
                                        rSize_);
                                    }
                                    else
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                        polymer(p).vertexRho(vId1).cDField(), 
                                        polymer(p).block(b).propagator(dir).qhead(), 
                                        rSize_);
                                    }
                                }
                                else
                                {
                                    if (polymer(p).block(b).propagator(dir).directionFlag()==0)
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                        polymer(p).vertexRho(vId1).cDField(), 
                                        polymer(p).block(b).propagator(dir).ref().qtail(), 
                                        rSize_);
                                    }
                                    else
                                    {
                                        sBlockHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                        (tmp, 
                                        polymer(p).vertexRho(vId1).cDField(), 
                                        polymer(p).block(b).propagator(dir).ref().qhead(), 
                                        rSize_);
                                    }
                                }
                                sBlock_[idx] -= factor*gpuSum(tmp, rSize_);
                                break;
                            }
                        }
                        ++idx;
                    }
                }

                cudaFree(tmp);
            }
#if CMP==1
            template <int D>
            void Mixture<D>::computeBlockCMP(Mesh<D> const &mesh, 
                                             DArray<CField> cFieldsRGrid, 
                                             DArray<RDFieldDft<D>> cFieldsKGrid,
                                             FFT<D> & fft)
            {
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(rSize_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                RDField<D> tmp;
                RDFieldDft<D> tmpDft;
                tmp.allocate(mesh.dimensions());
                tmpDft.allocate(mesh.dimensions());
                DArray<int> list;
                list.allocate(4);

                int np = nPolymer();
                int i = 0;
                
                for (int p1 = 0; p1 < np; ++p1)
                {
                    for (int p2 = p1; p2 < np; ++p2)
                    {
                        int nb1 = polymer(p1).nBlock();
                        int nb2 = polymer(p2).nBlock();
                        for (int b1 = 0; b1 < nb1; ++b1)
                        {
                            for (int b2 = 0; b2 < nb2; ++b2)
                            {   
                                int m1 = polymer(p1).block(b1).monomerId();;
                                int m2 = polymer(p2).block(b2).monomerId();;
                                list[0] = p1;
                                list[1] = b1;
                                list[2] = p2;
                                list[3] = b2;


                                double factor = polymer(p1).phi()/polymer(p1).length()
                                              * polymer(p2).phi()/polymer(p2).length()
                                              * kpN()
                                              * 0.5
                                              / mesh.size();
                                cudaConv<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmpDft.cDField(),
                                                                                  cFieldsKGrid[m1].cDField(),
                                                                                  bu0().cDField(),
                                                                                  kSize_);
                                fft.inverseTransform(tmpDft, tmp);
                                inPlacePointwiseMul1<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp.cDField(),
                                                                                              cFieldsRGrid[m2].cDField(),
                                                                                              polymer(p2).phi()*polymer(p2).block(b2).length(),
                                                                                              rSize_);
                                blockIds_.append(list);
                                uBlockCMP_.append(factor*gpuSum(tmp.cDField(), rSize_));
                                ++i;
                            }
                        }
                    }
                }
                nUCompCMP_ = i;
                list.deallocate();
                tmp.deallocate();
                tmpDft.deallocate();
            }
#endif
            template <int D>
            void Mixture<D>::computeBlockRepulsion(Mesh<D> const &mesh, 
                                                   DArray<CField> cFieldsRGrid, 
                                                   DArray<RDFieldDft<D>> cFieldsKGrid,
                                                   FFT<D> & fft)
            {
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(rSize_, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                RDField<D> tmp;
                RDFieldDft<D> tmpDft;
                tmp.allocate(mesh.dimensions());
                tmpDft.allocate(mesh.dimensions());
                DArray<int> list;
                list.allocate(4);

                int np = nPolymer();
                int i = 0;
                
                for (int p1 = 0; p1 < np; ++p1)
                {
                    for (int p2 = p1; p2 < np; ++p2)
                    {
                        int nb1 = polymer(p1).nBlock();
                        int nb2 = polymer(p2).nBlock();
                        for (int b1 = 0; b1 < nb1; ++b1)
                        {
                            for (int b2 = 0; b2 < nb2; ++b2)
                            {   
                                if (p1 == p2 && b1 == b2)
                                {
                                    break;
                                }
                                else
                                {
                                    double chi;
                                    int m1 = polymer(p1).block(b1).monomerId();
                                    int m2 = polymer(p2).block(b2).monomerId();
                                    chi = interaction_->chi(m1, m2);

                                    if (chi != 0.0)
                                    {
                                        list[0] = p1;
                                        list[1] = b1;
                                        list[2] = p2;
                                        list[3] = b2;


                                        double factor = polymer(p1).phi()/polymer(p1).length()
                                                      * polymer(p2).phi()/polymer(p2).length()
                                                      * chi 
                                                      / mesh.size();
                                        
                                        cudaConv<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmpDft.cDField(),
                                                                                          cFieldsKGrid[m1].cDField(),
                                                                                          bu0().cDField(),
                                                                                          kSize_);
                                        fft.inverseTransform(tmpDft, tmp);
                                        inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp.cDField(),
                                                                                                     cFieldsRGrid[m2].cDField(),
                                                                                                     rSize_);
                                        blockIds_.append(list);
                                        uBlockChi_.append(factor*gpuSum(tmp.cDField(), rSize_));
                                        ++i;
                                    }
                                }
                            }
                        }
                    }
                }
                nUCompChi_ = i;
                list.deallocate();
                tmp.deallocate();
                tmpDft.deallocate();
            }
        }
    } // namespace Pspg
} // namespace Pscf
#endif
