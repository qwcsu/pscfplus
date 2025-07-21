#ifndef PGC_POLYMER_TPP
#define PGC_POLYMER_TPP

/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include "Polymer.h"
#include <pspg/math/GpuHeaders.h>

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            template <int D>
            Polymer<D>::Polymer()
            {
                setClassName("Polymer");
            }

            template <int D>
            Polymer<D>::~Polymer() = default;

            template <int D>
            void Polymer<D>::setPhi(double phi)
            {
                // UTIL_CHECK(ensemble() == Species::Closed)
                UTIL_CHECK(phi >= 0.0)
                UTIL_CHECK(phi <= 1.0)
                phi_ = phi;
            }

            template <int D>
            void Polymer<D>::setMu(double mu)
            {
                UTIL_CHECK(ensemble() == Species::Open)
                mu_ = mu;
            }

            /*
             * Set unit cell dimensions in all solvers.
             */
            template <int D>
            void Polymer<D>::setupUnitCell(UnitCell<D> const &unitCell, const WaveList<D> &wavelist)
            {
                nParams_ = unitCell.nParameter();
                for (int j = 0; j < nBlock(); ++j)
                {
                    block(j).setupUnitCell(unitCell, wavelist);
                }
            }

            /*
             * Compute solution to MDE and concentrations.
             */
            template <int D>
            void Polymer<D>::compute(DArray<WField> const &wFields)
            {
                // Setup solvers for all blocks
                int monomerId;
                for (int j = 0; j < nBlock(); ++j)
                {
                    monomerId = block(j).monomerId();
                    block(j).setupSolver(wFields[monomerId]);
                }
                
                solve();
            }

            /*
            template <int D>
            void Polymer<D>::computeJoint(const Mesh<D> &mesh)
            {

                int nx = mesh.size();
                int nJoint = 0; 
                int jointId = 0;
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                for (int vertexId = 0; vertexId < nVertex(); ++vertexId)
                {
                    if (vertex(vertexId).size() > 1)
                        ++nJoint;
                }

                joint_.allocate(nJoint);
                SJ_.allocate(nJoint);

                cudaReal *tmp;
                cudaMalloc((void **)&tmp, nx * sizeof(cudaReal));

                for (int vertexId = 0; vertexId < nVertex(); ++vertexId)
                {
                    if (vertex(vertexId).size() > 1)
                    {
                        joint_[jointId].setDiscretization(mesh.dimensions());
                        assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(joint_[jointId].jFieldRGrid().cDField(), 1.0, nx);
                        
                        for (int i = 0; i < vertex(vertexId).size(); ++i)
                        {
                            int blockid = vertex(vertexId).inPropagatorId(i)[0];
                            int dir = vertex(vertexId).inPropagatorId(i)[1];
                            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(joint_[jointId].jFieldRGrid().cDField(),
                                                                                         block(blockid).propagator(dir).qtail(), nx);
                        }
                        scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(joint_[jointId].jFieldRGrid().cDField(),
                                                                           1.0/Q(), nx);

                        sJHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp,
                                                                         joint_[jointId].jFieldRGrid().cDField(), 
                                                                         nx);

                        scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp,
                                                                           1.0/mesh.size(), nx);
                        SJ_[jointId] = -gpuSum(tmp, nx);
                        // std::cout << SJ_[jointId] << "\n";
                        ++jointId;
                    }
                }

                cudaFree(tmp);
            }
            */

            template <int D>
            void Polymer<D>::computeVertex(const Mesh<D> &mesh)
            {
                int nx = mesh.size();
                int nv = nVertex(); 
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                cudaReal *tmp;
                cudaMalloc((void **)&tmp, nx * sizeof(cudaReal));
                if (!vRho_.isAllocated())
                    vRho_.allocate(nv);
                if (!sV_.isAllocated())
                    sV_.allocate(nv);
                double factor;
                factor = phi()/double(nVertex()*length()*nx);

                for (int v = 0; v < nv; ++v)
                {
                    if (!vRho_[v].isAllocated())
                        vRho_[v].allocate(mesh.dimensions());
                    
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                    (vRho_[v].cDField(), 1.0, nx);
                    int vs = vertex(v).size();
                    
                    for (int i = 0; i < vs; ++i)
                    {
                        int blockId = vertex(v).inPropagatorId(i)[0];
                        int dir = vertex(v).inPropagatorId(i)[1];

                        if (block(blockId).propagator(dir).isAllocated())
                        {
                            if (block(blockId).propagator(dir).directionFlag()==0)
                            {
                                inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                (vRho_[v].cDField(), block(blockId).propagator(dir).qtail(), nx);
                            }
                            else
                            {
                                inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                (vRho_[v].cDField(), block(blockId).propagator(dir).qhead(), nx);
                            }
                            
                        }    
                        else
                        {
                            if (block(blockId).propagator(dir).directionFlag()==0)
                            {
                                inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                (vRho_[v].cDField(), block(blockId).propagator(dir).ref().qtail(), nx);
                            }
                            else
                            {
                                inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                                (vRho_[v].cDField(), block(blockId).propagator(dir).ref().qhead(), nx);
                            }
                        }        
                    }
                    
                    scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                    (vRho_[v].cDField(),1.0/Q(), nx);
                    
                    sVHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                    (tmp, vRho_[v].cDField(), nx);

                    #include <iostream>
                    #include <fstream>
                    #include <iomanip>

                    std::string fieldFileName = std::to_string(v);
                    std::ofstream file;
                    file.open(fieldFileName);
                    double *vrho;
                    vrho = new double [nx];
                    cudaMemcpy(vrho, vRho_[v].cDField(), sizeof(cudaReal) * nx, cudaMemcpyDeviceToHost);
                    // for (int i = 0; i < nx; ++i)
                    //     file << std::scientific << std::setprecision(8) << vrho[i] << "\n";
                    if (D == 3)
                    {
                        for (int z = 0; z < mesh.dimensions()[2]; ++z)
                        {
                            for (int y = 0; y < mesh.dimensions()[1]; ++y)
                            {
                                for (int x = 0; x < mesh.dimensions()[0]; ++x)
                                {
                                    file << std::scientific << std::setprecision(8) << vrho[z+y*mesh.dimensions()[2]+x*mesh.dimensions()[1]*mesh.dimensions()[2]] << "\n";
                                }
                            }
                        }
                    }
                    else if (D == 2)
                    {
                        for (int y = 0; y < mesh.dimensions()[1]; ++y)
                        {
                            for (int x = 0; x < mesh.dimensions()[0]; ++x)
                            {
                                file << std::scientific << std::setprecision(8) << vrho[y+x*mesh.dimensions()[1]] << "\n";
                            }
                        }
                    }
                    
                    delete [] vrho;
                    file.close();
                                                                           
                    sV_[v] = factor*gpuSum(tmp, nx);
                    
                }
                cudaFree(tmp);
            }

            /*
             * Compute stress from a polymer chain.
             */

            template <int D>
            void Polymer<D>::computeStress(WaveList<D> &wavelist)
            {
                double prefactor;

                // Initialize stress_ to 0
                for (int i = 0; i < nParams_; ++i)
                {
                    stress_[i] = 0.0;
                }

                for (int i = 0; i < nBlock(); ++i)
                {
                    prefactor = exp(mu_) / length();

                    block(i).computeStress(prefactor);

                    for (int j = 0; j < nParams_; ++j)
                    {
                        stress_[j] += block(i).stress(j);
                    }
                }
            }

        }
    }
}

#endif
