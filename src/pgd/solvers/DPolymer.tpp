#ifndef D_POLYMER_TPP
#define D_POLYMER_TPP

#include "DPolymer.h"

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            template <int D>
            DPolymer<D>::DPolymer()
            {
                setClassName("DPolymer");
            }

            template <int D>
            void DPolymer<D>::setPhi(double phi)
            {
                // UTIL_CHECK(ensemble() == Species::Closed)
                UTIL_CHECK(phi >= 0.0)
                UTIL_CHECK(phi <= 1.0)
                phi_ = phi;
            }

            template <int D>
            void DPolymer<D>::setMu(double mu)
            {

                UTIL_CHECK(ensemble() == Species::Open)
                mu_ = mu;
            }

            template <int D>
            void DPolymer<D>::setupUnitCell(UnitCell<D> const &unitCell,
                                            const WaveList<D> &wavelist)
            {
                nParams_ = unitCell.nParameter();

                for (int j = 0; j < nBond(); ++j)
                {
                    bond(j).setupUnitCell(unitCell, wavelist, N());
                }
            }

            template <int D>
            void DPolymer<D>::compute(DArray<WField> const &wFields)
            {
                int monomerId;

                for (int j = 0; j < nBond(); ++j)
                {

                    if (bond(j).bondtype() == 1)
                    {
                        monomerId = bond(j).monomerId(0);
                        bond(j).setupSolver(wFields[monomerId], N());
                    }
                }

                solve();
            }

            template <int D>
            void DPolymer<D>::computeSegment(const Mesh<D> &mesh,
                                             DArray<WField> const &wFields)
            {
                int nx = mesh.size();
                int ns = N(); 
                int nb = nBond();
                double factor = 1.0/(nx * ns);
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                Pspg::ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                if (!sS_.isAllocated())
                    sS_.allocate(ns);
                if (!sB_.isAllocated())
                    sB_.allocate(nb);
                if (!sRho_.isAllocated())
                    sRho_.allocate(ns);
                for (int s = 0; s < ns; ++s)   
                {
                    if (!sRho_[s].isAllocated())
                        sRho_[s].allocate(mesh.dimensions());
                }      

                cudaReal *tmp;
                cudaMalloc((void **)&tmp, nx * sizeof(cudaReal));
                int s = 0;
                for (int b = 0; b < nb; ++b)
                {
                    if (bond(b).bondtype())
                    {
                        int nbs = bond(b).length();
                        for (int bs = 0; bs < nbs; ++bs)
                        {
                            assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (sRho_[s].cDField(), bond(b).expWp().cDField(), nx);
                            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (sRho_[s].cDField(), bond(b).propagator(0).q(bs), nx);
                            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (sRho_[s].cDField(), bond(b).propagator(1).q(nbs-1-bs), nx);
                            scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (sRho_[s].cDField(),1.0/Q(), nx);

                            ++s;
                        }
                    }
                }   

                s = 0;
                for (int b = 0; b < nb; ++b)
                {
                    sB_[b] = 0.0;
                    if (bond(b).bondtype())
                    {
                        int nbs = bond(b).length();
                        int m = bond(b).monomerId(0);
                        
                        for (int bs = 0; bs < nbs; ++bs)
                        {
                            sSHelper<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (tmp, sRho_[s].cDField(), wFields[m].cDField(), Q(), nx);
                            sS_[s] = factor*gpuSum(tmp, nx);
                            // std::cout 
                            // << "Entropy component of "
                            //           << "bond " << b << "  "
                            //           << "segment " << bs << "  "
                            //           <<  sS_[s] << "\n";
                            sB_[b] += sS_[s];
                            ++s;
                        }
                    }
                }              
                cudaFree (tmp);
                
            }

            template <int D>
            void DPolymer<D>::computeStress(WaveList<D> &wavelist)
            {
                double prefactor = 1.0 / Q();

                // Initialize stress_ to 0
                for (int i = 0; i < nParams_; ++i)
                {
                    stress_[i] = 0.0;
                }
                for (int i = 0; i < nBond(); ++i)
                {
                    bond(i).computeStress(wavelist, prefactor);

                    for (int j = 0; j < nParams_; ++j)
                    {
                        stress_[j] += bond(i).stress(j);
                    }
                }
            }

        }
    }
}

#endif // !D_POLYMER_TPP