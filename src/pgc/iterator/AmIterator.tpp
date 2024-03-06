#ifndef PGC_AM_ITERATOR_TPP
#define PGC_AM_ITERATOR_TPP
/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include "AmIterator.h"
#include <pgc/System.h>
#include <util/format/Dbl.h>
#include <util/containers/FArray.h>
#include <util/misc/Timer.h>
#include <ctime>

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            using namespace Util;

            template <int D>
            AmIterator<D>::AmIterator()
                : Iterator<D>(),
                  epsilon_(0),
                  lambda_(0),
                  nHist_(0),
                  maxHist_(0),
                  isFlexible_(0)
            {
                setClassName("AmIterator");
            }

            template <int D>
            AmIterator<D>::AmIterator(System<D> *system)
                : Iterator<D>(system),
                  epsilon_(0),
                  lambda_(0),
                  nHist_(0),
                  maxHist_(0),
                  isFlexible_(0)
            {
                setClassName("AmIterator");
            }

            template <int D>
            AmIterator<D>::~AmIterator()
            {
                delete[] temp_;
                cudaFree(d_temp_);
            }

            template <int D>
            void AmIterator<D>::readParameters(std::istream &in)
            {
                isFlexible_ = 0; // default value (fixed cell)
                read(in, "maxItr", maxItr_);
                read(in, "epsilon", epsilon_);
                read(in, "maxHist", maxHist_);
                readOptional(in, "isFlexible", isFlexible_);
            }

            template <int D>
            void AmIterator<D>::allocate()
            {
                devHists_.allocate(maxHist_ + 1);
                omHists_.allocate(maxHist_ + 1);
                histMat_.allocate(maxHist_ + 1);

                if (isFlexible_)
                {
                    devCpHists_.allocate(maxHist_ + 1);
                    CpHists_.allocate(maxHist_ + 1);
                }
                wArrays_.allocate(systemPtr_->mixture().nMonomer());
                dArrays_.allocate(systemPtr_->mixture().nMonomer());
                tempDev.allocate(systemPtr_->mixture().nMonomer());
#if DFT == 0
                int ns = systemPtr_->basis().nStar();

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    wArrays_[i].allocate(ns);
                    dArrays_[i].allocate(ns);
                    tempDev[i].allocate(ns);
                }
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(ns, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
                // std::cout <<  NUMBER_OF_BLOCKS << "\n";
                // std::cout <<  THREADS_PER_BLOCK << "\n";
                // exit(1);
#endif
#if DFT == 1
                int size = systemPtr_->mesh().size();
                tempDev_dct.allocate(systemPtr_->mixture().nMonomer());
                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    wArrays_[i].allocate(size);
                    dArrays_[i].allocate(size);
                    tempDev[i].allocate(size);
                    tempDev_dct[i].allocate(size);
                }
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
#endif

                // allocate d_temp_ here i suppose
                cudaMalloc((void **)&d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
                temp_ = new cudaReal[NUMBER_OF_BLOCKS];
            }

            template <int D>
            int AmIterator<D>::solve()
            {
                // Define Timer objects
                Timer solverTimer;
                Timer stressTimer;
                Timer updateTimer;
                Timer::TimePoint now;
                // Solve MDE for initial state
                solverTimer.start();
                systemPtr_->mixture().compute(systemPtr_->wFieldsRGridPh(),
                                              systemPtr_->cFieldsRGrid());
                // cudaReal a[512];
                // cudaMemcpy(a, systemPtr_->cFieldRGrid(1).cDField(), sizeof(cudaReal)*512, cudaMemcpyDeviceToHost);
                // for(int i = 0; i < 512; ++i)
                //     std::cout << a[i] << "\n";
                // cudaFree(a);
                // std::cout << "\n";
                // exit(1);

#if DFT == 0
                systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                          systemPtr_->cFields());
                systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->wFieldsRGridPh(),
                                                          systemPtr_->wFields());
#endif
                now = Timer::now();
                solverTimer.stop(now);

                // Compute stress for initial state
                if (isFlexible_)
                {
                    stressTimer.start(now);

                    systemPtr_->mixture().computeStress(systemPtr_->wavelist());

                    for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                    {
                        Log::file() << "Stress    " << m << " = "
                                    << std::setw(21) << std::setprecision(14)
                                    << systemPtr_->mixture().stress(m) << "\n";
                    }
                    for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                    {
                        Log::file() << "Parameter " << m << " = "
                                    << std::setw(21) << std::setprecision(14)
                                    << (systemPtr_->unitCell()).parameter(m) << "\n";
                    }
                    now = Timer::now();
                    stressTimer.stop(now);
                }

                // Anderson-Mixing iterative loop
                int itr;
                for (itr = 1; itr <= maxItr_; ++itr)
                {
                    updateTimer.start(now);
                    clock_t time1 = clock();
                    if (itr % 10 == 0 || itr == 1)
                    {
                        Log::file() << "-------------------------------------"
                                    << "\n";
                        Log::file() << "iteration #" << itr << "\n";
                    }
                    if (itr <= maxHist_)
                    {
                        lambda_ = 1.0 - pow(0.9, itr);
                        nHist_ = itr - 1;
                    }
                    else
                    {
                        lambda_ = 1.0;
                        nHist_ = maxHist_;
                    }

                    computeDeviation();

                    if (isConverged(itr))
                    {
                        updateTimer.stop();
                        if (itr > maxHist_ + 1)
                        {
                            invertMatrix_.deallocate();
                            coeffs_.deallocate();
                            vM_.deallocate();
                        }
                        Log::file() << "------- CONVERGED ---------" << std::endl;
                        // Output final timing results
                        double updateTime = updateTimer.time();
                        double solverTime = solverTimer.time();
                        double stressTime = 0.0;
                        double totalTime = updateTime + solverTime;
                        if (isFlexible_)
                        {
                            stressTime = stressTimer.time();
                            totalTime += stressTime;
                        }
                        Log::file() << "\n";
                        Log::file() << " * Error     = " << Dbl(error_) << '\n';
                        Log::file() << "\n\n";
                        Log::file() << "Iterator times contributions:\n";
                        Log::file() << "\n";
                        Log::file() << "solver time  = " << solverTime << " s,  "
                                    << solverTime / totalTime << "\n";
                        Log::file() << "stress time  = " << stressTime << " s,  "
                                    << stressTime / totalTime << "\n";
                        Log::file() << "update time  = " << updateTime << " s,  "
                                    << updateTime / totalTime << "\n";
                        Log::file() << "total time   = " << totalTime << " s  ";
                        Log::file() << "\n\n";
                        if (isFlexible_)
                        {
                            Log::file() << "\n";
                            Log::file() << "Final stress values:"
                                        << "\n";
                            for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                            {
                                Log::file() << "Stress    " << m << " = "
                                            << systemPtr_->mixture().stress(m) << "\n";
                            }
                            Log::file() << "\n";
                            Log::file() << "Final unit cell parameter values:"
                                        << "\n";
                            for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                            {
                                Log::file() << "Parameter " << m << " = " << std::setprecision(11)
                                            << (systemPtr_->unitCell()).parameter(m) << "\n";
                            }
                            Log::file() << "\n";
                        }
                        return 0;
                    }
                    else
                    {
                        // Resize history based matrix appropriately
                        // consider making these working space local
                        if (itr <= maxHist_ + 1)
                        {
                            if (nHist_ > 0)
                            {
                                invertMatrix_.allocate(nHist_, nHist_);
                                coeffs_.allocate(nHist_);
                                vM_.allocate(nHist_);
                            }
                        }
                        int status = minimizeCoeff(itr);

                        if (status == 1)
                        {
                            // abort the calculations and treat as failure (time out)
                            // perform some clean up stuff
                            invertMatrix_.deallocate();
                            coeffs_.deallocate();
                            vM_.deallocate();
                            return 1;
                        }

                        buildOmega(itr);

                        if (itr <= maxHist_)
                        {
                            if (nHist_ > 0)
                            {
                                invertMatrix_.deallocate();
                                coeffs_.deallocate();
                                vM_.deallocate();
                            }
                        }
                        now = Timer::now();
                        updateTimer.stop(now);

                        // Solve MDE
                        solverTimer.start(now);
#if DFT == 0
                        systemPtr_->fieldIo().convertBasisToRGrid(systemPtr_->wFields(),
                                                                  systemPtr_->wFieldsRGridPh());
#endif
                        systemPtr_->mixture().compute(systemPtr_->wFieldsRGridPh(),
                                                      systemPtr_->cFieldsRGrid());
#if DFT == 0
                        systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                                  systemPtr_->cFields());
#endif
                        now = Timer::now();
                        solverTimer.stop(now);

                        if (isFlexible_)
                        {
                            stressTimer.start(now);
                            systemPtr_->mixture().computeStress(systemPtr_->wavelist());
                            if (itr % 10 == 0 || itr == 1)
                            {
                                for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                                {
                                    Log::file() << "Stress    " << m << " = "
                                                << std::setw(21) << std::setprecision(14)
                                                << systemPtr_->mixture().stress(m) << "\n";
                                }
                                for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                                {
                                    Log::file() << "Parameter " << m << " = "
                                                << std::setw(21) << std::setprecision(14)
                                                << (systemPtr_->unitCell()).parameter(m) << "\n";
                                }
                            }
                            now = Timer::now();
                            stressTimer.stop(now);
                        }
                    }
                    clock_t time2 = clock();
                    // double t1 = ((double)(time2 - time1)) / CLOCKS_PER_SEC ;
                    // std::cout << "iteration time: " << t1 << "s" << std::endl;
                }

                if (itr > maxHist_ + 1)
                {
                    invertMatrix_.deallocate();
                    coeffs_.deallocate();
                    vM_.deallocate();
                }

                // Failure: Not converged after maxItr iterations.
                return 1;
            }

            template <int D>
            void AmIterator<D>::computeDeviation()
            {
#if CMP == 0
#if DFT == 0

                int ns = systemPtr_->basis().nStar();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(ns, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                omHists_.append(systemPtr_->wFields());

                if (isFlexible_)
                {
                    CpHists_.append(systemPtr_->unitCell().parameters());
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(), 0.0, ns);
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                    {
                        pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                    systemPtr_->cField(j).cDField(),
                                                                                    systemPtr_->mixture().bu0().cDField(),
                                                                                    systemPtr_->interaction().chi(i, j),
                                                                                    ns);
                    }
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                    {
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField() + 1,
                                                                                   systemPtr_->wField(j).cDField() + 1,
                                                                                   -systemPtr_->interaction().idemp(i, j),
                                                                                   ns - 1);
                    }
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                               systemPtr_->wField(i).cDField(),
                                                                               -1.0,
                                                                               1);
                }

#endif
#if DFT == 1

                int size = systemPtr_->mesh().size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                omHists_.append(systemPtr_->wFieldsRGridPh());
                if (isFlexible_)
                {
                    CpHists_.append(systemPtr_->unitCell().parameters());
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    systemPtr_->mixture().polymer(0).block(0).fct().forwardTransform(systemPtr_->cFieldRGrid(i).cDField());
                    systemPtr_->mixture().polymer(0).block(0).fct().forwardTransform(systemPtr_->wFieldRGridPh(i).cDField());
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(), 0.0, size);
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                    {
                        pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                    systemPtr_->cFieldRGrid(j).cDField(),
                                                                                    systemPtr_->mixture().bu0().cDField(),
                                                                                    systemPtr_->interaction().chi(i, j),
                                                                                    size);
                    }
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                    {
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField() + 1,
                                                                                   systemPtr_->wFieldRGridPh(j).cDField() + 1,
                                                                                   -systemPtr_->interaction().idemp(i, j),
                                                                                   size - 1);
                    }
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                               systemPtr_->wFieldRGridPh(i).cDField(),
                                                                               -1.0,
                                                                               1);
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    systemPtr_->mixture().polymer(0).block(0).fct().inverseTransform(systemPtr_->cFieldRGrid(i).cDField());
                    systemPtr_->mixture().polymer(0).block(0).fct().inverseTransform(systemPtr_->wFieldRGridPh(i).cDField());
                    systemPtr_->mixture().polymer(0).block(0).fct().inverseTransform(tempDev[i].cDField());
                }

#endif
#endif
#if CMP == 1
#if DFT == 0
                int nbs = systemPtr_->basis().nStar();
                int nm = systemPtr_->mixture().nMonomer();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nbs, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                omHists_.append(systemPtr_->wFields());

                if (isFlexible_)
                {
                    CpHists_.append(systemPtr_->unitCell().parameters());
                }
                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(), 0.0, nbs);
                }
                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                   systemPtr_->cField(j).cDField(),
                                                                                   systemPtr_->mixture().kpN(),
                                                                                   nbs);
                    }
                    pointWiseSubtractFloat<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                    systemPtr_->mixture().kpN(),
                                                                                    1);
                }
                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                   systemPtr_->cField(j).cDField(),
                                                                                   systemPtr_->interaction().chi(i, j),
                                                                                   nbs);
                    }
                }
                for (int i = 0; i < nm; ++i)
                {
                    pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                               systemPtr_->wField(i).cDField(),
                                                                               -1.0,
                                                                               nbs);
                }

#endif
#endif
                devHists_.append(tempDev);

                if (isFlexible_)
                {
                    FArray<double, 6> tempCp;
                    for (int i = 0; i < (systemPtr_->unitCell()).nParameter(); i++)
                    {
                        // format????
                        tempCp[i] = -10 * ((systemPtr_->mixture()).stress(i));
                    }
                    devCpHists_.append(tempCp);
                }
            }

            template <int D>
            bool AmIterator<D>::isConverged(int itr)
            {
#if DFT == 0
                int ns = systemPtr_->basis().nStar();

                double error;
                double maxErr = 0.0;
                // double *max_g, *max_c;
                // cudaMalloc((void**)&max_g, 32 * sizeof(double));
                // max_c = new double [32];
                // for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                // {
                //     searchMax<<<32, 32>>>(devHists_[0][i].cDField(), max_g, ns);
                //     cudaMemcpy(max_c, max_g, sizeof(double)*32, cudaMemcpyDeviceToHost);
                //     for (int j = 0; j < 32; ++j)
                //     {
                //         if (max_c[j] > maxErr)
                //             maxErr = max_c[j];
                //     }
                // }
                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                    maxErr = gpuMaxAbs(devHists_[0][i].cDField(), ns);
#endif
#if DFT == 1
                int size = systemPtr_->mesh().size();
                double error;
                double maxErr = 0.0;

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                    maxErr = gpuMaxAbs(devHists_[0][i].cDField(), size);
#endif

                if (isFlexible_)
                {
                    for (int i = 0; i < systemPtr_->unitCell().nParameter(); i++)
                    {
                        if (abs(devCpHists_[0][i]) > maxErr)
                            maxErr = abs(devCpHists_[0][i]);
                    }
                }

                error = maxErr;
                error_ = error;

                if (itr % 10 == 0 || itr == 1)
                {
                    // Log::file() << " dError :" << Dbl(dError) << '\n';
                    // Log::file() << " wError :" << Dbl(wError) << '\n';
                    Log::file() << "  Error :" << Dbl(error) << '\n';
                }

                if (error < epsilon_)
                {
                    final_error = error;
                    return true;
                }
                else
                {
                    return false;
                }
            }

            template <int D>
            int AmIterator<D>::minimizeCoeff(int itr)
            {
#if DFT == 0
                int ns = systemPtr_->basis().nStar();
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(ns, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
                if (itr == 1)
                {
                    // do nothing
                    histMat_.reset();
                    return 0;
                }
                else
                {
                    int nMonomer = systemPtr_->mixture().nMonomer();
                    int nParameter = systemPtr_->unitCell().nParameter();
                    double elm, elm_cp;
                    // clear last column and shift everything downwards if necessary
                    histMat_.clearColumn(nHist_);
                    // calculate the new values for d(k)d(k') matrix
                    for (int i = 0; i < nHist_; ++i)
                    {
                        for (int j = i; j < nHist_; ++j)
                        {
                            invertMatrix_(i, j) = 0;
                            for (int k = 0; k < nMonomer; ++k)
                            {
                                elm = 0;
                                RDField<D> temp1, temp2;
                                temp1.allocate(ns);
                                temp2.allocate(ns);
                                pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][k].cDField(), devHists_[i + 1][k].cDField(),
                                                                                                 temp1.cDField(), ns);
                                pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][k].cDField(), devHists_[j + 1][k].cDField(),
                                                                                                 temp2.cDField(), ns);
                                elm += gpuInnerProduct(temp1.cDField(), temp2.cDField(), ns);

                                temp1.deallocate();
                                temp2.deallocate();

                                invertMatrix_(i, j) += elm;
                            }
                            if (isFlexible_)
                            {
                                elm_cp = 0;
                                for (int m = 0; m < nParameter; ++m)
                                {
                                    elm_cp += ((devCpHists_[0][m] - devCpHists_[i + 1][m]) *
                                               (devCpHists_[0][m] - devCpHists_[j + 1][m]));
                                }
                                invertMatrix_(i, j) += elm_cp;
                            }
                            invertMatrix_(j, i) = invertMatrix_(i, j);
                        }
                        vM_[i] = 0;
                        for (int j = 0; j < nMonomer; ++j)
                        {
                            RDField<D> temp;
                            temp.allocate(ns);
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][j].cDField(), devHists_[i + 1][j].cDField(),
                                                                                             temp.cDField(), ns);
                            vM_[i] += gpuInnerProduct(temp.cDField(), devHists_[0][j].cDField(), ns);
                            temp.deallocate();
                        }
                        if (isFlexible_)
                        {
                            elm_cp = 0;
                            for (int m = 0; m < nParameter; ++m)
                            {
                                vM_[i] += ((devCpHists_[0][m] - devCpHists_[i + 1][m]) *
                                           (devCpHists_[0][m]));
                            }
                        }
                    }
                    if (itr == 2)
                    {
                        coeffs_[0] = vM_[0] / invertMatrix_(0, 0);
                        // std::cout << vM_[0] << "\n";
                        // std::cout << invertMatrix_(0, 0) << "\n";
                        // std::cout << coeffs_[0]  << "\n";
                    }
                    else
                    {
                        LuSolver solver;
                        solver.allocate(nHist_);
                        solver.computeLU(invertMatrix_);
                        /*
                        int status = solver.solve(vM_, coeffs_);
                        if (status) {
                           if (status == 1) {
                              //matrix is singular do something
                              return 1;
                           }
                           }*/
                        solver.solve(vM_, coeffs_);
                        // for the sake of simplicity during porting
                        // we leaves out checks for singular matrix here
                        //--GK 09 11 2019
                    }
                    return 0;
                }
#endif
#if DFT == 1
                int size = systemPtr_->mesh().size();
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
                if (itr == 1)
                {
                    // do nothing
                    histMat_.reset();
                    return 0;
                }
                else
                {
                    int nMonomer = systemPtr_->mixture().nMonomer();
                    int nParameter = systemPtr_->unitCell().nParameter();
                    double elm, elm_cp;
                    // clear last column and shift everything downwards if necessary
                    histMat_.clearColumn(nHist_);
                    // calculate the new values for d(k)d(k') matrix
                    for (int i = 0; i < nHist_; ++i)
                    {
                        for (int j = i; j < nHist_; ++j)
                        {
                            invertMatrix_(i, j) = 0;
                            for (int k = 0; k < nMonomer; ++k)
                            {
                                elm = 0;
                                RDField<D> temp1, temp2;
                                temp1.allocate(size);
                                temp2.allocate(size);
                                pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][k].cDField(), devHists_[i + 1][k].cDField(),
                                                                                                 temp1.cDField(), size);
                                pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][k].cDField(), devHists_[j + 1][k].cDField(),
                                                                                                 temp2.cDField(), size);
                                elm += gpuInnerProduct(temp1.cDField(), temp2.cDField(), size);

                                temp1.deallocate();
                                temp2.deallocate();

                                invertMatrix_(i, j) += elm;
                            }
                            if (isFlexible_)
                            {
                                elm_cp = 0;
                                for (int m = 0; m < nParameter; ++m)
                                {
                                    elm_cp += ((devCpHists_[0][m] - devCpHists_[i + 1][m]) *
                                               (devCpHists_[0][m] - devCpHists_[j + 1][m]));
                                }
                                invertMatrix_(i, j) += elm_cp;
                            }
                            invertMatrix_(j, i) = invertMatrix_(i, j);
                        }
                        vM_[i] = 0;
                        for (int j = 0; j < nMonomer; ++j)
                        {
                            RDField<D> temp;
                            temp.allocate(size);
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][j].cDField(), devHists_[i + 1][j].cDField(),
                                                                                             temp.cDField(), size);
                            vM_[i] += gpuInnerProduct(temp.cDField(), devHists_[0][j].cDField(), size);
                            temp.deallocate();
                        }
                        if (isFlexible_)
                        {
                            elm_cp = 0;
                            for (int m = 0; m < nParameter; ++m)
                            {
                                vM_[i] += ((devCpHists_[0][m] - devCpHists_[i + 1][m]) *
                                           (devCpHists_[0][m]));
                            }
                        }
                    }
                    if (itr == 2)
                    {
                        coeffs_[0] = vM_[0] / invertMatrix_(0, 0);
                        // std::cout << vM_[0] << "\n";
                        // std::cout << invertMatrix_(0, 0) << "\n";
                        // std::cout << coeffs_[0]  << "\n";
                    }
                    else
                    {
                        LuSolver solver;
                        solver.allocate(nHist_);
                        solver.computeLU(invertMatrix_);
                        /*
                        int status = solver.solve(vM_, coeffs_);
                        if (status) {
                           if (status == 1) {
                              //matrix is singular do something
                              return 1;
                           }
                           }*/
                        solver.solve(vM_, coeffs_);
                        // for the sake of simplicity during porting
                        // we leaves out checks for singular matrix here
                        //--GK 09 11 2100.0*019
                    }
                    return 0;
                }
#endif
            }

            template <int D>
            void AmIterator<D>::buildOmega(int itr)
            {
#if DFT == 0
                int ns = systemPtr_->basis().nStar();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(ns, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                if (itr == 1)
                {
                    for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                            omHists_[0][i].cDField(), ns);
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                                   devHists_[0][i].cDField(), lambda_, ns);
                    }

                    if (isFlexible_)
                    {
                        cellParameters_.clear();
                        for (int m = 0; m < (systemPtr_->unitCell()).nParameter(); ++m)
                        {
                            cellParameters_.append(CpHists_[0][m] + lambda_ * devCpHists_[0][m]);
                        }
                        systemPtr_->unitCell().setParameters(cellParameters_);
                        systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                        systemPtr_->mixture().setU0(systemPtr_->unitCell(),
                                                    systemPtr_->wavelist());
                        systemPtr_->mixture().setupUnitCell(systemPtr_->unitCell(), systemPtr_->wavelist());
                    }
                }
                else
                {
                    // should be strictly correct. coeffs_ is a vector of size 1 if itr ==2

                    for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wArrays_[j].cDField(), omHists_[0][j].cDField(), ns);
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dArrays_[j].cDField(), devHists_[0][j].cDField(), ns);
                    }

                    for (int i = 0; i < nHist_; ++i)
                    {
                        for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                        {
                            // wArrays
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(omHists_[i + 1][j].cDField(),
                                                                                             omHists_[0][j].cDField(),
                                                                                             tempDev[0].cDField(),
                                                                                             ns);

                            pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wArrays_[j].cDField(),
                                                                                       tempDev[0].cDField(), coeffs_[i],
                                                                                       ns);

                            // dArrays
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[i + 1][j].cDField(),
                                                                                             devHists_[0][j].cDField(), tempDev[0].cDField(),
                                                                                             ns);
                            pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dArrays_[j].cDField(),
                                                                                       tempDev[0].cDField(), coeffs_[i], ns);
                        }
                        // std::cout << coeffs_[i] << "\n";
                    }

                    for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                            wArrays_[i].cDField(), ns);
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                                   dArrays_[i].cDField(), lambda_, ns);
                    }

                    if (isFlexible_)
                    {

                        for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                        {
                            wCpArrays_[m] = CpHists_[0][m];
                            dCpArrays_[m] = devCpHists_[0][m];
                        }
                        for (int i = 0; i < nHist_; ++i)
                        {
                            for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                            {
                                wCpArrays_[m] += coeffs_[i] * (CpHists_[i + 1][m] -
                                                               CpHists_[0][m]);
                                dCpArrays_[m] += coeffs_[i] * (devCpHists_[i + 1][m] -
                                                               devCpHists_[0][m]);
                            }
                        }

                        cellParameters_.clear();
                        for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                        {
                            cellParameters_.append(wCpArrays_[m] + lambda_ * dCpArrays_[m]);
                        }

                        systemPtr_->unitCell().setParameters(cellParameters_);
                        systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                        systemPtr_->mixture().setU0(systemPtr_->unitCell(),
                                                    systemPtr_->wavelist());
                        systemPtr_->mixture().setupUnitCell(systemPtr_->unitCell(), systemPtr_->wavelist());
                    }
                }
#endif
#if DFT == 1
                int size = systemPtr_->mesh().size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(size, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                if (itr == 1)
                {
                    for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wFieldRGridPh(i).cDField(),
                                                                            omHists_[0][i].cDField(), size);
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wFieldRGridPh(i).cDField(),
                                                                                   devHists_[0][i].cDField(), lambda_, size);
                    }

                    if (isFlexible_)
                    {
                        cellParameters_.clear();
                        for (int m = 0; m < (systemPtr_->unitCell()).nParameter(); ++m)
                        {
                            cellParameters_.append(CpHists_[0][m] + lambda_ * devCpHists_[0][m]);
                        }
                        systemPtr_->unitCell().setParameters(cellParameters_);
                        systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                        systemPtr_->mixture().setU0(systemPtr_->unitCell(),
                                                    systemPtr_->wavelist());
                        systemPtr_->mixture().setupUnitCell(systemPtr_->unitCell(), systemPtr_->wavelist());
                    }
                }
                else
                {
                    // should be strictly correct. coeffs_ is a vector of size 1 if itr ==2

                    for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wArrays_[j].cDField(), omHists_[0][j].cDField(), size);
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dArrays_[j].cDField(), devHists_[0][j].cDField(), size);
                    }

                    for (int i = 0; i < nHist_; ++i)
                    {
                        for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j)
                        {
                            // wArrays
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(omHists_[i + 1][j].cDField(),
                                                                                             omHists_[0][j].cDField(),
                                                                                             tempDev[0].cDField(),
                                                                                             size);

                            pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wArrays_[j].cDField(),
                                                                                       tempDev[0].cDField(), coeffs_[i],
                                                                                       size);

                            // dArrays
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[i + 1][j].cDField(),
                                                                                             devHists_[0][j].cDField(), tempDev[0].cDField(),
                                                                                             size);
                            pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dArrays_[j].cDField(),
                                                                                       tempDev[0].cDField(), coeffs_[i], size);
                        }
                        // std::cout << coeffs_[i] << "\n";
                    }

                    for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wFieldRGridPh(i).cDField(),
                                                                            wArrays_[i].cDField(), size);
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wFieldRGridPh(i).cDField(),
                                                                                   dArrays_[i].cDField(), lambda_, size);
                    }

                    if (isFlexible_)
                    {

                        for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                        {
                            wCpArrays_[m] = CpHists_[0][m];
                            dCpArrays_[m] = devCpHists_[0][m];
                        }
                        for (int i = 0; i < nHist_; ++i)
                        {
                            for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                            {
                                wCpArrays_[m] += coeffs_[i] * (CpHists_[i + 1][m] -
                                                               CpHists_[0][m]);
                                dCpArrays_[m] += coeffs_[i] * (devCpHists_[i + 1][m] -
                                                               devCpHists_[0][m]);
                            }
                        }

                        cellParameters_.clear();
                        for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                        {
                            cellParameters_.append(wCpArrays_[m] + lambda_ * dCpArrays_[m]);
                        }

                        systemPtr_->unitCell().setParameters(cellParameters_);
                        systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                        systemPtr_->mixture().setU0(systemPtr_->unitCell(),
                                                    systemPtr_->wavelist());
                        systemPtr_->mixture().setupUnitCell(systemPtr_->unitCell(), systemPtr_->wavelist());
                    }
                }
#endif
            }

        }
    }
}
#endif
