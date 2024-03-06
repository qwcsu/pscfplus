#ifndef D_AM_ITERATOR_TPP
#define D_AM_ITERATOR_TPP
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
#include <pgd/System.h>
#include <pspg/math/GpuHeaders.h>
#include <util/format/Dbl.h>
#include <util/containers/FArray.h>
#include <util/misc/Timer.h>
#include <ctime>

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
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

                wArrays_.allocate(systemPtr_->dmixture().nMonomer());
                dArrays_.allocate(systemPtr_->dmixture().nMonomer());
                tempDev.allocate(systemPtr_->dmixture().nMonomer());

                for (int i = 0; i < systemPtr_->dmixture().nMonomer(); ++i)
                {
                    wArrays_[i].allocate(systemPtr_->basis().nStar());
                    dArrays_[i].allocate(systemPtr_->basis().nStar());
                    tempDev[i].allocate(systemPtr_->basis().nStar());
                }
            }

            template <int D>
            int AmIterator<D>::solve()
            {
                systemPtr_->dmixture().setupUnitCell(systemPtr_->unitCell(),
                                                     systemPtr_->wavelist());
                systemPtr_->dmixture().compute(systemPtr_->wFieldsRGrid(),
                                               systemPtr_->cFieldsRGrid());

                systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                          systemPtr_->cFields());
                systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->wFieldsRGrid(),
                                                          systemPtr_->wFields());

                if (isFlexible_)
                {
                    // exit(1);
                    systemPtr_->dmixture().computeStress(systemPtr_->wavelist());
                    for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                    {
                        Log::file() << "Stress    " << m << " = "
                                    << std::setw(21) << std::setprecision(14)
                                    << 100 * systemPtr_->dmixture().stress(m) << "\n";
                    }
                    for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                    {
                        Log::file() << "Parameter " << m << " = "
                                    << std::setw(21) << std::setprecision(14)
                                    << (systemPtr_->unitCell()).parameter(m) << "\n";
                    }
                }

                int itr;
                for (itr = 1; itr <= maxItr_; ++itr)
                {
                    if (itr % 50 == 0 || itr == 1)
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
                        lambda_ = 1.0; //- pow(0.1, maxHist_);
                        nHist_ = maxHist_;
                    }

                    computeDeviation();

                    if (isConverged(itr))
                    {
                        if (itr > maxHist_ + 1)
                        {
                            invertMatrix_.deallocate();
                            coeffs_.deallocate();
                            vM_.deallocate();
                        }

                        if (isFlexible_)
                        {
                            Log::file() << "\n";
                            Log::file() << "Final stress values:"
                                        << "\n";
                            for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                            {
                                Log::file() << "Stress    " << m << " = "
                                            << std::setw(21) << std::setprecision(14)
                                            << 100.0 * systemPtr_->dmixture().stress(m) << "\n";
                            }
                            Log::file() << "\n";
                            Log::file() << "Final unit cell parameter values:"
                                        << "\n";
                            for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                            {
                                Log::file() << "Parameter " << m << " = "
                                            << std::setw(21) << std::setprecision(14)
                                            << (systemPtr_->unitCell()).parameter(m) << "\n";
                            }
                            Log::file() << "\n";
                        }

                        return 0;
                    }
                    else
                    {
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
                            invertMatrix_.deallocate();
                            coeffs_.deallocate();
                            vM_.deallocate();
                            return 1;
                        }

                        buildOmega(itr);
                        // exit(1);
                        if (itr <= maxHist_)
                        {
                            if (nHist_ > 0)
                            {
                                invertMatrix_.deallocate();
                                coeffs_.deallocate();
                                vM_.deallocate();
                            }
                        }

                        systemPtr_->fieldIo().convertBasisToRGrid(systemPtr_->wFields(),
                                                                  systemPtr_->wFieldsRGrid());

                        systemPtr_->dmixture().compute(systemPtr_->wFieldsRGrid(),
                                                       systemPtr_->cFieldsRGrid());

                        systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                                  systemPtr_->cFields());

                        if (isFlexible_)
                        {
                            systemPtr_->dmixture().computeStress(systemPtr_->wavelist());
                            if (itr % 50 == 0 || itr == 1)
                            {
                                for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                                {
                                    Log::file() << "Stress    " << m << " = "
                                                << std::setw(21) << std::setprecision(14)
                                                << systemPtr_->dmixture().stress(m) << "\n";
                                }
                                for (int m = 0; m < systemPtr_->unitCell().nParameter(); ++m)
                                {
                                    Log::file() << "Parameter " << m << " = "
                                                << std::setw(21) << std::setprecision(14)
                                                << (systemPtr_->unitCell()).parameter(m) << "\n";
                                }
                            }
                        }

                        // double c[systemPtr_->mesh().size()];
                        // cudaMemcpy(c, systemPtr_->cFieldRGrid(0).cDField(),
                        //           sizeof(cudaReal)*systemPtr_->mesh().size(),
                        //           cudaMemcpyDeviceToHost);
                        // for(int i = 0; i < systemPtr_->mesh().size(); ++i)
                        //     std::cout << c[i] << std::endl;

                        systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                                  systemPtr_->cFields());
                    }
                }
                if (itr > maxHist_ + 1)
                {
                    invertMatrix_.deallocate();
                    coeffs_.deallocate();
                    vM_.deallocate();
                }

                return 1;
            }

            template <int D>
            void AmIterator<D>::computeDeviation()
            {
                int nbs = systemPtr_->basis().nStar();
                int nm = systemPtr_->dmixture().nMonomer();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(nbs, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                omHists_.append(systemPtr_->wFields());

                if (isFlexible_)
                {
                    CpHists_.append(systemPtr_->unitCell().parameters());
                }

                for (int i = 0; i < nm; ++i)
                {
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(), 0.0, nbs);
                }
#if CMP == 1
                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                   systemPtr_->cField(j).cDField(),
                                                                                   systemPtr_->dmixture().kpN(),
                                                                                   nbs);
                    }
                    pointWiseSubtractFloat<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                    systemPtr_->dmixture().kpN(),
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
                    inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                 systemPtr_->dmixture().bu0().cDField(),
                                                                                 nbs);
                    pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                               systemPtr_->wField(i).cDField(),
                                                                               -1.0,
                                                                               nbs);
                }
#endif
#if CMP == 0
                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                                    systemPtr_->cField(j).cDField(),
                                                                                    systemPtr_->dmixture().bu0().cDField(),
                                                                                    systemPtr_->interaction().chi(i, j),
                                                                                    nbs);

                        //  pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                        //  (tempDev[i].cDField(),
                        //   systemPtr_->cField(j).cDField(),
                        //   0.9,
                        //   nbs);
                    }
                }

                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField() + 1,
                                                                                   systemPtr_->wField(j).cDField() + 1,
                                                                                   -systemPtr_->interaction().idemp(i, j),
                                                                                   nbs - 1);
                    }
                }

                for (int i = 0; i < nm; ++i)
                {
                    pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tempDev[i].cDField(),
                                                                               systemPtr_->wField(i).cDField(),
                                                                               -1.0,
                                                                               1);

                    // subtractUniform <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                    // (tempDev[i].cDField(),
                    //  0.9,
                    //  1);
                }
#endif

                devHists_.append(tempDev);

                if (isFlexible_)
                {
                    FArray<double, 6> tempCp;
                    for (int i = 0; i < (systemPtr_->unitCell()).nParameter(); i++)
                    {
                        // format????
                        tempCp[i] = -80.0 * ((systemPtr_->dmixture()).stress(i));
                        // std::cout << "tempCp[" << i << "] = " << tempCp [i] << "\n";//exit(1);
                    }
                    devCpHists_.append(tempCp);
                }
            }

            template <int D>
            bool AmIterator<D>::isConverged(int itr)
            {
                double error;
                double dError = 0;
                // double wError = 0;
                // double max_error = 0;
                int nStar = systemPtr_->basis().nStar();

                for (int i = 0; i < systemPtr_->dmixture().nMonomer(); ++i)
                {
                    double hostHists[nStar];
                    cudaMemcpy(hostHists, devHists_[0][i].cDField(), sizeof(double) * nStar, cudaMemcpyDeviceToHost);
                    for (int j = 0; j < systemPtr_->basis().nStar(); ++j)
                    {
                        double absHostHists = abs(hostHists[j]);
                        if (absHostHists > dError)
                            dError = absHostHists;
                    }
                }
                // std::cout << dError << std::endl;
                // exit(1);

                if (isFlexible_)
                {
                    for (int i = 0; i < systemPtr_->unitCell().nParameter(); i++)
                    {
                        // dError += devCpHists_[0][i] *  devCpHists_[0][i];
                        // // wError += systemPtr_->unitCell().nParameter();
                        // wError +=  systemPtr_->unitCell().parameter(i) * systemPtr_->unitCell().parameter(i);
                        double absHostCpHists = abs(devCpHists_[0][i]);
                        if (absHostCpHists > dError)
                            dError = absHostCpHists;
                    }
                }
                // std::cout << "max_error" << dError << "\n";
                // exit(1);

                // error = sqrt(dError / wError);
                error = dError;
                if (itr % 50 == 0 || itr == 1)
                {
                    // Log::file() << " dError :" << Dbl(dError) << '\n';
                    // Log::file() << " wError :" << Dbl(wError) << '\n';
                    Log::file() << "Error :" << Dbl(error) << '\n';
                }

                if (error < epsilon_)
                {
                    Log::file() << std::endl
                                << "------- CONVERGED ---------" << std::endl;
                    Log::file() << "Final error        : " << Dbl(error) << "\n";
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
                if (itr == 1)
                {
                    histMat_.reset();
                    return 0;
                }
                else
                {
                    int nMonomer = systemPtr_->dmixture().nMonomer();
                    int nParameter = systemPtr_->unitCell().nParameter();
                    int nStar = systemPtr_->basis().nStar();

                    int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                    ThreadGrid::setThreadsLogical(nStar, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                    double elm, elm_cp;

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
                                temp1.allocate(systemPtr_->basis().nStar());
                                temp2.allocate(systemPtr_->basis().nStar());
                                pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][k].cDField(), devHists_[i + 1][k].cDField(),
                                                                                                 temp1.cDField(), systemPtr_->basis().nStar());
                                pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][k].cDField(), devHists_[j + 1][k].cDField(),
                                                                                                 temp2.cDField(), systemPtr_->basis().nStar());
                                elm += gpuInnerProduct(temp1.cDField(), temp2.cDField(), systemPtr_->basis().nStar());
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
                    }

                    for (int i = 0; i < nHist_; ++i)
                    {
                        vM_[i] = 0;
                        for (int j = 0; j < nMonomer; ++j)
                        {
                            RDField<D> temp;
                            temp.allocate(systemPtr_->basis().nStar());
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[0][j].cDField(), devHists_[i + 1][j].cDField(),
                                                                                             temp.cDField(), systemPtr_->basis().nStar());
                            vM_[i] += gpuInnerProduct(temp.cDField(), devHists_[0][j].cDField(), systemPtr_->basis().nStar());
                            temp.deallocate();
                        }
                        // vM_[i] = histMat_.makeVm(i, nHist_);
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
                    }
                    else
                    {
                        LuSolver solver;
                        solver.allocate(nHist_);
                        solver.computeLU(invertMatrix_);
                        solver.solve(vM_, coeffs_);
                    }
                    return 0;
                }
            }

            template <int D>
            void AmIterator<D>::buildOmega(int itr)
            {
                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(systemPtr_->basis().nStar(), NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                if (itr == 1)
                {
                    for (int i = 0; i < systemPtr_->dmixture().nMonomer(); ++i)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                            omHists_[0][i].cDField(), systemPtr_->basis().nStar());
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                                   devHists_[0][i].cDField(), lambda_, systemPtr_->basis().nStar());
                    }

                    if (isFlexible_)
                    {
                        cellParameters_.clear();
                        for (int m = 0; m < (systemPtr_->unitCell()).nParameter(); ++m)
                        {
                            cellParameters_.append(CpHists_[0][m] + lambda_ * devCpHists_[0][m]);
                            // std::cout<< "cell param : " << CpHists_[0][m] +lambda_* devCpHists_[0][m] << "\n";
                        }

                        systemPtr_->unitCell().setParameters(cellParameters_);
                        systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                        systemPtr_->dmixture().setU0(systemPtr_->unitCell(),
                                                     systemPtr_->wavelist());
                        systemPtr_->dmixture().setupUnitCell(systemPtr_->unitCell(),
                                                             systemPtr_->wavelist());
                    }
                }
                else
                {
                    for (int j = 0; j < systemPtr_->dmixture().nMonomer(); ++j)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wArrays_[j].cDField(),
                                                                            omHists_[0][j].cDField(), systemPtr_->basis().nStar());
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dArrays_[j].cDField(),
                                                                            devHists_[0][j].cDField(), systemPtr_->basis().nStar());
                    }
                    for (int i = 0; i < nHist_; ++i)
                    {
                        for (int j = 0; j < systemPtr_->dmixture().nMonomer(); ++j)
                        {
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(omHists_[i + 1][j].cDField(),
                                                                                             omHists_[0][j].cDField(),
                                                                                             tempDev[0].cDField(),
                                                                                             systemPtr_->basis().nStar());
                            pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wArrays_[j].cDField(),
                                                                                       tempDev[0].cDField(), coeffs_[i],
                                                                                       systemPtr_->basis().nStar());
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(devHists_[i + 1][j].cDField(),
                                                                                             devHists_[0][j].cDField(), tempDev[0].cDField(),
                                                                                             systemPtr_->basis().nStar());
                            pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dArrays_[j].cDField(),
                                                                                       tempDev[0].cDField(), coeffs_[i], systemPtr_->basis().nStar());
                        }
                    }
                    for (int i = 0; i < systemPtr_->dmixture().nMonomer(); ++i)
                    {
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                            wArrays_[i].cDField(), systemPtr_->basis().nStar());
                        pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(systemPtr_->wField(i).cDField(),
                                                                                   dArrays_[i].cDField(), lambda_, systemPtr_->basis().nStar());
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
                        systemPtr_->unitCell().setLattice();
                        systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                        systemPtr_->dmixture().setU0(systemPtr_->unitCell(),
                                                     systemPtr_->wavelist());
                        systemPtr_->dmixture().setupUnitCell(systemPtr_->unitCell(),
                                                             systemPtr_->wavelist());
                    }
                }
            }

        }
    }
}

#endif
