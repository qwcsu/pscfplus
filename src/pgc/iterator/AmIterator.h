#ifndef PGC_AM_ITERATOR_H
#define PGC_AM_ITERATOR_H

/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include <pgc/iterator/Iterator.h> // base class
#include <pgc/solvers/Mixture.h>
#include <pscf/math/LuSolver.h>
#include <util/containers/DArray.h>
#include <util/containers/DMatrix.h>
#include <util/containers/RingBuffer.h>
#include <pgc/iterator/HistMat.h>
#include <pspg/field/RDField.h>
#include <util/containers/FArray.h>

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            using namespace Util;

            /**
             * Anderson mixing iterator for the pseudo spectral method
             *
             * \ingroup Pscf_Continuous_Iterator_Module
             */
            template <int D>
            class AmIterator : public Iterator<D>
            {
            public:
                typedef RDField<D> WField;
                typedef RDField<D> CField;
                /**
                 * Default constructor
                 */
                AmIterator();

                /**
                 * Constructor
                 *
                 * \param system pointer to a system object
                 */
                explicit AmIterator(System<D> *system);

                /**
                 * Destructor
                 */
                ~AmIterator();

                /**
                 * Read all parameters and initialize.
                 *
                 * \param in input filestream
                 */
                void readParameters(std::istream &in);

                /**
                 * Allocate all arrays
                 *
                 */
                void allocate();

                /**
                 * Iterate to a solution
                 *
                 */
                int solve();

                /**
                 * Getter for epsilon
                 */
                double epsilon();

                /**
                 * Getter for the maximum number of field histories to
                 * convolute into a new field
                 */
                int maxHist();

                /**
                 * Getter for the maximum number of iteration before convergence
                 */
                int maxItr();

                /**
                 * Compute the deviation of wFields from a mean field solution
                 */
                void computeDeviation();

                /**
                 * Compute the error from deviations of wFields and compare with epsilon_
                 * \return true for error < epsilon and false for error >= epsilon
                 */
                bool isConverged(int itr);

                /**
                 * Determine the coefficients that would minimize invertMatrix_ Umn
                 *
                 */
                int minimizeCoeff(int itr);

                /**
                 * Rebuild wFields for the next iteration from minimized coefficients
                 */
                void buildOmega(int itr);

                double final_error;

            private:
                /// error tolerance
                double epsilon_;

                /// current error
                double error_;

                /// Flexible unit cell (1) or rigid cell (0), default value = 0
                bool isFlexible_;

                /// free parameter for minimization
                double lambda_;

                /// number of histories to convolute into a new solution [0,maxHist_]
                int nHist_;

                // maximum number of histories to convolute into a new solution
                // AKA size of matrix
                int maxHist_;

                /// number of maximum iteration to perform
                int maxItr_;

                int kSize_;

                /// holds histories of deviation for each monomer
                /// 1st index = history, 2nd index = monomer, 3rd index = ngrid
                // The ringbuffer used is now slightly modified to return by reference

                RingBuffer<DArray<RDField<D>>> devHists_;
                RingBuffer<DArray<RDField<D>>> omHists_;

                /// holds histories of deviation for each cell parameter
                /// 1st index = history, 2nd index = cell parameter
                // The ringbuffer used is now slightly modified to return by reference
                RingBuffer<FArray<double, 6>> devCpHists_;
                RingBuffer<FSArray<double, 6>> CpHists_;

                FSArray<double, 6> cellParameters_;

                /// Umn, matrix to be minimized
                DMatrix<double> invertMatrix_;

                /// Cn, coefficient to convolute previous histories with
                DArray<double> coeffs_;

                DArray<double> vM_;

                /// bigW, blended omega fields
                DArray<RDField<D>> wArrays_;

                /// bigWcP, blended parameter
                FArray<double, 6> wCpArrays_;

                /// bigDCp, blened deviation parameter. new wParameter = bigWCp + lambda * bigDCp
                FArray<double, 6> dCpArrays_;

                /// bigD, blened deviation fields. new wFields = bigW + lambda * bigD
                DArray<RDField<D>> dArrays_;

                DArray<RDField<D>> tempDev;
                DArray<RDField<D>> tempDev_dct;

                HistMat<cudaReal> histMat_;

                cudaReal *d_temp_;
                cudaReal *temp_;
                RDField<D> tmp;

                using Iterator<D>::setClassName;
                using Iterator<D>::systemPtr_;
                using ParamComposite::read;
                using ParamComposite::readOptional;

                // friend:
                // for testing purposes
            };

            template <int D>
            inline double AmIterator<D>::epsilon()
            {
                return epsilon_;
            }

            template <int D>
            inline int AmIterator<D>::maxHist()
            {
                return maxHist_;
            }

            template <int D>
            inline int AmIterator<D>::maxItr()
            {
                return maxItr_;
            }

#ifndef PSPG_AM_ITERATOR_TPP
            // Suppress implicit instantiation
            extern template class AmIterator<1>;
            extern template class AmIterator<2>;
            extern template class AmIterator<3>;
#endif

        }
    }
}
// #include "AmIterator.tpp"
#endif
