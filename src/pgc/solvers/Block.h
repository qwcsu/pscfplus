#ifndef PGC_BLOCK_H
#define PGC_BLOCK_H

/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include "Propagator.h"             // base class argument
#include <pscf/solvers/BlockTmpl.h> // base class template
#include <pspg/field/RDField.h>     // member
#include <pspg/field/RDFieldDft.h>  // member
#include <pspg/field/FFT.h>         // member
#include <pspg/field/FFTBatched.h>  // member
#include <pspg/field/FCT.h>
#include <pgc/solvers/WaveList.h>
#include <util/containers/FArray.h>
#include <pscf/crystal/UnitCell.h>

namespace Pscf
{
    template <int D>
    class Mesh;
    template <int D>
    class UnitCell;
}

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            using namespace Util;

            /**
             * Block within a branched polymer. (Continuous Gaussian chain)
             *
             * Derived from BlockTmpl< Propagator<D> >. A BlockTmpl< Propagator<D> >
             * has two Propagator<D> members and is derived from BlockDescriptor.
             *
             * \ingroup Pscf_Continuous_Solvers_Module
             */
            template <int D>
            class Block : public BlockTmpl<Propagator<D>>
            {

            public:
                /**
                 * Generic field (base class)
                 */
                typedef typename Pscf::Pspg::Continuous::Propagator<D>::Field Field;

                /**
                 * Monomer chemical potential field.
                 */
                typedef typename Pscf::Pspg::Continuous::Propagator<D>::WField WField;

                /**
                 * Constrained partition function q(r,s) for fixed s.
                 */
                typedef typename Pscf::Pspg::Continuous::Propagator<D>::QField QField;

                // Member functions

                /**
                 * Constructor.
                 */
                Block();

                /**
                 * Destructor.
                 */
                ~Block();

                /**
                 * Initialize discretization and allocate required memory.
                 *
                 * \param ds desired (optimal) value for contour length step
                 * \param mesh spatial discretization mesh
                 */
                void setDiscretization(double ds, const Mesh<D> &mesh);

                /**
                 * Setup parameters that depend on the unit cell.
                 *
                 * \param unitCell unit cell, defining cell dimensions
                 * \param waveList container for properties of recip wavevectors
                 */
                void setupUnitCell(const UnitCell<D> &unitCell,
                                   const WaveList<D> &wavelist);

                /**
                 * Set solver for this block.
                 *
                 * \param w  chemical potential field for this monomer type
                 */
                void setupSolver(WField const &w);

                /**
                 * Initialize FFT and batch FFT classes.
                 */
#if DFT == 0
                void setupFFT();
#endif
#if DFT == 1
                /**
                 * Initialize FCTclasses.
                 */
                void setupFCT();
#endif
                /**
                 * Compute step of integration loop, from i to i+1.
                 */
                void step(const cudaReal *q, cudaReal *qNew);

                /**
                 * Compute derivatives of free energy with respect to cell parameters.
                 *
                 * \param waveList container for properties of recip. latt. wavevectors.
                 * \param prefactor prefactor = exp(mu_)/length(), where mu_ and length()
                 * is the chemical potential and chain length of the corresponding polymer,
                 * respectively.
                 */
                void computeStress(double prefactor);

                /**
                 * Compute the integral using RI-K while the backword propagtors are being
                 * computed from "slices".
                 *
                 *  \param q  Foward propagator.
                 *  \param qs Backward propagtor.
                 *  \param ic Coefficient given by RI-K method.
                 */
                void computeInt(cudaReal *q, cudaReal *qs, int ic);

                void setWavelist(WaveList<D> &wavelist);

                /**
                 * Get derivative of free energy with respect to a unit cell parameter.
                 *
                 * \param n  unit cell parameter index
                 */
                double stress(int n);

                /**
                 * Return associated spatial Mesh by reference.
                 */
                Mesh<D> const &mesh() const;

                /**
                 * Contour length step size.
                 */
                double ds() const;

                /**
                 * Number of contour length steps.
                 */
                int ns() const;

#if DFT == 0
                /**
                 * Return the fast Fourier transform object by reference.
                 */
                FFT<D> &fft();
#endif
#if DFT == 1    
                /**
                 * Return the fast Fcosine transform object by reference.
                 */
                FCT<D> &fct();
#endif

                // Functions with non-dependent names from BlockTmpl< Propagator<D> >
                using BlockTmpl<Pscf::Pspg::Continuous::Propagator<D>>::setKuhn;
                using BlockTmpl<Pscf::Pspg::Continuous::Propagator<D>>::propagator;
                using BlockTmpl<Pscf::Pspg::Continuous::Propagator<D>>::cField;
                using BlockTmpl<Pscf::Pspg::Continuous::Propagator<D>>::length;
                using BlockTmpl<Pscf::Pspg::Continuous::Propagator<D>>::kuhn;

                // Functions with non-dependent names from BlockDescriptor
                using BlockDescriptor::id;
                using BlockDescriptor::length;
                using BlockDescriptor::monomerId;
                using BlockDescriptor::setId;
                using BlockDescriptor::setLength;
                using BlockDescriptor::setMonomerId;
                using BlockDescriptor::setVertexIds;
                using BlockDescriptor::vertexId;
                using BlockDescriptor::vertexIds;

            private:
                // Fourier transform plan
#if DFT == 0
                FFT<D> fft_;

                // FFTBatched<D> fftBatched_;
#endif
#if DFT == 1
                // Fast cosine transform plan
                FCT<D> fct_;

                cudaReal *qtmp, *qstmp;
#endif
                /// Stress exerted by a polymer chain of a block.
                FArray<double, 6> stress_;

                // Array of elements containing exp(-K^2 b^2 ds/6)
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                RDField<D> expKsq_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                RDField<D> expKsq2_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                RDField<D> expKsq3_;
#endif
#if REPS == 4 || REPS == 3
                RDField<D> expKsq4_;
#endif
#if REPS == 4
                RDField<D> expKsq5_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                RDField<D> expW_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                RDField<D> expW2_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                RDField<D> expW3_;
#endif
#if REPS == 4 || REPS == 3
                RDField<D> expW4_;
#endif
#if REPS == 4
                RDField<D> expW5_;
#endif

                // Work array for real-space field.
                RDField<D> qr_;

                RDFieldDft<D> qk_;

                IntVec<D> kMeshDimensions_;

                int kSize_;

                RDField<D> qF, qsF;
                RDFieldDft<D> qtmpF, qstmpF;

                // cudaComplex* qkBatched_;
                // cudaComplex* qk2Batched_;
                // #endif

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                RDField<D> q1_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                RDField<D> q2_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                RDField<D> q3_;
#endif
#if REPS == 4 || REPS == 3
                RDField<D> q4_;
#endif
#if REPS == 4
                RDField<D> q5_;
#endif

                cudaReal *d_temp_;
                cudaReal *temp_;

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1 || REPS == 0
                cudaReal *expKsq_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS == 1
                cudaReal *expKsq2_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2
                cudaReal *expKsq3_host;
#endif
#if REPS == 4 || REPS == 3
                cudaReal *expKsq4_host;
#endif
#if REPS == 4
                cudaReal *expKsq5_host;
#endif

                /// Pointer to associated Mesh<D> object.
                Mesh<D> const *meshPtr_;

                int size_ph_;

                /// Contour length step size.
                double ds_;

                /// Number of contour length steps = # grid points - 1.
                int ns_;

                int nParams_;

                double *ic_;

                cudaReal *qkSum_;

                double inc_[6];

                WaveList<D> *wavelist_;

                cudaReal *dqk_, *dqk_c;
            };

            // Inline member functions

            /// Get number of contour steps.
            template <int D>
            inline int Block<D>::ns() const
            {
                return ns_;
            }

            /// Get number of contour steps.
            template <int D>
            inline double Block<D>::ds() const
            {
                return ds_;
            }

            /// Get derivative of free energy with respect to a unit cell parameter.
            template <int D>
            inline double Block<D>::stress(int n)
            {
                return stress_[n];
            }

            /// Get Mesh by reference.
            template <int D>
            inline Mesh<D> const &Block<D>::mesh() const
            {
                UTIL_ASSERT(meshPtr_)
                return *meshPtr_;
            }
#if DFT == 0
            template <int D>
            inline FFT<D> &Block<D>::fft()
            {
                return fft_;
            }
#endif
#if DFT == 1
            template <int D>
            inline FCT<D> &Block<D>::fct()
            {
                return fct_;
            }
#endif

#ifndef PGC_BLOCK_TPP
            // Suppress implicit instantiation
            extern template class Block<1>;
            extern template class Block<2>;
            extern template class Block<3>;
#endif

        }
    }
}
// #include "Block.tpp"
#endif
