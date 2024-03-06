#ifndef Bond_H
#define Bond_H

#include <pgd/solvers/DPropagator.h>
#include "BondTmpl.h"
#include <pspg/field/RDField.h>
#include <pspg/field/RDFieldDft.h>
#include <pspg/field/FFT.h>
#include <pspg/field/FFTBatched.h>
#include <util/containers/FArray.h>
#include <pscf/crystal/UnitCell.h>
#include <pgc/solvers/WaveList.h>

namespace Pscf
{
    template <int D>
    class Mesh;
}

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            using namespace Util;

            /**
             * Bond within a branched polymer. (discrete chain model)
             *
             * Derived from BondTmpl< DPropagator<D> >. A BondTmpl< DPropagator<D> >
             * has two SPropagator<D> members and is derived from BondDescriptor.
             *
             * \ingroup Pscf_Discrete_Solvers_Module
             */
            template <int D>
            class Bond : public BondTmpl<DPropagator<D>>
            {
            public:
                typedef typename Pscf::Pspg::Discrete::DPropagator<D>::Field Field;
                typedef typename Pscf::Pspg::Discrete::DPropagator<D>::WField WField;
                typedef typename Pscf::Pspg::Discrete::DPropagator<D>::QField QField;

                Bond();

                ~Bond();

                void setDiscretization(const Mesh<D> &mesh,
                                       const UnitCell<D> &unitCell);

                void setupUnitCell(const UnitCell<D> &unitCell,
                                   const WaveList<D> &wavelist,
                                   const int N);

                void setupSolver(WField const &w, int N);

                void setupFFT();

                void step(const cudaReal *q, cudaReal *qNew);

                void computeConcentration(double prefactor);

                void computeStress(WaveList<D> &wavelist, double prefactor);

                RDField<D> expW();

                RDField<D> expWp();

                // void computeStress(WaveList<D>& wavelist);

                Mesh<D> const &mesh() const;

                FFT<D> &fft();

                double stress(int n);

                using BondDescriptor::bondtype;
                using BondDescriptor::length;

                using BondTmpl<Pscf::Pspg::Discrete::DPropagator<D>>::kuhn;

                using BondTmpl<Pscf::Pspg::Discrete::DPropagator<D>>::propagator;

                using BondTmpl<Pscf::Pspg::Discrete::DPropagator<D>>::cField;

            private:
                FFT<D> fft_;

                // FFTBatched<D> fftBatched_;

                Mesh<D> const *meshPtr_;

                IntVec<D> kMeshDimensions_;

                int rSize_;
                int kSize_;

                RDField<D> expKsq_;

                cudaReal *expKsq_host;

                RDField<D> dexpKsq_;

                cudaReal *dexpKsq_host;

                RDField<D> expW_; // exp(-w/N)

                RDField<D> expWp_; // exp(+w/N)

                RDField<D> qr_;

                RDFieldDft<D> qk_;

                RDField<D> qsum_;

                RDFieldDft<D> qk1_;

                cudaReal *dk_host;

                RDField<D> dk_;

                cudaReal *kSq_d;

                int nParams_;

                FArray<double, 6> stress_;

                cudaReal *d_temp_;
                cudaReal *temp_;
            };

            template <int D>
            inline Mesh<D> const &Bond<D>::mesh() const
            {
                UTIL_ASSERT(meshPtr_);
                return *meshPtr_;
            }

            template <int D>
            inline RDField<D> Bond<D>::expW()
            {
                return expW_;
            }

            template <int D>
            inline RDField<D> Bond<D>::expWp()
            {
                return expWp_;
            }

            template <int D>
            inline FFT<D> &Bond<D>::fft()
            {
                return fft_;
            }

            template <int D>
            inline double Bond<D>::stress(int n)
            {
                return stress_[n];
            }

#ifndef BOND_TPP
            extern template class Bond<1>;
            extern template class Bond<2>;
            extern template class Bond<3>;
#endif
        }
    }
}

#endif