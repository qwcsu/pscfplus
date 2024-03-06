#ifndef D_POLYMER_H
#define D_POLYMER_H

#include "Bond.h"
#include <util/containers/DArray.h> 
#include <pspg/field/RDField.h>
#include <pgd/solvers/DPolymerTmpl.h>
#include <util/containers/FArray.h>
namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            using namespace Util;

            /**
             * Descriptor and solver for a branched polymer species
             * (Discrete chain model).
             *
             * The block-bond concentrations stored in the constituent Bond<D>
             * objects contain the block-bond concentrations (i.e., volume
             * fractions) computed in the most recent call of the compute
             * function.
             *
             * The phi() and mu() accessor functions, which are inherited
             * from DPolymerTmp< Bond<D> >, return the value of phi (spatial
             * average volume fraction of a species) or mu (chemical
             * potential) computed in the last call of the compute function.
             * If the ensemble for this species is closed, phi is read from
             * the parameter file and mu is computed. If the ensemble is
             * open, mu is read from the parameter file and phi is computed.
             *
             * \ingroup Pscf_Discrete_Solvers_Module
             */
            template <int D>
            class DPolymer : public DPolymerTmpl<Bond<D>>
            {
            public:
                typedef DPolymerTmpl<Bond<D>> Base;

                typedef typename Bond<D>::WField WField;

                DPolymer();

                ~DPolymer() = default;

                void setupUnitCell(UnitCell<D> const &unitCell,
                                   const WaveList<D> &wavelist);

                void compute(DArray<WField> const &wFields);

                void computeSegment(const Mesh<D> &mesh,
                                    DArray<WField> const &wFields);

                void computeStress(WaveList<D> &wavelist);

                double stress(int n);

                cudaReal sS(int n);

                using Base::bond;
                using Base::ensemble;
                using Base::N;
                using Base::nBond;
                using Base::Q;
                using Base::solve;

            protected:
                // protected inherited function with non-dependent names
                using ParamComposite::setClassName;

            private:
                // Stress contribution from this polymer species.
                FArray<double, 6> stress_;

                // Number of unit cell parameters.
                int nParams_;

                DArray<RDField<D>> sRho_;

                DArray<cudaReal> sS_;

                DArray<cudaReal> sB_;
            };

            template <int D>
            double DPolymer<D>::stress(int n)
            {
                return stress_[n];
            }

            template <int D>
            inline
            cudaReal DPolymer<D>::sS(int n)
            {
                return sS_[n];
            }

#ifndef D_POLYMER_TPP
            // Suppress implicit instantiation
            extern template class DPolymer<1>;
            extern template class DPolymer<2>;
            extern template class DPolymer<3>;
#endif
        }
    }
}

#endif // !D_POLYMER_H