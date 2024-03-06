#ifndef D_BLOCK_TMPL_H
#define D_BLOCK_TMPL_H

#include <pscf/chem/BondDescriptor.h>
#include <util/containers/Pair.h>

#include <cmath>

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            using namespace Util;

            template <class TP>
            class BondTmpl : public BondDescriptor
            {
            public:
                typedef TP DPropagator;

                typedef typename TP::CField CField;

                typedef typename TP::WField WField;

                BondTmpl();

                virtual ~BondTmpl();

                virtual void setKuhn(double kuhn1, double kuhn2);

                // virtual void setKuhn (double kuhn1);

                TP &propagator(int directionId);

                TP const &propagator(int directionId) const;

                typename TP::CField &cField();

                double kuhn() const;

            private:
                Pair<DPropagator> propagators_;

                CField cField_;

                double kuhn_;
            };

            template <class TP>
            BondTmpl<TP>::BondTmpl()
                : propagators_(),
                  cField_(),
                  kuhn_(0.0)
            {
                propagator(0).setDirectionId(0);
                propagator(1).setDirectionId(1);
                propagator(0).setPartner(propagator(1));
                propagator(1).setPartner(propagator(0));
            }

            template <class TP>
            BondTmpl<TP>::~BondTmpl() = default;

            /// non-inline
            template <class TP>
            void BondTmpl<TP>::setKuhn(double kuhn1, double kuhn2)
            {
                kuhn_ = std::sqrt(kuhn1 * kuhn2);
            }

            /// inline
            template <class TP>
            inline TP &BondTmpl<TP>::propagator(int directionId)
            {
                return propagators_[directionId];
            }

            template <class TP>
            inline TP const &BondTmpl<TP>::propagator(int directionId) const
            {
                return propagators_[directionId];
            }

            template <class TP>
            inline
                typename TP::CField &
                BondTmpl<TP>::cField()
            {
                return cField_;
            }

            template <class TP>
            inline double BondTmpl<TP>::kuhn() const
            {
                return kuhn_;
            }

        }
    }
}

#endif
