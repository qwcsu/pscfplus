#ifndef D_PROPAGATOR_H
#define D_PROPAGATOR_H

#include <pscf/solvers/PropagatorTmpl.h> // base class template
#include <pspg/field/RDField.h>          // member template
#include <util/containers/DArray.h>      // member template
#include <pgc/solvers/Propagator.h>

namespace Pscf
{
    template <int D>
    class Mesh;
}

namespace Pscf
{
    namespace Pspg
    {

        __global__ void assignUniformReal(cudaReal *result, cudaReal uniform, int size);

        __global__ void assignReal(cudaReal *result, const cudaReal *rhs, int size);

        __global__ void inPlacePointwiseMul(cudaReal *a, const cudaReal *b, int size);

        namespace Discrete
        {
            template <int D>
            class Bond;
            using namespace Util;

            /**
             * CKE solver for one-direction of one bond.
             *
             * \ingroup Pscf_Discrete_Solvers_Module
             */
            template <int D>
            class DPropagator : public PropagatorTmpl<DPropagator<D>>
            {

            public:
                typedef RDField<D> Field;
                typedef RDField<D> WField;
                typedef RDField<D> CField;
                typedef RDField<D> QField;

                DPropagator();

                ~DPropagator();

                void setBond(Bond<D> &Bond);

                void allocate(int N, const Mesh<D> &mesh);

                void computeHead();

                void solve();

                cudaReal computeQ();

                const cudaReal *q(int i) const;

                cudaReal *q(int i);

                cudaReal *head() const;

                const cudaReal *tail() const;

                Bond<D> &bond();

                bool isAllocated() const;

                using PropagatorTmpl<DPropagator<D>>::nSource;
                using PropagatorTmpl<DPropagator<D>>::source;
                using PropagatorTmpl<DPropagator<D>>::partner;
                using PropagatorTmpl<DPropagator<D>>::setIsSolved;
                using PropagatorTmpl<DPropagator<D>>::isSolved;
                using PropagatorTmpl<DPropagator<D>>::hasPartner;

            private:
                Bond<D> *bondPtr_;

                Mesh<D> const *meshPtr_;

                int N_;

                bool isAllocated_;

                cudaReal *qFields_d;
            };

            template <int D>
            inline void DPropagator<D>::setBond(Bond<D> &bond)
            {
                bondPtr_ = &bond;
            }

            template <int D>
            inline cudaReal *DPropagator<D>::head() const
            {
                return qFields_d;
            }

            template <int D>
            inline const cudaReal *DPropagator<D>::tail() const
            {
                return qFields_d + ((N_ - 1) * meshPtr_->size());
            }

            template <int D>
            inline const cudaReal *DPropagator<D>::q(int i) const
            {
                return qFields_d + (i * meshPtr_->size());
            }

            template <int D>
            inline cudaReal *DPropagator<D>::q(int i)
            {
                return qFields_d + (i * meshPtr_->size());
            }

            template <int D>
            inline Bond<D> &DPropagator<D>::bond()
            {
                assert(bondPtr_);
                return *bondPtr_;
            }

            template <int D>
            inline bool DPropagator<D>::isAllocated() const
            {
                return isAllocated_;
            }

#ifndef D_PROPAGATOR_TPP
            extern template class DPropagator<1>;
            extern template class DPropagator<2>;
            extern template class DPropagator<3>;
#endif

        }
    }
}

#endif // !D_PROPAGATOR_H