#ifndef PGC_PROPAGATOR_H
#define PGC_PROPAGATOR_H

/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include <pscf/solvers/PropagatorTmpl.h> // base class template
#include <pspg/field/RDField.h>          // member template
#include <util/containers/DArray.h>      // member template

namespace Pscf
{
    template <int D>
    class Mesh;
}

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            template <int D>
            class Block;
            using namespace Util;

            /**
             * MDE solver for one-direction of one block.
             *
             * \ingroup Pscf_Continuous_Solvers_Module
             */
            template <int D>
            class Propagator : public PropagatorTmpl<Propagator<D>>
            {

            public:
                // Public typedefs

                /**
                 * Generic field (function of position).
                 */
                // where is these two used?
                typedef RDField<D> Field;

                /**
                 * Chemical potential field type.
                 */
                // where is these two used?
                typedef RDField<D> WField;

                /**
                 * Monomer concentration field type.
                 */
                typedef RDField<D> CField;

                /**
                 * Propagator q-field type.
                 */
                typedef RDField<D> QField;

                // Member functions

                /**
                 * Constructor.
                 */
                Propagator();

                /**
                 * Destructor.
                 */
                ~Propagator();

                /**
                 * Associate this propagator with a block.
                 *
                 * \param block associated Block object.
                 */
                void setBlock(Block<D> &block);

                void setJoint(bool hasSource);

                /**
                 * Associate this propagator with a block.
                 *
                 * \param ns number of contour length steps
                 * \param mesh spatial discretization mesh
                 */
                void allocate(int ns, const Mesh<D> &mesh);

                /**
                 * Compute the forward propagators and store 
                 * the "slices" for the corresponding block.
                 */
                void solveForward();

                /**
                 * Compute the backward propagators with the 
                 * stored "slices" as well as the integrals 
                 * and normalized single chain partition 
                 * function that are needed for concentration
                 * fields and for the corresponding block.  
                 */
                void solveBackward(cudaReal *q, int n);

                /**
                 * Compute and return partition function for the molecule.
                 *
                 * This function is called by 
                 * void solveBackward(cudaReal *q, int n)
                 */
                double intQ(cudaReal *q, cudaReal *qs);

                /**
                 * Return q-field at specified step.
                 *
                 * \param i step index
                 */
                const cudaReal *q(int i) const;

                cudaReal *q(int i);

                /**
                 * Return q-field at beginning of block (initial condition).
                 */
                cudaReal *head() const;

                cudaReal *qhead();

                cudaReal returnQ();

                /**
                 * Return q-field at end of block.
                 */
                // const cudaReal *tail() const;
                const cudaReal *qtail() const;

                /**
                 * Get the associated Block object by reference.
                 */
                Block<D> &block();

                /**
                 * Has memory been allocated for this propagator?
                 */
                bool isAllocated() const;

                bool isJoint();

                using PropagatorTmpl<Propagator<D>>::nSource;
                using PropagatorTmpl<Propagator<D>>::source;
                using PropagatorTmpl<Propagator<D>>::partner;
                using PropagatorTmpl<Propagator<D>>::setIsSolved;
                using PropagatorTmpl<Propagator<D>>::isSolved;
                using PropagatorTmpl<Propagator<D>>::hasPartner;

            protected:
                /**
                 * Compute initial QField at head from tail QFields of sources.
                 */
                void computeHead();

            private:
                // new array purely in device
                cudaReal *qFields_d;
                // Workspace
                // removing this. Does not seem to be used anywhere
                // QField work_;

                /// Pointer to associated Block.
                Block<D> *blockPtr_;

                /// Pointer to associated Mesh
                Mesh<D> const *meshPtr_;

                /// Number of contour length steps = # grid points - 1.
                int ns_;

                /// Is this propagator allocated?
                bool isAllocated_;

                bool hasSource_;

                // work array for inner product. Allocated and free-d in con and des
                cudaReal *d_temp_;
                cudaReal *temp_;

                cudaReal *q_cpy;

                int nc_;

                double Q_;
            };

            // Inline member functions

            /*
             * Return q-field at beginning of block.
             */
            template <int D>
            inline cudaReal *Propagator<D>::head() const
            {
                return qFields_d;
            }

            /*
             * Return q-field at beginning of block.
             */
            template <int D>
            inline cudaReal *Propagator<D>::qhead()
            {
                return qFields_d;
            }

            /*
             * Return q-field at end of block, after solution.
             */
            // template <int D>
            // inline const cudaReal *Propagator<D>::tail() const
            // {
            //     return qFields_d + ((ns_ - 1) * meshPtr_->size());
            // }

            template <int D>
            inline const cudaReal *Propagator<D>::qtail() const
            {
                return qFields_d + (nc_ * meshPtr_->size());
            }

            /*
             * Return q-field at specified step.
             */
            template <int D>
            inline const cudaReal *Propagator<D>::q(int i) const
            {
                return qFields_d + (i * meshPtr_->size());
            }

            template <int D>
            inline cudaReal *Propagator<D>::q(int i)
            {
                return qFields_d + (i * meshPtr_->size());
            }

            /*
             * Get the associated Block object.
             */
            template <int D>
            inline Block<D> &Propagator<D>::block()
            {
                assert(blockPtr_);
                return *blockPtr_;
            }

            template <int D>
            inline bool Propagator<D>::isAllocated() const
            {
                return isAllocated_;
            }

            /*
             * Associate this propagator with a block and direction
             */
            template <int D>
            inline void Propagator<D>::setBlock(Block<D> &block)
            {
                blockPtr_ = &block;
            }

            template <int D>
            inline void Propagator<D>::setJoint(bool hasSource)
            {
                hasSource_ = hasSource;
            }

            template <int D>
            inline bool Propagator<D>::isJoint()
            {
                return hasSource_;
            }

            template <int D>
            inline cudaReal Propagator<D>::returnQ()
            {
                return Q_;
            }

#ifndef PSPG_PROPAGATOR_TPP
            // Suppress implicit instantiation
            extern template class Propagator<1>;
            extern template class Propagator<2>;
            extern template class Propagator<3>;
#endif

        }
    }
}

#endif
