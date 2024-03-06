#ifndef D_ITERATOR_H
#define D_ITERATOR_H

#include <util/param/ParamComposite.h> // base class
#include <util/global.h>

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            template <int D>
            class System;

            using namespace Util;

            /**
             * Base class for iterative solvers for SCF equations.
             *
             * \ingroup Pscf_Discrete_Iterator_Module
             */
            template <int D>
            class Iterator : public ParamComposite
            {

            public:
                Iterator();

                Iterator(System<D> *system);

                ~Iterator();

                virtual int solve() = 0;

                System<D> *systemPtr_;
            };
        }
    }
}
#include "Iterator.tpp"
#endif