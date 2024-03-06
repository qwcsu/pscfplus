#ifndef D_ITERATOR_H
#define D_ITERATOR_H
/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/
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