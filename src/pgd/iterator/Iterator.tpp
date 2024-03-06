#include "Iterator.h"
/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/
namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            using namespace Util;

            template <int D>
            Iterator<D>::Iterator()
            {
                setClassName("Iterator");
            }

            template <int D>
            Iterator<D>::Iterator(System<D> *system)
                : systemPtr_(system)
            {
                setClassName("Iterator");
            }

            template <int D>
            Iterator<D>::~Iterator()
            {
            }

        }
    }
}