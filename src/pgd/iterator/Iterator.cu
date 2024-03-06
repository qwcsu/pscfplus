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
            template class Iterator<1>;
            template class Iterator<2>;
            template class Iterator<3>;
        }
    }
}
