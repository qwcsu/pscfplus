/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include "Mixture.tpp"

namespace Pscf
{
   namespace Pspg
   {
      namespace Continuous
      {

         using namespace Util;

         // Explicit instantiation of relevant class instances
         template class Mixture<1>;
         template class Mixture<2>;
         template class Mixture<3>;

      }
   }
}
