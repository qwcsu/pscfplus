/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include "Polymer.tpp"

namespace Pscf
{
   namespace Pspg
   {
      namespace Continuous
      {

         using namespace Util;

         // Explicit instantiation of relevant class instances
         template class Polymer<1>;
         template class Polymer<2>;
         template class Polymer<3>;

      }
   }
}
