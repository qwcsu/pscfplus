/*
* PSCF++ Package 
*
* Copyright 2010 - 2017, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "FFTBatched.tpp"

namespace Pscf {
namespace Pspg {
   using namespace Util;

   template class FFTBatched<1>;
   template class FFTBatched<2>;
   template class FFTBatched<3>;

}
}
