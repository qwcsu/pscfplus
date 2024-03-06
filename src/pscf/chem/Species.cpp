/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Species.h"

namespace Pscf
{ 

   using namespace Util;

   Species::Species()
    : ensemble_(Species::Closed)
   {}

   /* 
   * Extract a Species::Ensemble from an istream as a string.
   */
   std::istream& operator >> (std::istream& in, Species::Ensemble& ensemble)
   {
      std::string buffer;
      in >> buffer;
      if (buffer == "Closed" || buffer == "closed") {
         ensemble = Species::Closed;
      } else 
      if (buffer == "Open" || buffer == "open") {
         ensemble = Species::Open;
      } else {
         UTIL_THROW("Invalid Species::Ensemble string in operator >>");
      } 
      return in;
   }
   
   /* 
   * Insert a Species::Ensemble to an ostream as a string.
   */
   std::ostream& operator<<(std::ostream& out, Species::Ensemble ensemble) 
   {
      if (ensemble == Species::Closed) {
         out << "Closed";
      } else 
      if (ensemble == Species::Open) {
         out << "Open";
      } else 
      if (ensemble == Species::Unknown) {
         out << "Unknown";
      } else {
         std::cout << "Invalid Species::Ensemble value on input" << std::endl;
         UTIL_THROW("Unrecognized value for Species::Ensemble");
      } 
      return out; 
   }

} // namespace Pscf
