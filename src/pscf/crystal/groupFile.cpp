/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "groupFile.h"
#include <util/global.h>

// Macros to generated quoted path to data directory, DAT_DIR
#define XSTR(s) STR(s)
#define STR(s) # s
#define DAT_DIR_STRING XSTR(DAT_DIR)

namespace Pscf { 

   using namespace Util;

   /**
   * Generates the file name from a group name.
   *
   * \param groupName standard name of space group
   */
   std::string makeGroupFileName(int D, std::string groupName)
   {
      std::string filename = DAT_DIR_STRING ;
      filename += "/groups/";
      if (D==1) {
         filename += "1/";
      } else
      if (D==2) {
         filename += "2/";
      } else
      if (D==3) {
         filename += "3/";
      } else {
         UTIL_THROW("Invalid dimension of space");
      }
      filename += groupName;
      return filename;
   }

} // namespace Pscf
