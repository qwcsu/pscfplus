/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include <pgc/System.h>

int main(int argc, char **argv)
{
   Pscf::Pspg::Continuous::System<2> system;

   // Process command line options
   system.setOptions(argc, argv);

   // Read parameters from default parameter file
   system.readParam();

   // Read command script to run system
   system.readCommands();
   // cudaDeviceReset();
   return 0;
}
