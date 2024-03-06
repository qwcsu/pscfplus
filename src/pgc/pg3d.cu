/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include <pgc/phase.h>

int main(int argc, char **argv)
{
   Pscf::Pspg::Continuous::System<3> system;

   system.setOptions(argc, argv);

   system.readParam();

   system.readCommands();
   return 0;

   // phase boundry
   /*
      Pscf::Pspg::System<3> system1;
      Pscf::Pspg::System<3> system2;

      // Process command line options
      system1.setOptionsOutside("param1", "command1");
      system2.setOptionsOutside("param2", "command2");
      system1.readParam();
      system2.readParam();
      // compute(&system1);
      // compute(&system2);
      // double chiN = riddr(&system1, &system2, 26.0, 40.0, dfc, 1e-2);
      double chiN = newton(1.39e+01, &system1, &system2, 0.01);
      std::cout << "The phase boundry point is ("
                << chiN
                << ", "
                << system1.mixture().monomer(0).step()      << ")"
                << std::endl;

      return 0;
   */
}
