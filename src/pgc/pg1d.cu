#include <pgc/System.h>

int main(int argc, char **argv)
{
   Pscf::Pspg::Continuous::System<1> system;

   // Process command line options
   system.setOptions(argc, argv);

   // Read parameters from default parameter file
   system.readParam();

   // Read command script to run system
   system.readCommands();
   // cudaDeviceReset();
   return 0;
}
