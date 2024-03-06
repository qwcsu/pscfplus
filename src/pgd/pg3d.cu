#include <pgd/phase.h>
// #include <pgd/System.h>

int main(int argc, char **argv)
{
  Pscf::Pspg::Discrete::System<3> sys3d;

  sys3d.setOptions(argc, argv);

  sys3d.readParam();

  sys3d.readCommands();

  return 0;

  // phase boundry

  //    Pscf::Pspg::DPD::System<3> system1;
  //    Pscf::Pspg::DPD::System<3> system2;

  // Process command line options
  //    system1.setOptionsOutside("param1", "command1");
  //    system2.setOptionsOutside("param2", "command2");
  //    system1.readParam();
  //    system2.readParam();
  //     //    compute(&system1);
  //     // compute(&system2);
  //     // double chiN = riddr(&system1, &system2, 2.483853247814E+01,2.503853247814E+01, dfc, 1e-2);
  //     double chiN_guess = 25.3890823099705;
  //     double chiN = newton(chiN_guess, &system1, &system2, 0.001);
  //     std::cout << "The phase boundry point is ("
  //               << chiN
  //               << ", "
  //               << system1.dpdmixture().monomer(0).step()      << ")"
  //               << std::endl;

  //     return 0;
}