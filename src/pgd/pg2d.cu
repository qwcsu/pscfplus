#include <pgd/phase.h>

int main(int argc, char **argv)
{
    Pscf::Pspg::Discrete::System<2> sys2d;

    sys2d.setOptions(argc, argv);

    sys2d.readParam();

    sys2d.readCommands();

    return 0;
}
