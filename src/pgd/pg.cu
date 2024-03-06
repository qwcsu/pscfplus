/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include <pgd/phase.h>
#include <getopt.h>

int main(int argc, char **argv)
{

  // Read program arguments
  int c;
  int dim1 = -1;
  int dim2 = -1;
  bool eFlag = false;
  char *dArg;
  opterr = 0;
  
  while ((c = getopt(argc, argv, "er:d:")) != -1)
  {
    switch (c)
    {
      case 'd':
        dArg = optarg;
      
        if (dArg[0] == '3' && dArg[2] == '3')
        {
          dim1 = 3;
          dim2 = 3;
        }
        else if (dArg[0] == '3' && dArg[2] == '2')
        {
          dim1 = 3;
          dim2 = 2;
        }
        else if (dArg[0] == '2' && dArg[2] == '3')
        {
          dim1 = 2;
          dim2 = 3;
        }
        else if (dArg[0] == '3' && dArg[2] == '1')
        {
          dim1 = 3;
          dim2 = 1;
        }
        else if (dArg[0] == '1' && dArg[2] == '3')
        {
          dim1 = 1;
          dim2 = 3;
        }
        else if (dArg[0] == '3' && dArg[2] == '0')
        {
          dim1 = 3;
          dim2 = 0;
        }
        else if (dArg[0] == '0' && dArg[2] == '3')
        {
          dim1 = 0;
          dim2 = 3;
        }
        else if (dArg[0] == '2' && dArg[2] == '2')
        {
          dim1 = 2;
          dim2 = 2;
        }
        else if (dArg[0] == '2' && dArg[2] == '1')
        {
          dim1 = 2;
          dim2 = 1;
        }
        else if (dArg[0] == '1' && dArg[2] == '2')
        {
          dim1 = 1;
          dim2 = 2;
        }
         else if (dArg[0] == '2' && dArg[2] == '0')
        {
          dim1 = 2;
          dim2 = 0;
        }
        else if (dArg[0] == '0' && dArg[2] == '2')
        {
          dim1 = 0;
          dim2 = 2;
        }
        else if (dArg[0] == '1' && dArg[2] == '1')
        {
          dim1 = 1;
          dim2 = 1;
        }
        else if (dArg[0] == '1' && dArg[2] == '0')
        {
          dim1 = 1;
          dim2 = 0;
        }
        else if (dArg[0] == '0' && dArg[2] == '1')
        {
          dim1 = 0;
          dim2 = 1;
        }
        else if (dArg[0] == '3')
        {
          dim1 = 3;
        }
        else if (dArg[0] == '2')
        {
          dim1 = 2;
        }
        else if (dArg[0] == '1')
        {
          dim1 = 1;
        }
        else
        {
          std::cout << "Wrong -d argument(s)" << "\n";
          exit(1);
        }
        break;
      case 'e':
        eFlag = true;
        break;
    }
  }
  
  if (dim1 == 3 && dim2 == -1)
  {
    Pscf::Pspg::Discrete::System<3> system;
    system.setOptionsOutside("param", "command", eFlag);
    system.readParam();
    system.readCommandsJson();

  }
  else if (dim1 == 2 && dim2 == -1)
  {
    Pscf::Pspg::Discrete::System<2> system;
    system.setOptionsOutside("param", "command", eFlag);
    system.readParam();
    system.readCommandsJson();

  }
  else if (dim1 == 1 && dim2 == -1)
  {
    Pscf::Pspg::Discrete::System<1> system;
    system.setOptionsOutside("param", "command", eFlag);
    system.readParam();
    system.readCommandsJson();

  }
  else if (dim1 == 3 && dim2 == 3)
  {
    Pscf::Pspg::Discrete::System<3> system1;
    Pscf::Pspg::Discrete::System<3> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 3 && dim2 == 2)
  {
    Pscf::Pspg::Discrete::System<3> system1;
    Pscf::Pspg::Discrete::System<2> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 2 && dim2 == 3)
  {
    Pscf::Pspg::Discrete::System<2> system1;
    Pscf::Pspg::Discrete::System<3> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 3 && dim2 == 1)
  {
    Pscf::Pspg::Discrete::System<3> system1;
    Pscf::Pspg::Discrete::System<1> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 1 && dim2 == 3)
  {
    Pscf::Pspg::Discrete::System<1> system1;
    Pscf::Pspg::Discrete::System<3> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 3 && dim2 == 0)
  {
    Pscf::Pspg::Discrete::System<3> system1;

    computeTwoPhases(&system1, eFlag, 1);

    return 0;
  }
  else if (dim1 == 0 && dim2 == 3)
  {
    Pscf::Pspg::Discrete::System<3> system2;

    computeTwoPhases(&system2, eFlag, 2);

    return 0;
  }
  else if (dim1 == 2 && dim2 == 2)
  {
    Pscf::Pspg::Discrete::System<2> system1;
    Pscf::Pspg::Discrete::System<2> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 2 && dim2 == 1)
  {
    Pscf::Pspg::Discrete::System<2> system1;
    Pscf::Pspg::Discrete::System<1> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 1 && dim2 == 2)
  {
    Pscf::Pspg::Discrete::System<1> system1;
    Pscf::Pspg::Discrete::System<2> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 2 && dim2 == 0)
  {
    Pscf::Pspg::Discrete::System<2> system1;

    computeTwoPhases(&system1, eFlag, 1);

    return 0;
  }
  else if (dim1 == 0 && dim2 == 2)
  {
    Pscf::Pspg::Discrete::System<2> system2;

    computeTwoPhases(&system2, eFlag, 2);

    return 0;
  }
  else if (dim1 == 1 && dim2 == 1)
  {
    Pscf::Pspg::Discrete::System<1> system1;
    Pscf::Pspg::Discrete::System<1> system2;

    computeTwoPhases(&system1, &system2, eFlag);

    return 0;
  }
  else if (dim1 == 1 && dim2 == 0)
  {
    Pscf::Pspg::Discrete::System<1> system1;

    computeTwoPhases(&system1, eFlag, 1);

    return 0;
  }
  else if (dim1 == 0 && dim2 == 1)
  {
    Pscf::Pspg::Discrete::System<1> system2;

    computeTwoPhases(&system2, eFlag, 2);

    return 0;
  }
  else
  {
    std::cout << "Wrong dimentionality" << "\n";

    return 0;
  }
}
