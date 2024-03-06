#include "Bond.tpp"

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {

            using namespace Util;

            // Explicit instantiation of relevant class instances
            template class Bond<1>;
            template class Bond<2>;
            template class Bond<3>;
        }
    }
}
