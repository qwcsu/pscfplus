#include "DMixture.tpp"

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            using namespace Util;

            // Explicit instantiation of relevant class instances
            template class DMixture<1>;
            template class DMixture<2>;
            template class DMixture<3>;
        }
    }
}
