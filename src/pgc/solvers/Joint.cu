#include "Joint.tpp"

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            using namespace Util;

            // Explicit instantiation of relevant class instances
            template class Joint<1>;
            template class Joint<2>;
            template class Joint<3>;
        }
    }
}
