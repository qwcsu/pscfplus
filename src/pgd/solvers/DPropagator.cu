#include "DPropagator.tpp"

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {

            using namespace Util;

            // Explicit instantiation of relevant class instances
            template class DPropagator<1>;
            template class DPropagator<2>;
            template class DPropagator<3>;
        }
    }
}
