#ifndef PGC_JOINT_TPP
#define PGC_JOINT_TPP

#include "Joint.h"         

namespace Pscf
{
    template <int D>
    class Mesh;
    template <int D>
    class UnitCell;
}

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {
            template<int D>
            Joint<D>::~Joint()
            {
                jField_.deallocate();
            }
            
            template <int D>
            void Joint<D>::setDiscretization(const Mesh<D> &mesh)
            {
                meshPtr_ = &mesh;
                
                jField_.allocate(meshPtr_->dimensions());
            }
        }
    }
}

#endif