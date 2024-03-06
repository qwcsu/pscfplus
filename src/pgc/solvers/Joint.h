#ifndef PGC_JOINT_H
#define PGC_JOINT_H

#include "Propagator.h"    
#include <pscf/mesh/Mesh.h>
#include <pspg/field/RDField.h> 
#include <util/containers/FArray.h>

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            using namespace Util;

            template <int D>
            class Joint
            {
            public:

                Joint() = default;

                ~Joint();

                RDField<D> &jFieldRGrid();

                void setDiscretization(const Mesh<D> &mesh);

            private:

                RDField<D> jField_;

                Mesh<D> const *meshPtr_;
            };

            template <int D>
            RDField<D> &Joint<D>::jFieldRGrid()
            {
                return jField_;
            }

#ifndef PGC_JOINT_TPP
            // Suppress implicit instantiation
            extern template class Joint<1>;
            extern template class Joint<2>;
            extern template class Joint<3>;
#endif
        }
    }
}

#endif