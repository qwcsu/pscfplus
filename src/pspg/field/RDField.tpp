#ifndef PGC_R_DFIELD_TPP
#define PGC_R_DFIELD_TPP

/*
* PSCF++ Package 
*
* Copyright 2010 - 2017, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "RDField.h"

namespace Pscf {
namespace Pspg {

   using namespace Util;

   /**
   * Default constructor.
   */
   template <int D>
   RDField<D>::RDField()
    : DField<cudaReal>()
   {}

   /*
   * Destructor.
   */
   template <int D>
   RDField<D>::~RDField()
   {}

   /*
   * Copy constructor.
   *
   * Allocates new memory and copies all elements by value.
   *
   *\param other the Field to be copied.
   */
   template <int D>
   RDField<D>::RDField(RDField<D> const & other)
    : DField<cudaReal>(),
      meshDimensions_(0)
   {
      if (!other.isAllocated()) {
         UTIL_THROW("Other Field must be allocated.");
      }

      capacity_ = other.capacity_;
      cudaMalloc((void**) &data_, capacity_ * sizeof(cudaReal));

      cudaMemcpy(data_, other.cDField(), capacity_ * sizeof(cudaReal), cudaMemcpyDeviceToDevice);
      meshDimensions_ = other.meshDimensions_;
   }

   /*
   * Assignment, element-by-element.
   *
   * This operator will allocate memory if not allocated previously.
   *
   * \throw Exception if other Field is not allocated.
   * \throw Exception if both Fields are allocated with unequal capacities.
   *
   * \param other the rhs Field
   */
   template <int D>
   RDField<D>& RDField<D>::operator = (RDField<D> const & other)
   {
      // // Check for self assignment
      if (this == &other) return *this;
      
      // Precondition
      if (!other.isAllocated()) {
         UTIL_THROW("Other Field must be allocated.");
      }

      if (!isAllocated()) {
         allocate(other.capacity());
         // std::cout << "hi " << other.capacity()<< " \n"; 
      } else if (capacity_ != other.capacity_) {
         UTIL_THROW("Cannot assign Fields of unequal capacity");
      }

      // Copy elements
      cudaMemcpy(data_, other.cDField(), capacity_ * sizeof(cudaReal), cudaMemcpyDeviceToDevice);
      meshDimensions_ = other.meshDimensions_;

      return *this;
      // DField<cudaReal>::operator = (other);
      // meshDimensions_ = other.meshDimensions_;

      return *this;
   }

}
}
#endif
