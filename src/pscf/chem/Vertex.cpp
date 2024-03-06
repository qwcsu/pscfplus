/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Vertex.h"
#include "BlockDescriptor.h"
#include "BondDescriptor.h"
#include <util/global.h>

namespace Pscf
{ 

   Vertex::Vertex()
    : inPropagatorIds_(),
      outPropagatorIds_(),
      id_(-1)
   {}

   Vertex::~Vertex()
   {}

   void Vertex::setId(int id)
   {  id_ = id; }

   void Vertex::addBlock(const BlockDescriptor& block)
   {
      // Preconditions
      if (id_ < 0) {
         UTIL_THROW("Negative vertex id");
      }
      if (block.id() < 0) {
         UTIL_THROW("Negative block id");
      }
      if (block.vertexId(0) == block.vertexId(1)) {
         UTIL_THROW("Error: Equal vertex indices in block");
      }

      Pair<int> propagatorId;
      propagatorId[0] = block.id();
      if (block.vertexId(0) == id_) {
         propagatorId[1] = 0;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 1;
         inPropagatorIds_.append(propagatorId);
      } else
      if (block.vertexId(1) == id_) {
         propagatorId[1] = 1;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 0;
         inPropagatorIds_.append(propagatorId);
      } else {
         UTIL_THROW("Neither block vertex id matches this vertex");
      }
   }

   void Vertex::addBond(const BondDescriptor& bond)
   {
      // Preconditions
      if (id_ < 0) {
         UTIL_THROW("Negative vertex id");
      }
      if (bond.id() < 0) {
         UTIL_THROW("Negative bond id");
      }
      if (bond.vertexId(0) == bond.vertexId(1)) {
         UTIL_THROW("Error: Equal vertex indices in bond");
      }

      Pair<int> propagatorId;
      propagatorId[0] = bond.id();
      if (bond.vertexId(0) == id_) {
         propagatorId[1] = 0;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 1;
         inPropagatorIds_.append(propagatorId);
      } else
      if (bond.vertexId(1) == id_) {
         propagatorId[1] = 1;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 0;
         inPropagatorIds_.append(propagatorId);
      } else {
         UTIL_THROW("Neither bond vertex id matches this vertex");
      }
   }

} 
