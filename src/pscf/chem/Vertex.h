#ifndef PSCF_VERTEX_H
#define PSCF_VERTEX_H

/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include <util/containers/GArray.h>
#include <util/containers/Pair.h>

namespace Pscf
{ 

   class BlockDescriptor;
   class BondDescriptor;
   using namespace Util;

   /**
   * A junction or chain end in a block polymer.
   *
   * \ingroup Pscf_Chem_Module
   */
   class Vertex
   {
   public:

      Vertex();
      ~Vertex();
  
      /**
      * Set the integer identifier of this vertex.
      * 
      * \param id identifier
      */ 
      void setId(int id);

      /**
      * Add block to the list of attached blocks.
      *
      * Preconditions: The id for this vertex must have been set, vertex
      * ids must have been set for the block, and the id of this vertex
      * must match one of the ids for the two vertices attached to the
      * block.
      * 
      * \param block attached BlockDescriptor object
      */ 
      void addBlock(const BlockDescriptor& block);

      /**
      * Add bond to the list of attached bonds.
      *
      * Preconditions: The id for this vertex must have been set, vertex
      * ids must have been set for the bond, and the id of this vertex
      * must match one of the ids for the two vertices attached to the
      * bond.
      * 
      * \param bond attached BondDescriptor object
      */ 
      void addBond(const BondDescriptor& bond);
      /**
      * Get the id of this vertex.
      */
      int id() const;

      /**
      * Get the number of attached blocks or bonds.
      */
      int size() const;

      /**
      * Get the block/bonds and direction of an incoming propagator.
      *
      * The first element of the integer pair is the block/bond id,
      * and the second is a direction id which is 0 if this 
      * vertex is vertex 1 of the block/bond, and 1 if this vertex
      * is vertex 0.
      *
      * \param i index of incoming propagator
      * \return Pair<int> containing block/bond index, direction index
      */
      const Pair<int>& inPropagatorId(int i) const;

      /**
      * Get the block/bond and direction of an outgoing propagator
      *
      * The first element of the integer pair is the block/bond id,
      * and the second is a direction id which is 0 if this 
      * vertex is vertex 0 of the block/bond, and 1 if this vertex
      * is vertex 1.
      *
      * \param i index of incoming propagator
      * \return Pair<int> containing block/bond index, direction index
      */
      const Pair<int>& outPropagatorId(int i) const;
   
   private:
   
      GArray< Pair<int> > inPropagatorIds_;
      GArray< Pair<int> > outPropagatorIds_;
      int id_;
   
   };

   inline int Vertex::id() const
   {  return id_; }

   inline int Vertex::size() const
   {  return outPropagatorIds_.size(); }

   inline 
   const Pair<int>& Vertex::inPropagatorId(int i) const
   {  return inPropagatorIds_[i]; }

   inline 
   const Pair<int>& Vertex::outPropagatorId(int i) const
   {  return outPropagatorIds_[i]; }

} 
#endif 
