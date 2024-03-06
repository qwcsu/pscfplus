#ifndef BOND_DESCRIPTOR_H
#define BOND_DESCRIPTOR_H

#include <util/containers/Pair.h>

#include <iostream>

namespace Pscf
{
    using namespace Util;

    /**
    * A linear bond (including block-bond and joint-bond) 
    * within a block copolymer.
    * (discrete chain)
    * 
    * \ingroup Pscf_Chem_Module
    */
    class BondDescriptor
    {
    public:

        /**
        * Constructor.
        */ 
        BondDescriptor();

        /**
        * Serialize to/from archive.
        *
        * \param ar input or output Archive
        * \param versionId archive format version index
        */ 
        template <class Archive>
        void serialize(Archive& ar, unsigned int versionId);

        /// \name Setters
        //@{
        /**
        * Set the id for this block.
        *
        * \param id integer index for this bond
        */ 
        void setId(int id);

        /**
        * Set indices of associated vertices.
        *
        * \param vertexAId integer id of vertex A
        * \param vertexBId integer id of vertex B
        */ 
        void setVertexIds(int vertexAId, int vertexBId);

        /**
        * Set the length of this bond.
        *
        * The "length" is the number of segments on this bond.
        *
        * \param length block length (number of monomers).
        */ 
        virtual void setLength(int N);

        //@}
        /// \name Accessors (getters)
        //@{
        
        /**
        * Get the id of this bond.
        */ 
        int id() const;

        /**
        * Get the monomer type id at vertex i.
        */ 
        int monomerId(int i) const;

        /**
        * Get the pair of associated vertex ids.
        */ 
        const Pair<int>& vertexIds() const;

        /**
        * Get id of an associated vertex.
        *
        * \param i index of vertex (0 or 1)
        */ 
        int vertexId(int i) const;

        /**
        * Get the number of segments in this bond.
        * For a block bond, the return value should be 
        * a positive integer number.
        */
        int length() const;

        /**
        * Get the type of this bond.
        * true for block-bond; false for joint-bond
        */
        bool bondtype() const;

        bool hasJointSegment() const;

        //@}

    private:

        /// Identifier for this bond, unique within the polymer.
        int id_;

        /// Two propagators, one for each direction. 
        Pair<int> vertexIds_{};

        /// Identifiers for the associated monomer types at the vertices of this bond.
        Pair<int> monomerIds_{};

        /// Typy of this bond. 0 for block bond; 1 for joint bond.
        bool type_;   

        bool hasJointSegment_;

        /// Number of segments on this bond.
        int N_; 

        friend
        std::istream& operator >> (std::istream& in, BondDescriptor &bond);

        friend 
        std::ostream& operator << (std::ostream& out, const BondDescriptor &bond);
    };

    /**
    * istream extractor for a BlockDescriptor.
    *
    * \param in  input stream
    * \param bond  BondDescriptor to be read from stream
    * \return modified input stream
    */
    std::istream& operator >> (std::istream& in,  BondDescriptor &bond);

    /**
    * ostream inserter for a BlockDescriptor.
    *
    * \param out  output stream
    * \param bond  BondDescriptor to be written to stream
    * \return modified output stream
    */
    std::istream& operator << (std::istream& out, const BondDescriptor &bond);

    // Inline member functions

    /*
    * Get the id of this bond.
    */ 
    inline 
    int BondDescriptor::id() const
    {  return id_; }

    /*
    * Get the monomer type id at vertex i.
    */ 
    inline 
    int BondDescriptor::monomerId(int i) const
    {  return monomerIds_[i]; }

    /*
    * Get the pair of associated vertex ids.
    */ 
    inline 
    const 
    Pair<int>& BondDescriptor::vertexIds() const
    {  return vertexIds_; }

    /*
    * Get id of an associated vertex.
    */ 
    inline 
    int BondDescriptor::vertexId(int i) const
    {  return vertexIds_[i]; }

    /*
    * Get the lnumber of sgements in this block.
    */
    inline 
    int BondDescriptor::length() const
    {  return N_; }

    /*
    * Get the type of this bond
    * 0 for block bond; 1 for joint bond.
    */
    inline 
    bool BondDescriptor::bondtype() const
    {  return type_; }

    inline 
    bool BondDescriptor::hasJointSegment() const
    {  return hasJointSegment_; }

    /*
    * Serialize to/from an archive.
    */
    template <class Archive>
    void BondDescriptor::serialize(Archive& ar, unsigned int)
    {
        ar & id_;
        ar & vertexIds_;
        ar & monomerIds_;
        ar & N_;
    }

        
    
}


#endif