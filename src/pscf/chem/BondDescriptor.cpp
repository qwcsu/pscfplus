#include "BondDescriptor.h"
#include <iomanip>

namespace Pscf
{   
    using namespace Util;

    /*
    * Constructor.
    */
    BondDescriptor::BondDescriptor()
        : id_(-1),
          N_(0)
    {
    }

    /*
    * Set the id for this bond.
    */ 
    void BondDescriptor::setId(int id)
    {  id_ = id; }

    /*
    * Set indices of associated vertices.
    */ 
    void BondDescriptor::setVertexIds(int vertexId0, int vertexId1)
    {     
        vertexIds_[0] = vertexId0; 
        vertexIds_[1] = vertexId1; 
    }

    /*
    * Set the number of segments in this block.
    */ 
    void BondDescriptor::setLength(int N)
    {  N_ = N; }

    /* 
    * Extract a BondDescriptor from an istream.
    */
    std::istream& operator >>(std::istream& in, BondDescriptor &bond)
    {
        in >> bond.id_;
        in >> bond.vertexIds_[0];
        in >> bond.vertexIds_[1];
        in >> bond.monomerIds_[0];
        in >> bond.monomerIds_[1];
        in >> bond.N_;
        if (bond.N_ == 0)
            bond.type_ = 0;
        else    
            bond.type_ = 1;
        return in;
    }

    /* 
    * Output a BondDescriptor to an ostream, without line breaks.
    */
    std::ostream& operator <<(std::ostream& out, const BondDescriptor &bond) 
    {
        out << bond.id_;
        out << std::setw(5) << bond.vertexIds_[0];
        out << std::setw(5) << bond.vertexIds_[1];
        out << std::setw(5) << bond.monomerIds_[0];
        out << std::setw(5) << bond.monomerIds_[1];
        out << std::setw(5) << bond.N_;
        if (bond.N_ == 0)
            out << "    joint bond";
        else    
            out << "    block bond";
        return out;
    }
    

}