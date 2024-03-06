#ifndef D_POLYMER_TMPL_H
#define D_POLYMER_TMPL_H

#include <pscf/chem/Species.h>         // base class
#include <util/param/ParamComposite.h> // base class

#include <pscf/chem/Monomer.h>      // member template argument
#include <pscf/chem/Vertex.h>       // member template argument
#include <util/containers/Pair.h>   // member template
#include <util/containers/DArray.h> // member template
#include <util/containers/DMatrix.h>

#include <cmath>

namespace Pscf
{

    class Bond;

    using namespace Util;

    template <class Bond>
    class DPolymerTmpl : public Species, public ParamComposite
    {
    public:
        typedef typename Bond::DPropagator DPropagator;

        DPolymerTmpl();

        ~DPolymerTmpl() = default;

        virtual void readParameters(std::istream &in);

        virtual void solve();

        Bond &bond(int id);

        Bond const &bond(int id) const;

        const Vertex &vertex(int id) const;

        DPropagator &propagator(int bondId, int dirId);

        DPropagator const &propagator(int bondId, int dirId) const;

        DPropagator &propagator(int id);

        const Pair<int> &propagatorId(int i) const;

        /// Accessors
        int nBond() const;

        double Q() const;

        int nVertex() const;

        int nPropagator() const;

        int N() const;

    protected:
        virtual void makePlan();

        // virtual void readParameters(std::istream& in);
    private:
        /// Array of Bond objects in this polymer.
        DArray<Bond> bonds_;

        /// Array of Vertex objects in this polymer.
        DArray<Vertex> vertices_;

        /// Propagator ids, indexed in order of computation.
        DArray<Pair<int>> propagatorIds_;

        /// Number of bonds in this polymer
        int nBond_;

        /// Number of vertices (ends or junctions) in this polymer
        int nVertex_;

        /// Number of Joint segments in this polymer
        // int nJSegment_;

        /// Number of propagators (two per bond).
        int nPropagator_;

        double q_;
    };

    template <class Bond>
    inline int DPolymerTmpl<Bond>::nBond() const
    {
        return nBond_;
    }

    template <class Bond>
    inline Bond &DPolymerTmpl<Bond>::bond(int id)
    {
        return bonds_[id];
    }

    template <class Bond>
    inline const Bond &DPolymerTmpl<Bond>::bond(int id) const
    {
        return bonds_[id];
    }

    template <class Bond>
    inline int DPolymerTmpl<Bond>::nVertex() const
    {
        return nVertex_;
    }

    template <class Bond>
    inline const Vertex &DPolymerTmpl<Bond>::vertex(int id) const
    {
        return vertices_[id];
    }

    template <class Bond>
    inline int DPolymerTmpl<Bond>::nPropagator() const
    {
        return nPropagator_;
    }

    template <class Bond>
    inline int DPolymerTmpl<Bond>::N() const
    {
        int value = 0;
        for (int bondId = 0; bondId < nBond_; ++bondId)
        {
            if (bonds_[bondId].bondtype() == 1)
                value += bonds_[bondId].length();
        }
        // value += nJSegment_;
        return value;
    }

    template <class Bond>
    inline
        typename Bond::DPropagator &
        DPolymerTmpl<Bond>::propagator(int bondId, int dirId)
    {
        return bond(bondId).propagator(dirId);
    }

    template <class Bond>
    inline
        typename Bond::DPropagator const &
        DPolymerTmpl<Bond>::propagator(int bondId, int dirId) const
    {
        return bond(bondId).propagator(dirId);
    }

    template <class Bond>
    inline
        typename Bond::DPropagator &
        DPolymerTmpl<Bond>::propagator(int id)
    {
        Pair<int> propId = propagatorId(id);
        return propagator(propId[0], propId[1]);
    }

    template <class Bond>
    inline Pair<int> const &
    DPolymerTmpl<Bond>::propagatorId(int id) const
    {
        UTIL_CHECK(id >= 0)
        UTIL_CHECK(id <= nPropagator_)
        return propagatorIds_[id];
    }

    template <class Bond>
    inline double DPolymerTmpl<Bond>::Q() const
    {
        return q_;
    }

    template <class Bond>
    DPolymerTmpl<Bond>::DPolymerTmpl()
        : bonds_(),
          vertices_(),
          propagatorIds_(),
          nBond_(0),
          nVertex_(0),
          nPropagator_(0)
    {
        setClassName("DPolymerTmpl");
    }

    template <class Bond>
    void DPolymerTmpl<Bond>::readParameters(std::istream &in)
    {
        read<int>(in, "nBond", nBond_);
        read<int>(in, "nVertex", nVertex_);
        // read<int>(in, "nJSegment", nJSegment_);

        bonds_.allocate(nBond_);
        vertices_.allocate(nVertex_);
        propagatorIds_.allocate(2 * nBond_);

        readDArray<Bond>(in, "bonds", bonds_, nBond_);

        for (int vertexId = 0; vertexId < nVertex_; ++vertexId)
        {
            vertices_[vertexId].setId(vertexId);
        }

        int vertexId0, vertexId1;

        Bond *bondPtr;

        for (int bondId = 0; bondId < nBond_; ++bondId)
        {
            bondPtr = &(bonds_[bondId]);
            vertexId0 = bondPtr->vertexId(0);
            vertexId1 = bondPtr->vertexId(1);
            vertices_[vertexId0].addBond(*bondPtr);
            vertices_[vertexId1].addBond(*bondPtr);
        }

        makePlan();

        ensemble_ = Species::Closed;
        readOptional<Species::Ensemble>(in, "ensemble", ensemble_);
        if (ensemble_ == Species::Closed)
        {
            read(in, "phi", phi_);
        }
        else
        {
            UTIL_THROW("Cannot set the ensemble not to be closed");
        }

        Vertex const *vertexPtr = nullptr;
        DPropagator const *sourcePtr = nullptr;
        DPropagator *propagatorPtr = nullptr;
        Pair<int> propagatorId;

        int bondId, directionId, vertexId, i;
        for (bondId = 0; bondId < nBond(); ++bondId)
        {
            for (directionId = 0; directionId < 2; ++directionId)
            {
                vertexId = bond(bondId).vertexId(directionId);
                vertexPtr = &vertex(vertexId);
                propagatorPtr = &bond(bondId).propagator(directionId);

                for (i = 0; i < vertexPtr->size(); ++i)
                {
                    propagatorId = vertexPtr->inPropagatorId(i);

                    if (propagatorId[0] == bondId)
                    {
                        UTIL_CHECK(propagatorId[1] != directionId)
                    }
                    else
                    {
                        sourcePtr =
                            &bond(propagatorId[0]).propagator(propagatorId[1]);
                        propagatorPtr->addSource(*sourcePtr);
                    }
                }
            }
        }
    }

    template <class Bond>
    void DPolymerTmpl<Bond>::makePlan()
    {
        if (nPropagator_ != 0)
        {
            UTIL_THROW("nPropagator !=0 on entry");
        }

        DMatrix<bool> isFinished;
        isFinished.allocate(nBond_, 2);
        for (int i = 0; i < nBond_; ++i)
        {
            for (int iDirection = 0; iDirection < 2; ++iDirection)
            {
                isFinished(i, iDirection) = false;
            }
        }

        Pair<int> propagatorId;
        Vertex *inVertexPtr = nullptr;
        int inVertexId = -1;
        bool isReady;
        while (nPropagator_ < (nBond_) * 2)
        {
            for (int iBond = 0; iBond < nBond_; ++iBond)
            {
                for (int iDirection = 0; iDirection < 2; ++iDirection)
                {
                    if (!isFinished(iBond, iDirection))
                    {
                        inVertexId = bonds_[iBond].vertexId(iDirection);
                        inVertexPtr = &vertices_[inVertexId];
                        isReady = true;
                        for (int j = 0; j < inVertexPtr->size(); ++j)
                        {
                            propagatorId = inVertexPtr->inPropagatorId(j);
                            if (propagatorId[0] != iBond)
                            {
                                if (!isFinished(propagatorId[0], propagatorId[1]))
                                {
                                    isReady = false;
                                    break;
                                }
                            }
                        }
                        if (isReady)
                        {
                            propagatorIds_[nPropagator_][0] = iBond;
                            propagatorIds_[nPropagator_][1] = iDirection;
                            isFinished(iBond, iDirection) = true;
                            ++nPropagator_;
                        }
                    }
                }
            }
        }
    }

    template <class Bond>
    void DPolymerTmpl<Bond>::solve()
    {
        // Clear all propagators
        for (int j = 0; j < nPropagator(); ++j)
        {
            propagator(j).setIsSolved(false);
        }

        for (int j = 0; j < nPropagator(); ++j)
        {
            UTIL_CHECK(propagator(j).isReady())
            propagator(j).solve();
            // std::cout << propagatorIds_[j][0] << "   "
            //           << propagatorIds_[j][1] << "\n";
        }

        q_ = bond(0).propagator(0).computeQ();
        // std::cout << q_ << std::endl;
        // exit(1);

        double prefactor = 1.0 / (N() * q_);

        for (int i = 0; i < nBond(); ++i)
        {
            if (bond(i).bondtype() == 1)
                bond(i).computeConcentration(prefactor);
        }
    }

}

#endif
