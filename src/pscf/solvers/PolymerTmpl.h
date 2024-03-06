#ifndef PSCF_POLYMER_TMPL_H
#define PSCF_POLYMER_TMPL_H

/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include <pscf/chem/Species.h>         // base class
#include <util/param/ParamComposite.h> // base class

#include <pscf/chem/Monomer.h>      // member template argument
#include <pscf/chem/Vertex.h>       // member template argument
#include <util/containers/Pair.h>   // member template
#include <util/containers/DArray.h> // member template
#include <util/containers/DMatrix.h>

#include <pspg/math/GpuHeaders.h>

#include <cmath>

namespace Pscf
{

    class Block;
    using namespace Util;

    /**
     * Descriptor and MDE solver for an acyclic block polymer.
     *
     * A PolymerTmpl<Block> object has arrays of Block and Vertex
     * objects. Each Block has two propagator MDE solver objects.
     * The compute() member function solves the modified diffusion
     * equation (MDE) for the entire molecule and computes monomer
     * concentration fields for all blocks.
     *
     * \ingroup Pscf_Solver_Module
     */
    template <class Block>
    class PolymerTmpl : public Species, public ParamComposite
    {

    public:
        // Modified diffusion equation solver for one block.
        typedef typename Block::Propagator Propagator;

        // Monomer concentration field.
        typedef typename Propagator::CField CField;

        // Chemical potential field.
        typedef typename Propagator::WField WField;

        /**
         * Constructor.
         */
        PolymerTmpl();

        /**
         * Destructor.
         */
        ~PolymerTmpl();

        /**
         * Read and initialize.
         *
         * \param in input parameter stream
         */
        virtual void readParameters(std::istream &in);

        /**
         * Solve modified diffusion equation.
         *
         * Upon return, q functions and block concentration fields
         * are computed for all propagators and blocks.
         */
        virtual void solve();

        /// \name Accessors (objects, by reference)
        //@{

        /**
         * Get a specified Block.
         *
         * \param id block index, 0 <= id < nBlock
         */
        Block &block(int id);

        /**
         * Get a specified Block by const reference.
         *
         * \param id block index, 0 <= id < nBlock
         */
        Block const &block(int id) const;

        /**
         * Get a specified Vertex by const reference.
         *
         * Both chain ends and junctions are vertices.
         *
         * \param id vertex index, 0 <= id < nVertex
         */
        const Vertex &vertex(int id) const;

        /**
         * Get propagator for a specific block and direction.
         *
         * \param blockId integer index of associated block
         * \param directionId integer index for direction (0 or 1)
         */
        Propagator &propagator(int blockId, int directionId);

        /**
         * Get a const propagator for a specific block and direction.
         *
         * \param blockId integer index of associated block
         * \param directionId integer index for direction (0 or 1)
         */
        Propagator const &propagator(int blockId, int directionId) const;

        /**
         * Get propagator indexed in order of computation.
         *
         * The propagator index must satisfy 0 <= id < 2*nBlock.
         *
         * \param id integer index, in order of computation plan
         */
        Propagator &propagator(int id);

        /**
         * Propagator identifier, indexed by order of computation.
         *
         * The return value is a pair of integers. The first of
         * which is a block index between 0 and nBlock - 1 and
         * the second is a direction id, which must be 0 or 1.
         */
        const Pair<int> &propagatorId(int i) const;

        //@}
        /// \name Accessors (by value)
        //@{

        /**
         * Number of blocks.
         */
        int nBlock() const;

        /**
         * Number of vertices (junctions and chain ends).
         */
        int nVertex() const;

        /**
         * Number of propagators (twice nBlock).
         */
        int nPropagator() const; //

        /**
         * Total length of all blocks = volume / reference volume.
         */
        double length() const;

        double Q() const;

        //@}

    protected:
        virtual void makePlan();

    private:
        /// Array of Block objects in this polymer.
        DArray<Block> blocks_;

        /// Array of Vertex objects in this polymer.
        DArray<Vertex> vertices_;

        /// Propagator ids, indexed in order of computation.
        DArray<Pair<int>> propagatorIds_;

        /// Number of blocks in this polymer
        int nBlock_;

        /// Number of vertices (ends or junctions) in this polymer
        int nVertex_;

        /// Number of propagators (two per block).
        int nPropagator_;

        double Q_;
    };

    template <class Block>
    inline double PolymerTmpl<Block>::Q() const
    {
        return Q_;
    }


    /*
     * Number of vertices (ends and/or junctions)
     */
    template <class Block>
    inline int PolymerTmpl<Block>::nVertex() const
    {
        return nVertex_;
    }

    /*
     * Number of blocks.
     */
    template <class Block>
    inline int PolymerTmpl<Block>::nBlock() const
    {
        return nBlock_;
    }

    /*
     * Number of propagators.
     */
    template <class Block>
    inline int PolymerTmpl<Block>::nPropagator() const
    {
        return nPropagator_;
    }

    /*
     * Total length of all blocks = volume / reference volume
     */
    template <class Block>
    inline double PolymerTmpl<Block>::length() const
    {
        double value = 0.0;
        for (int blockId = 0; blockId < nBlock_; ++blockId)
        {
            value += blocks_[blockId].length();
        }
        return value;
    }

    /*
     * Get a specified Vertex.
     */
    template <class Block>
    inline const Vertex &PolymerTmpl<Block>::vertex(int id) const
    {
        return vertices_[id];
    }

    /*
     * Get a specified Block.
     */
    template <class Block>
    inline Block &PolymerTmpl<Block>::block(int id)
    {
        return blocks_[id];
    }

    /*
     * Get a specified Block by const reference.
     */
    template <class Block>
    inline Block const &PolymerTmpl<Block>::block(int id) const
    {
        return blocks_[id];
    }

    /*
     * Get a propagator id, indexed in order of computation.
     */
    template <class Block>
    inline Pair<int> const &PolymerTmpl<Block>::propagatorId(int id) const
    {
        UTIL_CHECK(id >= 0)
        UTIL_CHECK(id < nPropagator_)
        return propagatorIds_[id];
    }

    /*
     * Get a propagator indexed by block and direction.
     */
    template <class Block>
    inline
        typename Block::Propagator &
        PolymerTmpl<Block>::propagator(int blockId, int directionId)
    {
        return block(blockId).propagator(directionId);
    }

    /*
     * Get a const propagator indexed by block and direction.
     */
    template <class Block>
    inline
        typename Block::Propagator const &
        PolymerTmpl<Block>::propagator(int blockId, int directionId) const
    {
        return block(blockId).propagator(directionId);
    }

    /*
     * Get a propagator indexed in order of computation.
     */
    template <class Block>
    inline
        typename Block::Propagator &
        PolymerTmpl<Block>::propagator(int id)
    {
        Pair<int> propId = propagatorId(id);
        return propagator(propId[0], propId[1]);
    }

    // Non-inline functions

    /*
     * Constructor.
     */
    template <class Block>
    PolymerTmpl<Block>::PolymerTmpl()
        : Species(),
          blocks_(),
          vertices_(),
          propagatorIds_(),
          nBlock_(0),
          nVertex_(0),
          nPropagator_(0)
    {
        setClassName("PolymerTmpl");
    }

    /*
     * Destructor.
     */
    template <class Block>
    PolymerTmpl<Block>::~PolymerTmpl() = default;

    template <class Block>
    void PolymerTmpl<Block>::readParameters(std::istream &in)
    {
        read<int>(in, "nBlock", nBlock_);
        read<int>(in, "nVertex", nVertex_);

        // Allocate all arrays
        blocks_.allocate(nBlock_);
        vertices_.allocate(nVertex_);
        propagatorIds_.allocate(2 * nBlock_);

        readDArray<Block>(in, "blocks", blocks_, nBlock_);

        // Set vertex indices
        for (int vertexId = 0; vertexId < nVertex_; ++vertexId)
        {
            vertices_[vertexId].setId(vertexId);
        }

        // Add blocks to vertices
        int vertexId0, vertexId1;
        Block *blockPtr;
        for (int blockId = 0; blockId < nBlock_; ++blockId)
        {
            blockPtr = &(blocks_[blockId]);
            vertexId0 = blockPtr->vertexId(0);
            vertexId1 = blockPtr->vertexId(1);
            vertices_[vertexId0].addBlock(*blockPtr);
            vertices_[vertexId1].addBlock(*blockPtr);
        }

        makePlan();

        // Read ensemble and phi or mu
        ensemble_ = Species::Closed;
        readOptional<Species::Ensemble>(in, "ensemble", ensemble_);
        if (ensemble_ == Species::Closed)
        {
            read(in, "phi", phi_);
        }
        else
        {
            read(in, "mu", mu_);
        }
        // std::cout << "phi = " << phi_ <<"\n";
        // std::cout << "mu  = " << mu_ <<"\n";

        // exit(1);
        // Set sources for all propagators
        Vertex const *vertexPtr = nullptr;
        Propagator const *sourcePtr = 0;
        Propagator *propagatorPtr = 0;
        Pair<int> propagatorId;
        int blockId, directionId, vertexId, i;
        for (blockId = 0; blockId < nBlock(); ++blockId)
        {
            // Add sources
            for (directionId = 0; directionId < 2; ++directionId)
            {
                vertexId = block(blockId).vertexId(directionId);
                vertexPtr = &vertex(vertexId);
                propagatorPtr = &block(blockId).propagator(directionId);
                for (i = 0; i < vertexPtr->size(); ++i)
                {
                    propagatorId = vertexPtr->inPropagatorId(i);
                    if (propagatorId[0] == blockId)
                    {
                        UTIL_CHECK(propagatorId[1] != directionId)
                    }
                    else
                    {
                        sourcePtr =
                            &block(propagatorId[0]).propagator(propagatorId[1]);
                        propagatorPtr->addSource(*sourcePtr);
                    }
                }
            }
        }
    }

    template <class Block>
    void PolymerTmpl<Block>::makePlan()
    {
        if (nPropagator_ != 0)
        {
            UTIL_THROW("nPropagator !=0 on entry");
        }

        // Allocate and initialize isFinished matrix
        DMatrix<bool> isFinished;
        isFinished.allocate(nBlock_, 2);
        for (int iBlock = 0; iBlock < nBlock_; ++iBlock)
        {
            for (int iDirection = 0; iDirection < 2; ++iDirection)
            {
                isFinished(iBlock, iDirection) = false;
            }
        }

        Pair<int> propagatorId;
        Vertex *inVertexPtr = nullptr;
        int inVertexId = -1;
        bool isReady;
        while (nPropagator_ < nBlock_ * 2)
        {
            for (int iBlock = 0; iBlock < nBlock_; ++iBlock)
            {
                for (int iDirection = 0; iDirection < 2; ++iDirection)
                {
                    if (!isFinished(iBlock, iDirection))
                    {
                        inVertexId = blocks_[iBlock].vertexId(iDirection);
                        inVertexPtr = &vertices_[inVertexId];
                        isReady = true;
                        for (int j = 0; j < inVertexPtr->size(); ++j)
                        {
                            propagatorId = inVertexPtr->inPropagatorId(j);
                            if (propagatorId[0] != iBlock)
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
                            propagatorIds_[nPropagator_][0] = iBlock;
                            propagatorIds_[nPropagator_][1] = iDirection;
                            isFinished(iBlock, iDirection) = true;
                            ++nPropagator_;
                        }
                    }
                }
            }
        }
    }

    /*
     * Compute solution to MDE and concentrations.
     */
    template <class Block>
    void PolymerTmpl<Block>::solve()
    {
        int nx = blocks_[0].mesh().size();
        int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
        Pspg::ThreadGrid::setThreadsLogical(nx, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
        // Clear all propagators
        for (int j = 0; j < nPropagator(); ++j)
        {
            propagator(j).setIsSolved(false);
        }

        DArray<Pair<int>> t;
        int partner[nBlock()];
        t.allocate(nBlock());
        for (int i = 0; i < nBlock(); ++i)
        {
            for (int j = 0; j < nBlock(); ++j)
            {
                if (i == propagatorIds_[j][0])
                {
                    t[i][0] = j;
                    // std::cout << t[i][0] << "\n";
                }
            }
        }
        // std::cout << "\n";
        for (int i = 0; i < nBlock(); ++i)
        {
            for (int j = nBlock(); j < nPropagator(); ++j)
            {
                if (i == propagatorIds_[j][0])
                {
                    t[i][1] = j;
                    // std::cout << t[i][0] << "\n";
                }
            }
        }
        for (int j = 0; j < nBlock(); ++j)
        {
            partner[t[j][1] - nBlock()] = t[j][0];
            // std::cout << t[j][1]-nBlock() << " " << t[j][0] << "\n";
        }
        t.deallocate();

        for (int j = 0; j < nBlock(); ++j)
        {
            UTIL_CHECK(propagator(j).isReady())
            propagator(j).solveForward();
        }

        for (int j = nBlock(); j < nPropagator(); ++j)
        {
            UTIL_CHECK(propagator(j).isReady())
            propagator(j).solveBackward(propagator(partner[j - nBlock()]).qhead(), j - nBlock());
        }

        Q_ = propagator(nBlock()).returnQ();
        
        double scale = 1.0 / Q_;

        for (int j = 0; j < nBlock(); ++j)
        {
            Pspg::scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(blocks_[j].cField().cDField(), scale, nx);
        }

        if (ensemble() == Species::Closed)
        {
            mu_ = log(phi_ / Q_);
        }
        else if (ensemble() == Species::Open)
        {
            phi_ = exp(mu_) * Q_;
        }
    }

   
}

#endif
