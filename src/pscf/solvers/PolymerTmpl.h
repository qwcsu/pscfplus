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

        virtual void reduce();

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

        const DArray<GArray<int>> mapping () const;

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

        DArray<GArray<int>> blockMapping_;

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

        DArray<GArray<int>> propDependence_;
        DArray<int> propReplace_;
        DArray<Pair<int>> propPartners_;
        std::vector<Pair<int>> propReduced_;
        DArray<GArray<int>> propMapping_;
        DArray<int> blockMonomer_;
        DArray<double> blockLength_;
        int nPropReduced_;
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

    template <class Block>
    inline 
    const 
    DArray<GArray<int>> PolymerTmpl<Block>::mapping() const
    {
        return propMapping_;
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
        blockMonomer_.allocate(2 * nBlock_);
        blockLength_.allocate(2 * nBlock_);
        propReplace_.allocate(2 * nBlock_);
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
#if CMP == 1
        UTIL_CHECK(ensemble() == Species::Closed)
#endif
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
        int blockId, directionId, vertexId;
        for (blockId = 0; blockId < nBlock(); ++blockId)
        {
            // Add sources
            for (directionId = 0; directionId < 2; ++directionId)
            {
                vertexId = block(blockId).vertexId(directionId);
                vertexPtr = &vertex(vertexId);
                propagatorPtr = &block(blockId).propagator(directionId);
                int order = propagatorPtr->order();
                blockMonomer_[order] = block(blockId).monomerId();
                blockLength_[order] = block(blockId).length();
                for (int i = 0; i < vertexPtr->size(); ++i)
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
                        int source = sourcePtr->order();
                        // propagatorPtr->addSource(*sourcePtr);
                        propDependence_[order].append(source);
                    }
                }
            }
        }

        for (int i = 0; i < nPropagator_; ++i)
        {
            propReplace_[i] = -1;
        }
        for (int i = 0; i < nPropagator_; ++i)
        {
            for (int j = 0; j < nPropagator_; ++j)
            {
                int size1 = propDependence_[i].size();
                int size2 = propDependence_[j].size();
                int m1 = blockMonomer_[i];
                int m2 = blockMonomer_[j];
                double l1 = blockLength_[i];
                double l2 = blockLength_[j];
                if (size1 == size2 &&
                    m1 == m2 &&
                    l1 == l2)
                {
                    if (size1 == 0)
                    {
                        if (propReplace_[j] == -1)
                            propReplace_[j] = i;
                        for (int k = 0; k < nPropagator_; ++k)
                        {
                            for (int s = 0; s < propDependence_[k].size(); ++s)
                            {
                                if (propDependence_[k][s] == j)
                                    propDependence_[k][s] = propReplace_[j];
                            }
                        }
                    }
                    else 
                    {
                        bool depFlag = true;
                        for (int l = 0; l < size1; ++l)
                        {
                            if (propDependence_[i][l] != propDependence_[j][l])
                            {
                                depFlag = false;
                                break;
                            } 
                        }
                        if (depFlag == true)
                        {
                            if (propReplace_[j] == -1)
                                propReplace_[j] = i;
                            for (int k = 0; k < nPropagator_; ++k)
                            {
                                for (int s = 0; s < propDependence_[k].size(); ++s)
                                {
                                    if (propDependence_[k][s] == j)
                                        propDependence_[k][s] = propReplace_[j];
                                }
                            }
                        }
                    }

                }
            }
        }
        // for (int i = 0; i < nPropagator_; ++i)
        // {
        //     std::cout << i << ", " 
        //               << blockMonomer_[i] << ", " 
        //               << blockLength_[i] << ": ";
        //     for (int j = 0; j < propDependence_[i].size(); ++j)
        //     {
        //         std::cout << propDependence_[i][j] << "   ";
        //     }
        //     std::cout << "Replaced by " << propReplace_[i] << "   ";
        //     std::cout << std::endl;
        // }

        nPropReduced_ = 0;
        for (int i = 0; i < nPropagator_; ++i)
        {
            for (int j = 0; j < nPropagator_; ++j)
            {
                if (propReplace_[j] == i)
                {
                    ++nPropReduced_;
                    break;
                }
            }
        }

        if (!propMapping_.isAllocated())
            propMapping_.allocate(nPropagator_);

        for (int i = 0; i < nPropagator_; ++i)
        {
            for (int j = 0; j < nPropagator_; ++j)
            {                
                if (propReplace_[j] == i)
                {
                    propMapping_[i].append(j);
                }
            }
        }
        // std::cout << nPropReduced_ << std::endl;
        // for (int i = 0; i < nPropagator_; ++i)
        // {
        //     std::cout << i << ":   ";
        //     for (int j = 0; j < propMapping_[i].size(); ++j)
        //     {
        //         std::cout << propMapping_[i][j] << "   ";
        //     }
        //     std::cout << std::endl;
        // }

        for (int i = 0; i < nPropagator_; ++i)
        {
            for (int j = 0; j < propDependence_[i].size(); ++j)
            {
                // std::cout << i << ": " << propDependence_[i][j] << "\n";
                propagator(i).addSource(propagator(propDependence_[i][j]));
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
        if (!propDependence_.isAllocated())
            propDependence_.allocate(2*nBlock_);

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
        Propagator *propagatorPtr = 0;
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
                            propagatorPtr = &block(iBlock).propagator(iDirection);
                            propagatorPtr->setOrder(nPropagator_);
                            ++nPropagator_;
                        }
                    }
                }
            }
        }
        for (int iBlock = 0; iBlock < nBlock_; ++iBlock)
        {
            if (block(iBlock).propagator(0).order() < block(iBlock).propagator(1).order())
            {
                block(iBlock).propagator(0).setDirectionFlag(0);
                block(iBlock).propagator(1).setDirectionFlag(1);
            }
            else
            {
                block(iBlock).propagator(0).setDirectionFlag(1);
                block(iBlock).propagator(1).setDirectionFlag(0);
            }
            // std::cout << block(iBlock).propagator(0).order() << "(" << block(iBlock).propagator(0).directionFlag() << ")   ";
            // std::cout << block(iBlock).propagator(1).order() << "(" << block(iBlock).propagator(1).directionFlag() << ")\n";
        }
    }

    template <class Block>
    void PolymerTmpl<Block>::reduce()
    {
        if (!propPartners_.isAllocated())
            propPartners_.allocate(nPropagator_);
        for (int i = 0; i < nPropagator_; ++i)
        {
            propPartners_[i][0] = propReplace_[i];
            if (propagator(i).partner().isAllocated())
                propPartners_[i][1] = propagator(i).partner().order();
            else   
                propPartners_[i][1] = propagator(propagator(i).partner().order()).ref().order();

            int tmp;
            if  (propPartners_[i][0] > propPartners_[i][1])
            {
                tmp = propPartners_[i][0];
                propPartners_[i][0] = propPartners_[i][1];
                propPartners_[i][1] = tmp; 
            }
            // std::cout << propPartners_[i] << "\n";
        }

        bool insertFlag;
        for (int i = 0; i < nPropagator_; ++i)
        {
            insertFlag = true;
            for (int j = 0; j < propReduced_.size(); ++j)
            {
                if (propReduced_[j][0] == propPartners_[i][0] && propReduced_[j][1] == propPartners_[i][1])
                {
                    insertFlag = false;
                    break;
                }
            }
            if (insertFlag)
                propReduced_.insert(propReduced_.end(), propPartners_[i]);
        }

        if (!blockMapping_.isAllocated())
            blockMapping_.allocate(propReduced_.size());
        for (int j = 0; j < propReduced_.size(); ++j)
        { 
            for (int i = 0; i < propMapping_[propReduced_[j][0]].size(); ++i)
            {
                blockMapping_[j].append(propagator(propMapping_[propReduced_[j][0]][i]).block().id());
            }
        }
        // for (int j = 0; j < propReduced_.size(); ++j)
        // { 
        //     for (int i = 0; i < propMapping_[propReduced_[j][0]].size(); ++i)
        //     {
        //         std::cout << blockMapping_[j][i] << " ";
        //     }
        //     std::cout << "\n";
        // }
        for (int j = 0; j < nBlock(); ++j)
        {
            // std::cout << "block " << j << " : ";
            // if (blocks_[j].propagator(0).isAllocated())
            //     std::cout << " p0 =  " << blocks_[j].propagator(0).order() << " and ";
            // else 
            //     std::cout << " p0 =  " << blocks_[j].propagator(0).ref().order() << " and ";
            // if (blocks_[j].propagator(1).isAllocated())
            //     std::cout << " p1 =  " << blocks_[j].propagator(1).order() << " \n ";
            // else
            //     std::cout << " p1 =  " << blocks_[j].propagator(1).ref().order() << " \n ";

            if (!blocks_[j].propagator(0).isAllocated())
            {
                blocks_[j].propagator(0).setPropagator(blocks_[j].propagator(0).ref(), 
                                                       blocks_[j].propagator(0).ref().order());
                propagator(blocks_[j].propagator(0).ref().order()).setReused(true);        
                // std::cout << propagator(blocks_[j].propagator(0).ref().order()).isReused() << "\n";                 
            }
                

            if (!blocks_[j].propagator(1).isAllocated())
            {
                blocks_[j].propagator(1).setPropagator(blocks_[j].propagator(1).ref(),
                                                       blocks_[j].propagator(1).ref().order());
                propagator(blocks_[j].propagator(1).ref().order()).setReused(true);   
                // std::cout << propagator(blocks_[j].propagator(1).ref().order()).isReused() << "\n";         
            }
        }
        // exit(1);
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
            
        for (int j = 0; j < propReduced_.size(); ++j)
        {
            propagator(propReduced_[j][0]).solveForward();
            //     
            for (int i = 1; i < propMapping_[propReduced_[j][0]].size(); ++i)
            {
                // std::cout << propMapping_[propReduced_[j][0]][i] << "\n";
                propagator(propMapping_[propReduced_[j][0]][i]).setIsSolved(true);
            }
        }
        
        
        for (int j = propReduced_.size()-1; j >= 0; --j)
        {
            bool isReused = propagator(propReduced_[j][0]).isReused();
            propagator(propReduced_[j][1]).solveBackward(propagator(propReduced_[j][0]).qhead(), isReused);
            for (int i = 1; i < propMapping_[propReduced_[j][1]].size(); ++i)
            {
                // std::cout << propMapping_[propReduced_[j][1]][i] << "\n";
                propagator(propMapping_[propReduced_[j][1]][i]).setIsSolved(true);
                // std::cout << "block " << propagator(propMapping_[propReduced_[j][1]][i]).block() 
                //           << " has not been calculated, which should be replaced by "
                //           << propagator(propReduced_[j][1]).block() << "\n"; 
                propagator(propMapping_[propReduced_[j][1]][i]).block().setField(propagator(propReduced_[j][1]).block().cField());
            }
        }
        // exit(1);
        Q_ = propagator(propReduced_[0][1]).returnQ();
        // std::cout << Q_ << "\n";


        // exit(1);
        
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
        // std::cout << phi_ << "\n";
        // std::cout << mu_ << "\n";
        // exit(1);
    }

   
}

#endif
