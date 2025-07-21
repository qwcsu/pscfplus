#ifndef PGC_MIXTURE_H
#define PGC_MIXTURE_H

/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include "Polymer.h"
#include "Solvent.h"
#include <pscf/solvers/MixtureTmpl.h>
#include <pscf/inter/Interaction.h>
#include <util/containers/DArray.h>
#include <util/containers/GArray.h>
#include <pscf/crystal/Basis.h>
#include <pscf/inter/ChiInteraction.h>

namespace Pscf
{
    template <int D>
    class Mesh;
}

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            /**
             * Solver for a mixture of polymers and solvents.
             *
             * A Mixture contains a list of Polymer and Solvent objects. Each
             * such object can solve the single-molecule statistical mechanics
             * problem for an ideal gas of the associated species in a set of
             * specified chemical potential fields, and thereby compute
             * concentrations and single-molecule partition functions. A
             * Mixture is thus both a chemistry descriptor and an ideal-gas
             * solver.
             *
             * A Mixture is associated with a Mesh<D> object, which models a
             * spatial discretization mesh.
             *
             * \ingroup Pscf_Continuous_Solvers_Module
             */
            template <int D>
            class Mixture : public MixtureTmpl<Polymer<D>, Solvent<D>>
            {

            public:
                // Public typedefs

                /**
                 * Monomer chemical potential field type.
                 */
                typedef typename Propagator<D>::WField WField;

                /**
                 * Monomer concentration or volume fraction field type.
                 */
                typedef typename Propagator<D>::CField CField;

                // Public member functions

                /**
                 * Constructor.
                 */
                Mixture();

                /**
                 * Destructor.
                 */
                ~Mixture();

                /**
                 * Read all parameters and initialize.
                 *
                 * This function reads in a complete description of
                 * the chemical composition and structure of all species,
                 * as well as the target contour length step size ds.
                 *
                 * \param in input parameter stream
                 */
                void readParameters(std::istream &in);

                void setInteraction(ChiInteraction &interaction);

                /**
                 * Create an association with the mesh and allocate memory.
                 *
                 * The Mesh<D> object must have already been initialized,
                 * e.g., by reading its parameters from a file, so that the
                 * mesh dimensions are known on entry.
                 *
                 * \param mesh associated Mesh<D> object (stores address).
                 */
                void setMesh(Mesh<D> const &mesh,
                             UnitCell<D> &unitCell);

                void setU0(UnitCell<D> &unitCell,
                           WaveList<D> &wavelist);
                /**
                 * Set unit cell parameters used in solver.
                 *
                 * \param unitCell UnitCell<D> object that contains Bravais lattice.
                 */
                void setupUnitCell(const UnitCell<D> &unitCell, const WaveList<D> &wavelist);

                /**
                 * Compute concentrations.
                 *
                 * This function calls the compute function of every molecular
                 * species, and then adds the resulting block concentration
                 * fields for blocks of each type to compute a total monomer
                 * concentration (or volume fraction) for each monomer type.
                 * Upon return, values are set for volume fraction and chemical
                 * potential (mu) members of each species, and for the
                 * concentration fields for each Block and Solvent. The total
                 * concentration for each monomer type is returned in the
                 * cFields output parameter.
                 *
                 * The arrays wFields and cFields must each have size nMonomer(),
                 * and contain fields that are indexed by monomer type index.
                 *
                 * \param wFields array of chemical potential fields (input)
                 * \param cFields array of monomer concentration fields (output)
                 */
                void
                compute(DArray<WField> const &wFields, DArray<CField> &cFields);

                /**
                 * Get monomer reference volume.
                 */
                void computeStress(WaveList<D> &wavelist);
#if CMP == 1
                void computeBlockCMP(Mesh<D> const &mesh, 
                                     DArray<CField> cFieldsRGrid, 
                                     DArray<RDFieldDft<D>> cFieldsKGrid,
                                     FFT<D> & fft);
#endif
                void computeBlockRepulsion(Mesh<D> const &mesh, 
                                           DArray<CField> cFieldsRGrid, 
                                           DArray<RDFieldDft<D>> cFieldsKGrid,
                                           FFT<D> & fft);

                void computeBlockEntropy(DArray<RDField<D>> const &wFields,
                                         Mesh<D> const &mesh);

                double sigma();
#if CMP == 1
                double kpN();
#endif
                RDField<D> &bu0();

                /**
                 * Get derivative of free energy w/ respect to cell parameter.
                 *
                 * Get precomputed value of derivative of free energy per monomer
                 * with respect to unit cell parameter number n.
                 *
                 * \int n unit cell parameter id
                 */
                double stress(int n)
                {
                    return stress_[n];
                }

                int nParameter()
                {
                    return nParams_;
                }

                int nUCompChi()
                {
                    return nUCompChi_;
                }
#if CMP==1
                int nUCompCMP()
                {
                    return nUCompCMP_;
                }
#endif
                /**
                 * Get monomer reference volume.
                 */
                double vMonomer() const;

                cudaReal sBlock(int n);

                cudaReal uBlockRepulsion(int n);
#if CMP==1
                cudaReal uBlockCMP(int id);
#endif
                DArray<int> uBlockList(int n);

                using MixtureTmpl<Pscf::Pspg::Continuous::Polymer<D>,
                                  Pscf::Pspg::Solvent<D>>::nMonomer;
                using MixtureTmpl<Pscf::Pspg::Continuous::Polymer<D>,
                                  Pscf::Pspg::Solvent<D>>::nPolymer;
                using MixtureTmpl<Pscf::Pspg::Continuous::Polymer<D>,
                                  Pscf::Pspg::Solvent<D>>::nSolvent;
                using MixtureTmpl<Pscf::Pspg::Continuous::Polymer<D>,
                                  Pscf::Pspg::Solvent<D>>::polymer;

            protected:
                using MixtureTmpl<Pscf::Pspg::Continuous::Polymer<D>,
                                  Pscf::Pspg::Solvent<D>>::setClassName;
                using ParamComposite::read;
                using ParamComposite::readOptional;

            private:
                /// Derivatives of free energy w/ respect to cell parameters.
                FArray<double, 6> stress_;

                /// Monomer reference volume (set to 1.0 by default).
                double vMonomer_;

                /// Optimal contour length step size.
                double ds_;

                int ns_;

                /// Number of unit cell parameters.
                int nParams_;

                int rSize_;

                int kSize_;

                int nUCompChi_;
#if CMP==1
                int nUCompCMP_;
#endif
                /// Pointer to associated Mesh<D> object.
                Mesh<D> const *meshPtr_;

                ChiInteraction *interaction_;

                /// Return associated domain by reference.
                Mesh<D> const &mesh() const;

                DArray<RDFieldDft<D>> cFieldsKGrid_;

                double sigma_;
#if CMP == 1
                double kappa_;
#endif
                RDField<D> bu0_;

                RDField<D> dbu0_;

                cudaReal *dbu0_host;

                cudaReal *bu0_host;

                cudaReal *Chiphi_;

                cudaReal *mStress_;

                cudaReal *d_temp_;

                cudaReal *temp_;

                Basis<D> *basisPtr_;

                IntVec<D> kMeshDimensions_;

                DArray<cudaReal> sBlock_;

                GArray<DArray<int>> blockIds_;

                GArray<cudaReal> uBlockChi_;
                GArray<cudaReal> uBlockCMP_;
            };

            // Inline member function

            /*
             * Get monomer reference volume (public).
             */
            template <int D>
            inline double Mixture<D>::vMonomer() const
            {
                return vMonomer_;
            }

            /*
             * Get Mesh<D> by constant reference (private).
             */
            template <int D>
            inline Mesh<D> const &Mixture<D>::mesh() const
            {
                UTIL_ASSERT(meshPtr_)
                return *meshPtr_;
            }

            template <int D>
            inline RDField<D> &Mixture<D>::bu0()
            {
                return bu0_;
            }

            template <int D>
            inline double Mixture<D>::sigma()
            {
                return sigma_;
            }

            template <int D>
            inline cudaReal Mixture<D>::sBlock(int id)
            {
                return sBlock_[id];
            }

            template <int D>
            inline cudaReal Mixture<D>::uBlockRepulsion(int id)
            {
                return uBlockChi_[id];
            }
#if CMP==1
            template <int D>
            inline cudaReal Mixture<D>::uBlockCMP(int id)
            {
                return uBlockCMP_[id];
            }
#endif
            template <int D>
            inline DArray<int> Mixture<D>::uBlockList(int id)
            {
                return blockIds_[id];
            }
#if CMP == 1
            template <int D>
            inline double Mixture<D>::kpN()
            {
                return kappa_;
            }
#endif
#ifndef PSPG_MIXTURE_TPP
            // Suppress implicit instantiation
            extern template class Mixture<1>;
            extern template class Mixture<2>;
            extern template class Mixture<3>;
#endif

        }
    } // namespace Pspg
} // namespace Pscf
// #include "Mixture.tpp"
#endif
