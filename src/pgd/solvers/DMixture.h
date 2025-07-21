#ifndef D_MIXTURE_H
#define D_MIXTURE_H

#include "DPolymer.h"
#include <pgc/solvers/Solvent.h>
#include <pgd/solvers/DMixtureTmpl.h>
#include <pscf/inter/Interaction.h>
#include <util/containers/DArray.h>
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
        namespace Discrete
        {

            /**
             * Solver for a mixture of polymers (Discrete chain model).
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
             * \ingroup Pscf_Discrete_Solvers_Module
             */
            template <int D>
            class DMixture : public DMixtureTmpl<DPolymer<D>, Solvent<D>>
            {
            public:
                typedef typename DPropagator<D>::WField WField;

                typedef typename DPropagator<D>::CField CField;

                DMixture();

                ~DMixture() = default;

                void readParameters(std::istream &in);

                void setInteraction(ChiInteraction &interaction);

                void setMesh(Mesh<D> const &mesh,
                             UnitCell<D> &unitCell);

                void setBasis(Basis<D> &basis);

                void setU0(UnitCell<D> &unitCell,
                           WaveList<D> &wavelist);

                void setupUnitCell(const UnitCell<D> &unitCell,
                                   const WaveList<D> &wavelist);

                void compute(DArray<WField> const &wFields,
                             DArray<CField> &cFields);

                void computeStress(WaveList<D> &wavelist);

#if CMP == 1
                void computeBlockCMP(Mesh<D> const &mesh, 
                                     DArray<CField> cFieldsRGrid, 
                                     DArray<RDFieldDft<D>> cFieldsKGrid,
                                     FFT<D> & fft);
#endif
                void computeBlockRepulsion(Mesh<D> const &mesh, 
                                           DArray<RDField<D>> cFieldsRGrid, 
                                           DArray<RDFieldDft<D>> cFieldsKGrid,
                                           FFT<D> & fft);
                
                void computeBlockEntropy(DArray<RDField<D>> const &wFields,
                                         Mesh<D> const &mesh);

                RDField<D> &bu0();
                
                RDField<D> &bu0k();

                RDField<D> &dbu0();
                
                RDField<D> &dbu0K();

                double kpN();

                void setKpN(double kpN);

                double sigma();

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

                cudaReal uBlockRepulsion(int n);
#if CMP==1
                cudaReal uBlockCMP(int id);
#endif

                using DMixtureTmpl<Pscf::Pspg::Discrete::DPolymer<D>,
                                   Pscf::Pspg::Solvent<D>>::nPolymer;

                using DMixtureTmpl<Pscf::Pspg::Discrete::DPolymer<D>,
                                   Pscf::Pspg::Solvent<D>>::nSolvent;

                using DMixtureTmpl<Pscf::Pspg::Discrete::DPolymer<D>,
                                   Pscf::Pspg::Solvent<D>>::nMonomer;

                using DMixtureTmpl<Pscf::Pspg::Discrete::DPolymer<D>,
                                   Pscf::Pspg::Solvent<D>>::polymer;

                using DMixtureTmpl<Pscf::Pspg::Discrete::DPolymer<D>,
                                   Pscf::Pspg::Solvent<D>>::monomer;

            protected:
                using DMixtureTmpl<Pscf::Pspg::Discrete::DPolymer<D>,
                                   Pscf::Pspg::Solvent<D>>::setClassName;
                using ParamComposite::read;
                using ParamComposite::readOptional;

            private:

                double sigma_;

                double kappa_;

                Mesh<D> const *meshPtr_;

                ChiInteraction *interaction_;

                Mesh<D> const &mesh() const;

                int nParams_;

                FArray<double, 6> stress_;

                Basis<D> *basisPtr_;

                IntVec<D> kMeshDimensions_;

                int rSize_;

                int kSize_;

                int nUCompChi_;
#if CMP==1
                int nUCompCMP_;
#endif

                RDField<D> bu0_;

                RDField<D> bu0k_;

                RDField<D> dbu0_;

                RDField<D> dbu0K_;

                DArray<RDFieldDft<D>> cFieldsKGrid_;

                cudaReal *dbu0_host;

                cudaReal *bu0_host;

                cudaReal *bu0k_host;

                cudaReal *Chiphi_;

                cudaComplex *Kappaphi1_;

                cudaComplex *Kappaphi2_;

                cudaReal *mStress_;

                cudaReal *d_temp_;
                cudaReal *temp_;

                GArray<DArray<int>> blockIds_;

                GArray<cudaReal> uBlockChi_;
                GArray<cudaReal> uBlockCMP_;
            };

            template <int D>
            inline Mesh<D> const &DMixture<D>::mesh() const
            {
                UTIL_ASSERT(meshPtr_)
                return *meshPtr_;
            }

            template <int D>
            inline RDField<D> &DMixture<D>::bu0()
            {
                return bu0_;
            }

            template <int D>
            inline RDField<D> &DMixture<D>::bu0k()
            {
                return bu0k_;
            }

            template <int D>
            inline RDField<D> &DMixture<D>::dbu0()
            {
                return dbu0_;
            }

            template <int D>
            inline RDField<D> &DMixture<D>::dbu0K()
            {
                return dbu0K_;
            }

            template <int D>
            inline double DMixture<D>::kpN()
            {
                return kappa_;
            }

            template <int D>
            inline double DMixture<D>::sigma()
            {
                return sigma_;
            }

            template <int D>
            inline cudaReal DMixture<D>::uBlockRepulsion(int id)
            {
                return uBlockChi_[id];
            }
#if CMP==1
            template <int D>
            inline cudaReal DMixture<D>::uBlockCMP(int id)
            {
                return uBlockCMP_[id];
            }
#endif
#ifndef D_MIXTURE_TPP
            // Suppress implicit instantiation
            extern template class DMixture<1>;
            extern template class DMixture<2>;
            extern template class DMixture<3>;
#endif
        }
    }
}

#endif