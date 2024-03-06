#ifndef D_SYSTEM_H
#define D_SYSTEM_H

#include <pspg/field/FieldIo.h>
#include <pgd/iterator/AmIterator.h>
#include <pgd/solvers/DMixture.h>
#include <pscf/inter/ChiInteraction.h>
#include <pscf/crystal/Basis.h>
#include <util/param/ParamComposite.h> // base class
#include <util/misc/FileMaster.h>      // member
#include <util/containers/DArray.h>    // member template
#include <util/containers/Array.h>     // function parameter
#include <jsoncpp/json/json.h>

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            using namespace Util;

            template <int D>
            class System : public ParamComposite
            {
            public:
                typedef RDField<D> Field;

                typedef typename DPropagator<D>::WField WField;

                typedef typename DPropagator<D>::CField CField;

                System();

                ~System();

                void setOptions(int argc, char **argv);

                void setOptionsOutside(char *pArg, char *cArg, bool echo, int s = 1);

                void readParam();

                void readCommands();

                virtual void readParam(std::istream &in);

                virtual void readParameters(std::istream &in);

                void readCommands(std::istream &in);

                void fieldIO(std::string io,
                             std::string type,
                             std::string format,
                             std::string dir,
                             std::string caseid,
                             std::string prefix);

                void iterate();

                void readCommandsJson(int = 1);

                void readCommandsJson(std::string filename, int s);

                /// Accessors (access objects by reference)
                FileMaster &fileMaster();

                DMixture<D> &dmixture();

                ChiInteraction &interaction();

                AmIterator<D> &iterator();

                Mesh<D> &mesh();

                std::string groupName();

                UnitCell<D> &unitCell();

                WaveList<D> &wavelist();

                Basis<D> &basis();

                FieldIo<D> &fieldIo();

                FFT<D> &fft();

                void computeFreeEnergy();

                void outputThermo(std::ostream &out);

                void outputThermo(Json::Value &thermo);

                DArray<Field> &wFields();

                Field &wField(int monomerId);

                DArray<WField> &wFieldsRGrid();

                WField &wFieldRGrid(int monomerId);

                DArray<RDFieldDft<D>> &wFieldsKGrid();

                RDFieldDft<D> &wFieldKGrid(int monomerId);

                DArray<Field> &cFields();

                Field &cField(int monomerId);

                DArray<CField> &cFieldsRGrid();

                CField &cFieldRGrid(int monomerId);

                DArray<RDFieldDft<D>> &cFieldsKGrid();

                RDFieldDft<D> &cFieldKGrid(int monomerId);

                int systemId();

                double c(int monomerId);

                bool hasWFields() const;
                bool hasCFields() const;

                double fHelmholtz() const;

            private:

                int systemId_ = 1;

                FileMaster fileMaster_;

                DMixture<D> DMixture_;

                ChiInteraction *interactionPtr_;

                UnitCell<D> unitCell_;

                Mesh<D> mesh_;

                WaveList<D> *wavelistPtr_;

                std::string groupName_;

                Basis<D> *basisPtr_;

                FieldIo<D> fieldIo_;

                FFT<D> fft_;

                AmIterator<D> *iteratorPtr_;

                DArray<Field> wFields_;

                DArray<WField> wFieldsRGrid_;

                DArray<RDFieldDft<D>> wFieldsKGrid_;

                DArray<Field> cFields_;

                DArray<CField> cFieldsRGrid_;

                DArray<RDFieldDft<D>> cFieldsKGrid_;

                int rSize_;

                int kSize_;

                bool hasMixture_;

                bool hasUnitCell_;

                bool hasMesh_;

                bool isAllocated_;

                bool hasWFields_;

                bool hasCFields_;

                IntVec<D> kMeshDimensions_;

                void allocate();

                void initHomogeneous();

                DArray<double> c_;

                cudaReal fHelmholtz_;

                cudaReal U_;

                cudaReal UAB_;

                cudaReal UCMP_;

                cudaReal *S_;

                RDField<D> workArray;

                cudaReal *d_kernelWorkSpace_{};

                cudaReal *kernelWorkSpace_{};
            };

            template <int D>
            inline FileMaster &System<D>::fileMaster()
            {
                return fileMaster_;
            }

            template <int D>
            inline int System<D>::systemId()
            {
                return systemId_;
            }

            template <int D>
            inline DMixture<D> &System<D>::dmixture()
            {
                return DMixture_;
            }

            template <int D>
            inline ChiInteraction &System<D>::interaction()
            {
                UTIL_ASSERT(interactionPtr_)
                return *interactionPtr_;
            }

            template <int D>
            inline Mesh<D> &System<D>::mesh()
            {
                return mesh_;
            }

            template <int D>
            inline std::string System<D>::groupName()
            {
                return groupName_;
            }

            template <int D>
            inline UnitCell<D> &System<D>::unitCell()
            {
                return unitCell_;
            }

            template <int D>
            inline WaveList<D> &System<D>::wavelist()
            {
                return *wavelistPtr_;
            }

            template <int D>
            inline Basis<D> &System<D>::basis()
            {
                UTIL_ASSERT(basisPtr_)
                return *basisPtr_;
            }

            template <int D>
            inline AmIterator<D> &System<D>::iterator()
            {
                UTIL_ASSERT(iteratorPtr_)
                return *iteratorPtr_;
            }

            template <int D>
            inline FieldIo<D> &System<D>::fieldIo()
            {
                return fieldIo_;
            }

            template <int D>
            inline FFT<D> &System<D>::fft()
            {
                return fft_;
            }

            template <int D>
            inline DArray<RDField<D>> &System<D>::wFields()
            {
                return wFields_;
            }

            template <int D>
            inline RDField<D> &System<D>::wField(int id)
            {
                return wFields_[id];
            }

            template <int D>
            inline DArray<typename System<D>::WField> &System<D>::wFieldsRGrid()
            {
                return wFieldsRGrid_;
            }

            template <int D>
            inline
                typename System<D>::WField &
                System<D>::wFieldRGrid(int id)
            {
                return wFieldsRGrid_[id];
            }

            template <int D>
            inline DArray<RDFieldDft<D>> &System<D>::wFieldsKGrid()
            {
                return wFieldsKGrid_;
            }

            template <int D>
            inline RDFieldDft<D> &System<D>::wFieldKGrid(int id)
            {
                return wFieldsKGrid_[id];
            }

            template <int D>
            inline DArray<RDField<D>> &System<D>::cFields()
            {
                return cFields_;
            }

            template <int D>
            inline RDField<D> &System<D>::cField(int id)
            {
                return cFields_[id];
            }

            template <int D>
            inline DArray<typename System<D>::CField> &System<D>::cFieldsRGrid()
            {
                return cFieldsRGrid_;
            }

            template <int D>
            inline
                typename System<D>::CField &
                System<D>::cFieldRGrid(int id)
            {
                return cFieldsRGrid_[id];
            }

            template <int D>
            inline DArray<RDFieldDft<D>> &System<D>::cFieldsKGrid()
            {
                return cFieldsKGrid_;
            }

            template <int D>
            inline RDFieldDft<D> &System<D>::cFieldKGrid(int id)
            {
                return cFieldsKGrid_[id];
            }

            template <int D>
            inline double System<D>::c(int id)
            {
                return c_[id];
            }

            template <int D>
            inline bool System<D>::hasWFields() const
            {
                return hasWFields_;
            }

            template <int D>
            inline bool System<D>::hasCFields() const
            {
                return hasCFields_;
            }

            // Get precomputed Helmholtz free energy per monomer / kT.
            template <int D>
            inline double System<D>::fHelmholtz() const
            {
                return fHelmholtz_;
            }

#ifndef D_SYSTEM_TPP
            // Suppress implicit instantiation
            extern template class System<1>;
            extern template class System<2>;
            extern template class System<3>;
#endif
        }

    }
}

#endif // ! D_SYSTEM_H