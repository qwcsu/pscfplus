#ifndef D_MIXTURE_TMPL_H
#define D_MIXTURE_TMPL_H

/*
 * PSCF - Polymer Self-Consistent Field Theory
 *
 * Copyright 2016 - 2019, The Regents of the University of Minnesota
 * Distributed under the terms of the GNU General Public License.
 */

#include <pscf/chem/Monomer.h>
#include <util/param/ParamComposite.h>
#include <util/containers/DArray.h>

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {

            using namespace Util;

            /**
             * A mixture of polymer and solvent species.
             *
             * \ingroup Pscf_Solver_Module
             */
            template <class TP, class TS>
            class DMixtureTmpl : public ParamComposite
            {
            public:
                // Public typedefs

                /**
                 * Polymer species solver type.
                 */
                typedef TP DPolymer;

                /**
                 * Solvent species solver type.
                 */
                typedef TS Solvent;

                // Public member functions

                /**
                 * Constructor.
                 */
                DMixtureTmpl();

                /**
                 * Destructor.
                 */
                ~DMixtureTmpl();

                /**
                 * Read parameters from file and initialize.
                 *
                 * \param in input parameter file
                 */
                virtual void readParameters(std::istream &in);

                /// \name Accessors (by non-const reference)
                //@{

                /**
                 * Get a Monomer type descriptor.
                 *
                 * \param id integer monomer type index (0 <= id < nMonomer)
                 */
                Monomer &monomer(int id);

                /**
                 * Get a polymer object.
                 *
                 * \param id integer polymer species index (0 <= id < nPolymer)
                 */
                DPolymer &polymer(int id);

                /**
                 * Set a solvent solver object.
                 *
                 * \param id integer solvent species index (0 <= id < nSolvent)
                 */
                Solvent &solvent(int id);

                //@}
                /// \name Accessors (by value)
                //@{

                /**
                 * Get number of monomer types.
                 */
                int nMonomer() const;

                /**
                 * Get number of polymer species.
                 */
                int nPolymer() const;

                /**
                 * Get number of solvent (point particle) species.
                 */
                int nSolvent() const;

                //@}

            private:
                /**
                 * Array of monomer type descriptors.
                 */
                DArray<Monomer> monomers_;

                /**
                 * Array of polymer species solver objects.
                 *
                 * Array capacity = nPolymer.
                 */
                DArray<DPolymer> polymers_;

                /**
                 * Array of solvent species objects.
                 *
                 * Array capacity = nSolvent.
                 */
                DArray<Solvent> solvents_;

                /**
                 * Number of monomer types.
                 */
                int nMonomer_;

                /**
                 * Number of polymer species.
                 */
                int nPolymer_;

                /**
                 * Number of solvent species.
                 */
                int nSolvent_;
            };

            // Inline member functions

            template <class TP, class TS>
            inline int DMixtureTmpl<TP, TS>::nMonomer() const
            {
                return nMonomer_;
            }

            template <class TP, class TS>
            inline int DMixtureTmpl<TP, TS>::nPolymer() const
            {
                return nPolymer_;
            }

            template <class TP, class TS>
            inline int DMixtureTmpl<TP, TS>::nSolvent() const
            {
                return nSolvent_;
            }

            template <class TP, class TS>
            inline Monomer &DMixtureTmpl<TP, TS>::monomer(int id)
            {
                return monomers_[id];
            }

            template <class TP, class TS>
            inline TP &DMixtureTmpl<TP, TS>::polymer(int id)
            {
                return polymers_[id];
            }

            template <class TP, class TS>
            inline TS &DMixtureTmpl<TP, TS>::solvent(int id)
            {
                return solvents_[id];
            }

            // Non-inline member functions

            /*
             * Constructor.
             */
            template <class TP, class TS>
            DMixtureTmpl<TP, TS>::DMixtureTmpl()
                : ParamComposite(),
                  monomers_(),
                  polymers_(),
                  solvents_(),
                  nMonomer_(0),
                  nPolymer_(0),
                  nSolvent_(0)
            {
            }

            /*
             * Destructor.
             */
            template <class TP, class TS>
            DMixtureTmpl<TP, TS>::~DMixtureTmpl()
            {
            }

            /*
             * Read all parameters and initialize.
             */
            template <class TP, class TS>
            void DMixtureTmpl<TP, TS>::readParameters(std::istream &in)
            {
                read<int>(in, "nMonomer", nMonomer_);
                monomers_.allocate(nMonomer_);
                readDArray<Monomer>(in, "monomers", monomers_, nMonomer_);

                // Polymers
                read<int>(in, "nPolymer", nPolymer_);
                polymers_.allocate(nPolymer_);
                for (int i = 0; i < nPolymer_; ++i)
                {
                    readParamComposite(in, polymers_[i]);
                }

                // Set statistical segment lengths for all bonds
                double kuhn1, kuhn2;
                int monomerId1, monomerId2;
                for (int i = 0; i < nPolymer_; ++i)
                {
                    for (int j = 0; j < polymer(i).nBond(); ++j)
                    {
                        monomerId1 = polymer(i).bond(j).monomerId(0);
                        monomerId2 = polymer(i).bond(j).monomerId(1);
                        if (monomerId1 >= 0 && monomerId2 >= 0)
                        {
                            kuhn1 = monomer(monomerId1).step();
                            kuhn2 = monomer(monomerId2).step();
                            polymer(i).bond(j).setKuhn(kuhn1, kuhn2);
                        }
                        else if (monomerId1 >= 0 && monomerId2 < 0)
                        {
                            kuhn1 = monomer(monomerId1).step();
                            polymer(i).bond(j).setKuhn(kuhn1, kuhn1);
                        }
                        else if (monomerId1 < 0 && monomerId2 >= 0)
                        {
                            kuhn2 = monomer(monomerId2).step();
                            polymer(i).bond(j).setKuhn(kuhn2, kuhn2);
                        }
                    }
                }
            }

        }
    }
}
#endif
