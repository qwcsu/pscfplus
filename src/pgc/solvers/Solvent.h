#ifndef PGC_SOLVENT_H
#define PGC_SOLVENT_H

/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include <pscf/chem/Species.h>
#include <util/param/ParamComposite.h>
#include <pspg/field/RDField.h>

namespace Pscf
{
   namespace Pspg
   {

      using namespace Util;

      /**
       * Class representing a solvent species.
       *
       */
      template <int D>
      class Solvent : public Species, public ParamComposite
      {
      public:
         /**
          * Monomer concentration field.
          */
         typedef RDField<D> CField;

         /**
          * Monomer chemical potential field.
          */
         typedef RDField<D> WField;

         /**
          * Constructor.
          */
         Solvent()
         {
         }

         /**
          * Constructor.
          */
         ~Solvent()
         {
         }

         /**
          * Compute monomer concentration field and phi and/or mu.
          *
          * Pure virtual function: Must be implemented by subclasses.
          * Upon return, concentration field, phi and mu are all set.
          *
          * \param wField monomer chemical potential field.
          */
         virtual void compute(WField const &wField){};

         /**
          * Get monomer concentration field for this solvent.
          */
         const CField &concentration() const
         {
            return concentration_;
         }

      private:
         CField concentration_;
      };

   }
}

#endif
