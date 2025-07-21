/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "ChiInteraction.h"
#include <pscf/math/LuSolver.h>

namespace Pscf {

   using namespace Util;

   /*
   * Constructor.
   */
   ChiInteraction::ChiInteraction()
    : Interaction()
   {  setClassName("Interaction"); }

   /*
   * Destructor.
   */
   ChiInteraction::~ChiInteraction()
   {}

   /*
   * Read chi matrix from file.
   */
   void ChiInteraction::readParameters(std::istream& in)
   {
      UTIL_CHECK(nMonomer() > 0);
      chi_.allocate(nMonomer(), nMonomer());
      chiInverse_.allocate(nMonomer(), nMonomer());
      idemp_.allocate(nMonomer(), nMonomer());
      for (int i = 0; i < nMonomer(); ++i)
      {
         for (int j = 0; j < nMonomer(); ++j)
         {
            chi_(i,j) = 0.0;
         }
      }
      readDSymmMatrix(in, "chiN", chi_, nMonomer());

// #if (CHN != 0)
//       for (int i = 0; i < nMonomer(); ++i)
//       {
//          for (int j = 0; j < nMonomer(); ++j)
//          {
//             chi_(i,j) *= nSegment();
//             // std::cout << chi(i,j) << "\n";
//          }
//       }
//       // exit(1);
// #endif

      if (nMonomer() == 2) {
         double det = chi_(0,0)*chi_(1, 1) - chi_(0,1)*chi_(1,0);
         double norm = chi_(0,0)*chi_(0, 0) + chi_(1,1)*chi_(1,1)
                     + 2.0*chi_(0,1)*chi_(1,0);
         if (fabs(det/norm) < 1.0E-10) {
            UTIL_THROW("Singular chi matrix");
         }
         chiInverse_(0,1) = -chi_(0,1)/det;
         chiInverse_(1,0) = -chi_(1,0)/det;
         chiInverse_(1,1) = chi_(0,0)/det;
         chiInverse_(0,0) = chi_(1,1)/det;

      } else {
         LuSolver solver;
         solver.allocate(nMonomer());
         solver.computeLU(chi_);
         solver.inverse(chiInverse_);
      }

      double sum = 0;
      int i, j, k;

      for (i = 0; i < nMonomer(); ++i) {
         idemp_(0,i) = 0;
         for (j = 0; j < nMonomer(); ++j) {
            idemp_(0,i) -= chiInverse_(j,i);
         }
         sum -= idemp_(0,i);
         for (k = 0; k < nMonomer(); ++k) { //row
            idemp_(k,i) = idemp_(0,i);
         }
      }

      for (i = 0; i < nMonomer(); ++i) { //row
         for (j = 0; j < nMonomer(); ++j) { //coloumn
            idemp_(i,j) /= sum;
         }
         idemp_(i,i) +=1.0;
      }
      
      sum_inv_ = sum;

   }

   /*
   * Compute and return excess Helmholtz free energy per monomer.
   */
   double ChiInteraction::fHelmholtz(Array<double> const & c) const
   {
      int i, j;
      double sum = 0.0;
      for (i = 0; i < nMonomer(); ++i) {
         for (j = 0; j < nMonomer(); ++j) {
            sum += chi_(i, j)* c[i]*c[j];
         }
      }
      return 0.5*sum;
   }

   /*
   * Compute chemical potential from monomer concentrations
   */
   void
   ChiInteraction::computeW(Array<double> const & c,
                            Array<double>& w) const
   {
      int i, j;
      for (i = 0; i < nMonomer(); ++i) {
         w[i] = 0.0;
         for (j = 0; j < nMonomer(); ++j) {
            w[i] += chi_(i, j)* c[j];
         }
      }
   }

   /*
   * Compute concentrations and xi from chemical potentials.
   */
   void
   ChiInteraction::computeC(Array<double> const & w,
                            Array<double>& c, double& xi)
   const
   {
      double sum1 = 0.0;
      double sum2 = 0.0;
      int i, j;
      for (i = 0; i < nMonomer(); ++i) {
         for (j = 0; j < nMonomer(); ++j) {
            sum1 += chiInverse_(i, j)*w[j];
            sum2 += chiInverse_(i, j);
         }
      }
      xi = (sum1 - 1.0)/sum2;
      for (i = 0; i < nMonomer(); ++i) {
         c[i] = 0;
         for (j = 0; j < nMonomer(); ++j) {
            c[i] += chiInverse_(i, j)*( w[j] - xi );
         }
      }
   }

   /*
   * Compute Langrange multiplier from chemical potentials.
   */
   void
   ChiInteraction::computeXi(Array<double> const & w, double& xi)
   const
   {
      double sum1 = 0.0;
      double sum2 = 0.0;
      int i, j;
      for (i = 0; i < nMonomer(); ++i) {
         for (j = 0; j < nMonomer(); ++j) {
            sum1 += chiInverse_(i, j)*w[j];
            sum2 += chiInverse_(i, j);
         }
      }
      xi = (sum1 - 1.0)/sum2;
   }

   /*
   * Return dWdC = chi matrix.
   */
   void
   ChiInteraction::computeDwDc(Array<double> const & c, 
                               Matrix<double>& dWdC) const
   {
      int i, j;
      for (i = 0; i < nMonomer(); ++i) {
         for (j = 0; j < nMonomer(); ++j) {
            dWdC(i, j) = chi_(i, j);
         }
      }
   }

   void ChiInteraction::setChi(int i, int j, double chi)
   {  chi_(j, i) = chi_(i, j) = chi; }

} // namespace Pscf
