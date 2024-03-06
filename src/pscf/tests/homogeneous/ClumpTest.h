#ifndef PSCF_CLUMP_TEST_H
#define PSCF_CLUMP_TEST_H

#include <test/UnitTest.h>
#include <test/UnitTestRunner.h>

#include <pscf/homogeneous/Clump.h>

#include <fstream>

using namespace Pscf;
//using namespace Util;

class ClumpTest : public UnitTest 
{

public:

   void setUp()
   {}

   void tearDown()
   {}

  
   void testConstructor()
   {
      printMethod(TEST_FUNC);
      Homogeneous::Clump clump;
   } 

   void testSetters()
   {
      printMethod(TEST_FUNC);
      Homogeneous::Clump clump;
      clump.setMonomerId(0);
      clump.setSize(2.0);
      TEST_ASSERT(clump.monomerId() == 0);
      TEST_ASSERT(eq(clump.size(), 2.0));
      // std::cout << std::endl << clump << std::endl ;
   } 

   void testReadWrite() {
      printMethod(TEST_FUNC);
      printEndl();

      Homogeneous::Clump clump;
      std::ifstream in;
      openInputFile("in/Clump", in);

      in >> clump;
      TEST_ASSERT(clump.monomerId() == 0);
      TEST_ASSERT(eq(clump.size(), 2.0));
      std::cout << clump << std::endl ;
   }

};

TEST_BEGIN(ClumpTest)
TEST_ADD(ClumpTest, testConstructor)
TEST_ADD(ClumpTest, testSetters)
TEST_ADD(ClumpTest, testReadWrite)
TEST_END(ClumpTest)

#endif
