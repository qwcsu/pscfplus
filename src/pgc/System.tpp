#ifndef PGC_SYSTEM_TPP
#define PGC_SYSTEM_TPP
// #define double float
/*
* PSCF - Polymer Self-Consistent Field Theory 
*
* Copyright 2016 - 2022, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*
* Modified in 2024 by the Wang Lab of Computational Soft Materials at
* Colorado State University for PSCF+, an improved and extended version of PSCF.
*/

#include "System.h"
#include <pspg/math/GpuHeaders.h>

#include <pscf/homogeneous/Clump.h>
#include <pscf/crystal/shiftToMinimum.h>

#include <util/format/Str.h>
#include <util/format/Int.h>
#include <util/format/Dbl.h>

#include <string>
#include <getopt.h>
#include <algorithm>
#include <vector>

// Global variable for kernels
int THREADS_PER_BLOCK;
int NUMBER_OF_BLOCKS;

namespace Pscf
{
    namespace Pspg
    {
        namespace Continuous
        {

            using namespace Util;

            /*
             * Constructor
             */
            template <int D>
            System<D>::System()
                : mixture_(),
                  mesh_(),
                  unitCell_(),
                  fileMaster_(),
                  homogeneous_(),
                  interactionPtr_(nullptr),
                  iteratorPtr_(0),
                  basisPtr_(),
                  wavelistPtr_(0),
                  fieldIo_(),
                  wFields_(),
                  cFields_(),
                  f_(),
                  c_(),
                  fHelmholtz_(0.0),
                  pressure_(0.0),
                  hasMixture_(false),
                  hasUnitCell_(false),
                  isAllocated_(false),
                  hasWFields_(false),
                  hasCFields_(false)
            {
                setClassName("System");
                interactionPtr_ = new ChiInteraction();
                iteratorPtr_ = new AmIterator<D>(this);
                wavelistPtr_ = new WaveList<D>();
                basisPtr_ = new Basis<D>();

                ThreadGrid::init();
            }

            template <int D>
            System<D>::~System()
            {
                delete interactionPtr_;
                delete iteratorPtr_;
                delete wavelistPtr_;
                delete basisPtr_;
            }

            /*
             * Process command line options.
             */
            template <int D>
            void System<D>::setOptions(int argc, char **argv)
            {
                bool eFlag = false; // echo
                bool pFlag = false; // param file
                bool cFlag = false; // command file
                bool iFlag = false; // input prefix
                bool oFlag = false; // output prefix
                bool tFlag = false; // GPU input 2 (threads per block)
                char *pArg = 0;
                char *cArg = 0;
                char *iArg = 0;
                char *oArg = 0;
                int tArg = 0;
                // Read program arguments
                int c;
                opterr = 0;
                while ((c = getopt(argc, argv, "er:p:c:i:o:t:")) != -1)
                {
                    switch (c)
                    {
                    case 'e':
                        eFlag = true;
                        break;
                    case 'p': // parameter file
                        pFlag = true;
                        pArg = optarg;
                        break;
                    case 'c': // command file
                        cFlag = true;
                        cArg = optarg;
                        break;
                    case 'i': // input prefix
                        iFlag = true;
                        iArg = optarg;
                        break;
                    case 'o': // output prefix
                        oFlag = true;
                        oArg = optarg;
                        break;
                    case 't': // threads per block
                        THREADS_PER_BLOCK = atoi(optarg);
                        tFlag = true;
                        break;
                    case '?':
                        Log::file() << "Unknown option -" << optopt << std::endl;
                        UTIL_THROW("Invalid command line option");
                    default:
                        UTIL_THROW("Default exit (setOptions)");
                    }
                }

                // Set flag to echo parameters as they are read.
                if (eFlag)
                {
                    Util::ParamComponent::setEcho(true);
                }

                // If option -p, set parameter file name
                if (pFlag)
                {
                    fileMaster().setParamFileName(std::string(pArg));
                }
                // else
                // {
                //     fileMaster().setParamFileName(std::string("param"));
                // }

                // If option -c, set command file name
                if (cFlag)
                {
                    fileMaster().setCommandFileName(std::string(cArg));
                }
                // else
                // {
                //     fileMaster().setCommandFileName(std::string("command"));
                // }

                // If option -i, set path prefix for input files
                if (iFlag)
                {
                    fileMaster().setInputPrefix(std::string(iArg));
                }

                // If option -o, set path prefix for output files
                if (oFlag)
                {
                    fileMaster().setOutputPrefix(std::string(oArg));
                }

                // If option -t, set the threads per block.
                if (tFlag)
                {
                    ThreadGrid::setThreadsPerBlock(tArg);
                    exit(1);
                }
            }

            template <int D>
            void System<D>::setOptionsOutside(char *pArg, char *cArg, bool echo, int s)
            {
                systemId_ = s;
                Util::ParamComponent::setEcho(echo);
                fileMaster().setParamFileName(std::string(pArg));
                fileMaster().setCommandFileName(std::string(cArg));
                NUMBER_OF_BLOCKS = 32;
                THREADS_PER_BLOCK = 512;
            }

            /*
             * Read default parameter file.
             */
            template <int D>
            void System<D>::readParam()
            {
                // paramFile() returns the file pointer of parameter file
                readParam(fileMaster().paramFile());
                fileMaster().paramFile().clear();
                fileMaster().paramFile().seekg(0);
            }

            /*
             * Read parameter file (including open and closing brackets).
             */
            template <int D>
            void System<D>::readParam(std::istream &in)
            {
                readBegin(in, className().c_str());
                readParameters(in);
                readEnd(in);
            }

            /*
             * Read parameters and initialize.
             */
            template <int D>
            void System<D>::readParameters(std::istream &in)
            {
                readParamComposite(in, mixture());
                hasMixture_ = true;

                int nm = mixture().nMonomer();
                int np = mixture().nPolymer();
                int ns = 0;

                // Initialize homogeneous object
                homogeneous_.setNMolecule(np + ns);
                homogeneous_.setNMonomer(nm);
                initHomogeneous();

                // Read interaction (i.e., chi parameters)
                interaction().setNMonomer(mixture().nMonomer());
                readParamComposite(in, interaction());
                mixture().setInteraction(interaction());

                // Read unit cell type and its parameters
                read(in, "unitCell", unitCell_);
                hasUnitCell_ = true;

                /// Read crystallographic unit cell (used only to create basis)
                read(in, "mesh", mesh_);
                mixture().setMesh(mesh(), unitCell());
                hasMesh_ = true;

                // Construct wavelist
                wavelist().allocate(mesh(), unitCell());

                // #if DFT == 0
                wavelist().computeMinimumImages(mesh(), unitCell());
                // #endif
                mixture().setupUnitCell(unitCell(), wavelist());

                // Read group name, construct basis
                read(in, "groupName", groupName_);
                basis().makeBasis(mesh(), unitCell(), groupName_);
                fieldIo_.associate(unitCell_, mesh_, fft_, groupName_,
                                   basis(), fileMaster_);

                mixture().setU0(unitCell(), wavelist());

                wavelist().computedKSq(unitCell());

                // Allocate memory for w and c fields
                allocate();

                // Initialize iterator
                readParamComposite(in, iterator());
                iterator().allocate();

                for (int i = 0; i < mixture().nPolymer(); ++i)
                    for (int j = 0; j < mixture().polymer(i).nBlock(); ++j)
                    {
                        mixture().polymer(i).block(j).setWavelist(wavelist());
                    }
            }

            /*
             * Allocate memory for fields
             */
            template <int D>
            void System<D>::allocate()
            {
                // Preconditions
                UTIL_CHECK(hasMixture_)
                UTIL_CHECK(hasMesh_)

                // Allocate wFields and cFields
                int nMonomer = mixture().nMonomer();

                wFieldsRGrid_ph_.allocate(nMonomer);
                wFieldsKGrid_.allocate(nMonomer);
                cFieldsKGrid_.allocate(nMonomer);
                cFieldsRGrid_.allocate(nMonomer);
#if DFT == 0
                wFields_.allocate(nMonomer);
                cFields_.allocate(nMonomer);
#endif
                for (int i = 0; i < nMonomer; ++i)
                {
                    wFieldRGridPh(i).allocate(mesh().dimensions());
                    wFieldKGrid(i).allocate(mesh().dimensions());
                    cFieldRGrid(i).allocate(mesh().dimensions());
                    cFieldKGrid(i).allocate(mesh().dimensions());
#if DFT == 0
                    wField(i).allocate(basis().nStar());
                    cField(i).allocate(basis().nStar());
#endif
                }
#if DFT == 0
                workArray.allocate(mesh().dimensions());
#if CMP == 1
                cFieldTot_.allocate(mesh().dimensions());
#endif
                workArrayDft.allocate(mesh().dimensions());
                ThreadGrid::setThreadsLogical(mesh().size(), NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
#endif
#if DFT == 1
                workArray.allocate(mesh().dimensions());
                ThreadGrid::setThreadsLogical(mesh().size(), NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
#endif

                isAllocated_ = true;
            }

            template <int D>
            void System<D>::iterate()
            {
                Log::file() << "System " << systemId_  << ": "<< "\n";
                Log::file() << std::endl;

                // Read w fields in grid format iff not already set.
                if (!hasWFields_)
                {
                    Log::file() << "Read w field before iteration." << std::endl;
                    exit(1);
                }

                // Attempt to iteratively solve SCFT equations
                int fail = iterator().solve();
                hasCFields_ = true;

                if (fail)
                {
                    Log::file() << "Iterate has failed. Exiting " << std::endl;
                }
                computeFreeEnergy();
                outputThermo(Log::file());
            }

            template <int D>
            void System<D>::fieldIO(std::string io,
                                    std::string type,
                                    std::string format,
                                    std::string dir,
                                    std::string caseid,
                                    std::string prefix)
            {
                if (   io == "read"
                    && type == "omega"
                    && format == "basis")
                {
                    Log::file() << std::endl;
                    Log::file() << "Reading omega field in basis:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += "/";
                    fieldFileName += caseid;
                    fieldFileName += "_omega.basis";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().readFieldsBasis(fieldFileName, wFields());
                    fieldIo().convertBasisToRGrid(wFields(), wFieldsRGridPh());
                    hasWFields_ = true;
                }
                else if (   io == "write"
                         && type == "omega"
                         && format == "basis")
                {
                    Log::file() << std::endl;
                    Log::file() << "Writing omega field in basis:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += "/";
                    fieldFileName += prefix;
                    fieldFileName += caseid;
                    fieldFileName += "_omega.basis";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().convertRGridToBasis(wFieldsRGridPh(), wFields());
                    fieldIo().writeFieldsBasis(fieldFileName, wFields());
                }
                else if (   io == "write"
                         && type == "omega"
                         && format == "reciprocal")
                {
                    Log::file() << std::endl;
                    Log::file() << "Writing omega field in reciprocal-space grid:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += "/";
                    fieldFileName += prefix;
                    fieldFileName += caseid;
                    fieldFileName += "_omega.rcf";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().convertRGridToBasis(wFieldsRGridPh(), wFields());
                    fieldIo().convertBasisToKGrid(wFields(), wFieldsKGrid());
                    fieldIo().writeFieldsKGrid(fieldFileName, wFieldsKGrid());
                }
                else if (   io == "read"
                         && type == "omega"
                         && format == "real")
                {
                    Log::file() << std::endl;
                    Log::file() << "Reading omega field in real-space:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += caseid;
                    fieldFileName += "_omega.real";
                    Log::file() << " " << Str(fieldFileName, 20) << std::endl;
                    fieldIo().readFieldsRGrid(fieldFileName, wFieldsRGridPh());
                    hasWFields_ = true;
                }
                else if (   io == "write"
                         && type == "omega"
                         && format == "real")
                {
                    UTIL_CHECK(hasWFields_)
                    Log::file() << std::endl;
                    Log::file() << "Writing omega field in real-space:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += "/";
                    fieldFileName += prefix;
                    fieldFileName += caseid;
                    fieldFileName += "_omega.real";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().writeFieldsRGrid(fieldFileName, wFieldsRGridPh());
                }
                else if (   io == "write"
                         && type == "phi"
                         && format == "real")
                {
                    UTIL_CHECK(hasWFields_)
                    Log::file() << std::endl;
                    Log::file() << "Writing volume fraction field in real-space:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += "/";
                    fieldFileName += prefix;
                    fieldFileName += caseid;
                    fieldFileName += "_phi.real";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().writeFieldsRGrid(fieldFileName, cFieldsRGrid());
                }
                else if (   io == "write"
                         && type == "phi"
                         && format == "basis")
                {
                    UTIL_CHECK(hasWFields_)
                    Log::file() << std::endl;
                    Log::file() << "Writing volume fraction field in basis:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += "/";
                    fieldFileName += prefix;
                    fieldFileName += caseid;
                    fieldFileName += "_phi.basis";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());
                    fieldIo().writeFieldsBasis(fieldFileName, cFields());
                }
                else if (   io == "write"
                         && type == "phi"
                         && format == "reciprocal")
                {
                    Log::file() << std::endl;
                    Log::file() << "Writing volume fraction in reciprocal-space grid:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += "/";
                    fieldFileName += prefix;
                    fieldFileName += caseid;
                    fieldFileName += "_phi.rcf";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());
                    fieldIo().convertBasisToKGrid(cFields(), cFieldsKGrid());
                    fieldIo().writeFieldsKGrid(fieldFileName, cFieldsKGrid());
                }
                else
                {
                    std::cout << "      I/O type not included." << std::endl;
                    exit(1);
                }
            }


            template <int D>
            void System<D>::readCommandsJson(int s)
            {
                if (fileMaster().commandFileName().empty())
                {
                    UTIL_THROW("Empty command file name");
                }
                readCommandsJson(fileMaster().commandFileName(), s);
                fileMaster().commandFile().clear();
                fileMaster().commandFile().seekg(0);
            }

            template <int D>
            void System<D>::readCommandsJson(std::string filename, int s)
            {
                std::ifstream procFile (filename,
                             std::ifstream::binary);
                Json::Value proc;

                procFile >> proc;

                std::string caseid = proc[0]["CaseId"].asString();

                for (int i = 1; i < proc.size(); ++i)
                {
                    if (!proc[i]["FieldIO"].empty())
                    {
                        if (   proc[i]["FieldIO"]["System"].empty()
                            || proc[i]["FieldIO"]["System"] == systemId())
                            fieldIO(proc[i]["FieldIO"]["IO"].asString(),
                                    proc[i]["FieldIO"]["Type"].asString(),
                                    proc[i]["FieldIO"]["Format"].asString(),
                                    proc[i]["FieldIO"]["Directory"].asString(),
                                    proc[0]["CaseId"].asString(),
                                    "");

                    }
                    if (!proc[i]["PhiToWBasis"].empty())
                    {
                        hasCFields_ = false;

                        std::string fieldFileName = "./out/phi/";
                        fieldFileName += caseid;
                        fieldFileName += "_phi.real";

                        Log::file() << " " << Str(fieldFileName, 20) << std::endl;
                        fieldIo().readFieldsRGrid(fieldFileName, cFieldsRGrid());

                        fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());

                        std::string outFileName = "./out/phi/";
                        outFileName += caseid;
                        outFileName += "_phi.basis";
                        Log::file() << " " << Str(outFileName, 20) << std::endl;
                        fieldIo().writeFieldsBasis(outFileName, cFields());

                        for (int m1 = 0; m1 < mixture().nMonomer(); ++m1)
                        {
                            assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (wField(m1).cDField(), 0.0, basis().nStar());
                            for (int m2 = 0; m2 < mixture().nMonomer(); ++m2)
                            {
                                pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wField(m1).cDField(),
                                                                                           cField(m2).cDField(),
                                                                                           interaction().chi(m1, m2),
                                                                                           basis().nStar());
                            }
                        }
                        std::string outWFileName = "./out/omega/";
                        outWFileName += caseid;
                        outWFileName += "_omega.basis";
                        Log::file() << " " << Str(outWFileName, 20) << std::endl;
                        fieldIo().writeFieldsBasis(outWFileName, wFields());
                    }
                    else if(!proc[i]["SinglePhaseSCF"].empty())
                    {
                        Log::file() << "Single phase calculation: " << "\n";
                        Log::file() << "      case: " << proc[0]["CaseId"] << "\n";
                        Log::file() << std::endl;

                        iterate();

                        std::string outFileName;
                        outFileName = proc[i]["SinglePhaseSCF"]["OutputDirectory"].asString();
                        outFileName += proc[0]["CaseId"].asString();
                        outFileName += "_out.json";
                        Json::Value thermo;
                        thermo.resize(0);
                        outputThermo(thermo);
                        std::ofstream out(outFileName);
                        out << thermo; 
                    }
                    else if(!proc[i]["ACAP"].empty())
                    {
                        std::string s;
                        
                        s = proc[i]["ACAP"]["Variable"][0].asString();
                        if (s == "chiN")
                        {
                            // start point, current point, and end point.
                            // end ponit is read from command file, while
                            // start point is read from param file.
                            double start_chi, current_chi, end_chi;
                            // current increment, maximum increment, minimum increment,
                            // and adpative factor, which are read from command file
                            double inc, max_inc, min_inc, fac;
                            // Is finished? Set false initially
                            bool isFinished = false;
                            // Is chi increasing? Set true initially
                            bool dir = true;

                            std::cout << "ACAP:" << "\n";
                            
                            int m1 = proc[i]["ACAP"]["Variable"][1].asInt();
                            int m2 = proc[i]["ACAP"]["Variable"][2].asInt();

                            std::cout << "Variable: chi(" 
                                      << m1 << " " <<  m2 
                                      << ")"<< "\n";

                            start_chi = proc[i]["ACAP"]["InitialValue"].asDouble();  
                            std::cout << "Initial chi value: "<< start_chi << "\n";
                            end_chi = proc[i]["ACAP"]["FinalValue"].asDouble();
                            std::cout << "Final chi value: "<< end_chi << "\n";
                            inc = proc[i]["ACAP"]["InitialStep"].asDouble();
                            min_inc = proc[i]["ACAP"]["SmallestStep"].asDouble();
                            max_inc = proc[i]["ACAP"]["LargestStep"].asDouble();
                            fac = proc[i]["ACAP"]["StepScale"].asDouble();

                            if (start_chi == end_chi)
                            {
                                Log::file() << "The start point equals to the stop point." << std::endl;
                                exit(1);
                            }

                            if (start_chi > end_chi)
                            {
                                inc *= -1;
                                dir = false;
                            }
                            bool intermediateFlag = false;
                            int nip = 0;
                            std::vector<double> point;
                            if (!proc[i]["ACAP"]["IntermediateOuput"].empty())
                            {
                                nip = proc[i]["ACAP"]["IntermediateOuput"][0]["OutputPoints"].size();

                                for (int j = 0; j < nip; ++j)
                                {
                                    point.push_back(proc[i]["ACAP"]["IntermediateOuput"][0]["OutputPoints"][j].asDouble());
                                }

                                if (dir)
                                {
                                    std::sort(point.begin(), point.end());
                                }
                                else
                                {
                                    std::sort(point.rbegin(), point.rend());
                                }
                                intermediateFlag = true;
                            }
                            

                            current_chi = start_chi;

                            Json::Value thermo;
                            thermo.resize(0);

                            int pointid = 0;

                            while (!isFinished)
                            {
                                ++pointid;

                                if ((dir && current_chi >= end_chi) || ((!dir) && current_chi <= end_chi))
                                {
                                    current_chi = end_chi;
                                    intermediateFlag = true;
                                    isFinished = true;
                                }
                                interaction().chi_(m1, m2) = current_chi;
                                interaction().chi_(m2, m1) = current_chi;
                                unitCell().setParameters(unitCell().parameters()); // Reset cell parameters
                                // iteration

                                // Attempt to iteratively solve SCFT equations
                                Log::file() << std::endl;
                                Log::file() << "================================================" << std::endl;
                                Log::file() << "*Current chi = " << current_chi << "*" << std::endl
                                            << std::endl;
                                int fail = iterator().solve();

                                if (!fail)
                                {
                                    hasCFields_ = true;
                                    
                                    computeFreeEnergy();

                                    std::string outFileName = proc[i]["ACAP"]["OutputDirectory"].asString();
                                    outFileName += "chi_";
                                    outFileName += proc[0]["CaseId"].asString();
                                    outFileName += "_out.json";
                                    std::ofstream out(outFileName);
                                    outputThermo(thermo);
                                    out << thermo;
                                    out.close();

                                    outputThermo(Log::file());
                                    if (intermediateFlag)
                                    {
                                        std::string prefix = "chi=";
                                        prefix += std::to_string(current_chi);
                                        prefix += "_";
                                        for (int j = 1; j < proc[i]["ACAP"]["IntermediateOuput"].size(); ++j)
                                        {
                                            fieldIO("write",
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["Field"].asString(),
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["Format"].asString(),
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["OutputDirectory"].asString(),
                                                    proc[0]["CaseId"].asString(),
                                                    prefix);
                                        }
                                        intermediateFlag = false;
                                    }

                                    if (!point.empty() && (point[0] != start_chi)
                                        &&(dir && current_chi + inc >= point[0]
                                        || !dir && current_chi + inc <= point[0]))
                                    {
                                        current_chi = point[0];
                                        point.erase(point.begin());
                                        intermediateFlag = true;
                                    }
                                    else
                                    {
                                        current_chi += inc; // step forward
                                        if (abs(inc * fac) <= max_inc)
                                            inc *= fac;
                                        else
                                        {
                                            if (start_chi > end_chi)
                                                inc = -max_inc;
                                            else
                                                inc = max_inc;
                                        }
                                    }

                                }
                                else
                                {
                                    Log::file() << "Iterate has failed." << std::endl;
                                    
                                    std::string outFileName = proc[i]["ACAP"]["OutputDirectory"].asString();
                                    outFileName += "chi_";
                                    outFileName += proc[0]["CaseId"].asString();
                                    outFileName += "_out.json";
                                    std::ofstream out(outFileName);
                                    outputThermo(thermo);
                                    out << thermo;
                                    out.close();

                                    if (abs(inc) < min_inc)
                                    {
                                        Log::file() << "Smallest increment reached." << std::endl;
                                        // outRunfile.close();
                                    }
                                    else
                                    {
                                        current_chi -= inc;
                                        inc /= fac;
                                    }
                                }
                            }

                        }
                        else if (s =="b")
                        {
                            std::cout << "ACAP:" << "\n";

                            Log::file() << std::endl;

                            double start_b, current_b, end_b;
                            // current increment, maximum increment, minimum increment,
                            // and adpative factor, which are read from command file
                            double inc, max_inc, min_inc, fac;
                            int id;
                            // Is finished? Set false initially
                            bool isFinished = false;
                            // Is chi increasing? Set true initially
                            bool dir = true;

                            id = proc[i]["ACAP"]["Variable"][1].asInt();

                            std::cout << "Variable: b(" 
                                      << id << ")"<< "\n";

                            start_b = proc[i]["ACAP"]["InitialValue"].asDouble();  
                            std::cout << "Initial b(" << id <<") value:"<< start_b << "\n";
                            end_b = proc[i]["ACAP"]["FinalValue"].asDouble();
                            std::cout << "Final b(" << id<<") value: "<< end_b << "\n";
                            inc = proc[i]["ACAP"]["InitialStep"].asDouble();
                            min_inc = proc[i]["ACAP"]["SmallestStep"].asDouble();
                            max_inc = proc[i]["ACAP"]["LargestStep"].asDouble();
                            fac = proc[i]["ACAP"]["StepScale"].asDouble();

                            if (start_b == end_b)
                            {
                                Log::file() << "The start point equals to the stop point."
                                            << std::endl;
                                exit(1);
                            }
                            if (start_b > end_b)
                            {
                                inc *= -1;
                                dir = false;
                            }

                            int nip;
                            std::vector<double> point;
                            nip = proc[i]["ACAP"]["IntermediateOuput"][0]["OutputPoints"].size();

                            for (int j = 0; j < nip; ++j)
                            {
                                point.push_back(proc[i]["ACAP"]["IntermediateOuput"][0]["OutputPoints"][j].asDouble());
                            }

                            if (dir)
                            {
                                std::sort(point.begin(), point.end());
                            }
                            else
                            {
                                std::sort(point.rbegin(), point.rend());
                            }

                            current_b = start_b;

                            bool intermediateFlag = true;
                            
                            Json::Value thermo;
                            thermo.resize(0);

                            int pointid = 0;

                            while (!isFinished)
                            {
                                ++pointid;

                                if ((dir && current_b >= end_b) || ((!dir) && current_b <= end_b))
                                {
                                    current_b = end_b;
                                    intermediateFlag = true;
                                    isFinished = true;
                                }
                                
                                int monomerId;
                                this->mixture().monomer(id).setStep(current_b);
                                for (int i = 0; i < this->mixture().nPolymer(); ++i)
                                {
                                    for (int j = 0; j < this->mixture().polymer(i).nBlock(); ++j)
                                    {
                                        monomerId = this->mixture().polymer(i).block(j).monomerId();  
                                        if (monomerId == id)
                                        {
                                            this->mixture().polymer(i).block(j).setKuhn(current_b);
                                        }
                                    }
                                }
                                
                                unitCell().setParameters(unitCell().parameters()); // Reset cell parameters
                                // iteration

                                // Attempt to iteratively solve SCFT equations
                                Log::file() << std::endl;
                                Log::file() << "================================================" << std::endl;
                                Log::file() << "*Current b = " << current_b << "*" << std::endl
                                            << std::endl;

                                int fail = iterator().solve();
                                
                                if (!fail)
                                {
                                    hasCFields_ = true;
                                    computeFreeEnergy();

                                    std::string outFileName = proc[i]["ACAP"]["OutputDirectory"].asString();
                                    outFileName += "b_";
                                    outFileName += proc[0]["CaseId"].asString();
                                    outFileName += "_out.json";
                                    std::ofstream out(outFileName);
                                    outputThermo(thermo);
                                    out << thermo;
                                    out.close();

                                    outputThermo(Log::file());
                                    if (intermediateFlag)
                                    {
                                        std::string prefix = "b=";
                                        prefix += std::to_string(current_b);
                                        prefix += "_";
                                        for (int j = 1; j < proc[i]["ACAP"]["IntermediateOuput"].size(); ++j)
                                        {
                                            fieldIO("write",
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["Field"].asString(),
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["Format"].asString(),
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["OutputDirectory"].asString(),
                                                    proc[0]["CaseId"].asString(),
                                                    prefix);
                                        }
                                        intermediateFlag = false;
                                    }

                                    if (!point.empty() && (point[0] != start_b)
                                        &&(dir && current_b + inc >= point[0]
                                        || !dir && current_b + inc <= point[0]))
                                    {
                                        current_b = point[0];
                                        point.erase(point.begin());
                                        intermediateFlag = true;
                                    }
                                    else
                                    {
                                        current_b += inc; // step forward
                                        if (abs(inc * fac) <= max_inc)
                                            inc *= fac;
                                        else
                                        {
                                            if (start_b > end_b)
                                                inc = -max_inc;
                                            else
                                                inc = max_inc;
                                        }
                                    }
                                }
                                else
                                {
                                    Log::file() << "Iterate has failed." << std::endl;
                                    
                                    std::string outFileName = proc[i]["ACAP"]["OutputDirectory"].asString();
                                    outFileName += "b_";
                                    outFileName += proc[0]["CaseId"].asString();
                                    outFileName += "_out.json";
                                    std::ofstream out(outFileName);
                                    outputThermo(thermo);
                                    out << thermo;
                                    out.close();

                                    if (abs(inc) < min_inc)
                                    {
                                        Log::file() << "Smallest increment reached." << std::endl;
                                        // outRunfile.close();
                                    }
                                    else
                                    {
                                        current_b -= inc;
                                        inc /= fac;
                                    }
                                }
                            }

                            exit(1);
                        }
                        if (s == "chiN_all")
                        {
                            // start point, current point, and end point.
                            // end ponit is read from command file, while
                            // start point is read from param file.
                            double start_chi, current_chi, end_chi;
                            // current increment, maximum increment, minimum increment,
                            // and adpative factor, which are read from command file
                            double inc, max_inc, min_inc, fac;
                            // Is finished? Set false initially
                            bool isFinished = false;
                            // Is chi increasing? Set true initially
                            bool dir = true;

                            std::cout << "ACAP:" << "\n";

                            std::cout << "Variable: chi interaction between :"<< "\n";
                            for (int m1 = 0; m1 < mixture().nMonomer(); ++m1)
                            {
                                for (int m2 = 0; m2 < mixture().nMonomer(); ++m2)
                                {
                                    if (m1 < m2)
                                        std::cout << "      (" << m1 << "," << m2 << ")\n";
                                }
                            }
                            
                        
                            start_chi = proc[i]["ACAP"]["InitialValue"].asDouble();  
                            std::cout << "Initial chi value: "<< start_chi << "\n";
                            end_chi = proc[i]["ACAP"]["FinalValue"].asDouble();
                            std::cout << "Final chi value: "<< end_chi << "\n";
                            inc = proc[i]["ACAP"]["InitialStep"].asDouble();
                            min_inc = proc[i]["ACAP"]["SmallestStep"].asDouble();
                            max_inc = proc[i]["ACAP"]["LargestStep"].asDouble();
                            fac = proc[i]["ACAP"]["StepScale"].asDouble();

                            if (start_chi == end_chi)
                            {
                                Log::file() << "The start point equals to the stop point." << std::endl;
                                exit(1);
                            }

                            if (start_chi > end_chi)
                            {
                                inc *= -1;
                                dir = false;
                            }
                            bool intermediateFlag = false;
                            int nip = 0;
                            std::vector<double> point;
                            if (!proc[i]["ACAP"]["IntermediateOuput"].empty())
                            {
                                nip = proc[i]["ACAP"]["IntermediateOuput"][0]["OutputPoints"].size();

                                for (int j = 0; j < nip; ++j)
                                {
                                    point.push_back(proc[i]["ACAP"]["IntermediateOuput"][0]["OutputPoints"][j].asDouble());
                                }

                                if (dir)
                                {
                                    std::sort(point.begin(), point.end());
                                }
                                else
                                {
                                    std::sort(point.rbegin(), point.rend());
                                }
                                intermediateFlag = true;
                            }
                            

                            current_chi = start_chi;

                            Json::Value thermo;
                            thermo.resize(0);

                            int pointid = 0;

                            while (!isFinished)
                            {
                                ++pointid;

                                if ((dir && current_chi >= end_chi) || ((!dir) && current_chi <= end_chi))
                                {
                                    current_chi = end_chi;
                                    intermediateFlag = true;
                                    isFinished = true;
                                }

                                for (int m1 = 0; m1 < mixture().nMonomer(); ++m1)
                                {
                                    for (int m2 = 0; m2 < mixture().nMonomer(); ++m2)
                                    {
                                        if (m1 < m2)
                                        {
                                            interaction().chi_(m1, m2) = current_chi;
                                            interaction().chi_(m2, m1) = current_chi;
                                        }
                                    }
                                }
                                unitCell().setParameters(unitCell().parameters()); // Reset cell parameters
                                // iteration

                                // Attempt to iteratively solve SCFT equations
                                Log::file() << std::endl;
                                Log::file() << "================================================" << std::endl;
                                Log::file() << "*Current chi: " << "*" << std::endl
                                            << std::endl;
                                for (int m1 = 0; m1 < mixture().nMonomer(); ++m1)
                                {
                                    for (int m2 = 0; m2 < mixture().nMonomer(); ++m2)
                                    {
                                        if (m1 < m2)
                                        {
                                            Log::file() << "chi(" << m1 << "," << m2 << ") = " << interaction().chi_(m2, m1) << std::endl;
                                        }
                                    }
                                }
                                
                                int fail = iterator().solve();

                                if (!fail)
                                {
                                    hasCFields_ = true;
                                    
                                    computeFreeEnergy();

                                    std::string outFileName = proc[i]["ACAP"]["OutputDirectory"].asString();
                                    outFileName += "chi_";
                                    outFileName += proc[0]["CaseId"].asString();
                                    outFileName += "_out.json";
                                    std::ofstream out(outFileName);
                                    outputThermo(thermo);
                                    out << thermo;
                                    out.close();

                                    outputThermo(Log::file());
                                    if (intermediateFlag)
                                    {
                                        std::string prefix = "chi=";
                                        prefix += std::to_string(current_chi);
                                        prefix += "_";
                                        for (int j = 1; j < proc[i]["ACAP"]["IntermediateOuput"].size(); ++j)
                                        {
                                            fieldIO("write",
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["Field"].asString(),
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["Format"].asString(),
                                                    proc[i]["ACAP"]["IntermediateOuput"][j]["OutputDirectory"].asString(),
                                                    proc[0]["CaseId"].asString(),
                                                    prefix);
                                        }
                                        intermediateFlag = false;
                                    }

                                    if (!point.empty() && (point[0] != start_chi)
                                        &&(dir && current_chi + inc >= point[0]
                                        || !dir && current_chi + inc <= point[0]))
                                    {
                                        current_chi = point[0];
                                        point.erase(point.begin());
                                        intermediateFlag = true;
                                    }
                                    else
                                    {
                                        current_chi += inc; // step forward
                                        if (abs(inc * fac) <= max_inc)
                                            inc *= fac;
                                        else
                                        {
                                            if (start_chi > end_chi)
                                                inc = -max_inc;
                                            else
                                                inc = max_inc;
                                        }
                                    }

                                }
                                else
                                {
                                    Log::file() << "Iterate has failed." << std::endl;
                                    
                                    std::string outFileName = proc[i]["ACAP"]["OutputDirectory"].asString();
                                    outFileName += "chi_";
                                    outFileName += proc[0]["CaseId"].asString();
                                    outFileName += "_out.json";
                                    std::ofstream out(outFileName);
                                    outputThermo(thermo);
                                    out << thermo;
                                    out.close();

                                    if (abs(inc) < min_inc)
                                    {
                                        Log::file() << "Smallest increment reached." << std::endl;
                                        // outRunfile.close();
                                    }
                                    else
                                    {
                                        current_chi -= inc;
                                        inc /= fac;
                                    }
                                }
                            }

                        }  
                    }
                    else if(!proc[i]["PhaseBoundaryPoints"].empty())
                    {
                        break;
                    }
                    else if(!proc[i]["PhaseBoundary"].empty())
                    {
                        break;
                    }
                    // else
                    // {
                    //     std::cout << "Unknown Command" << "\n";
                    // }
                }
            }


            /*
             * Read and execute commands from the default command file.
             */
            template <int D>
            void System<D>::readCommands()
            {
                if (fileMaster().commandFileName().empty())
                {
                    UTIL_THROW("Empty command file name");
                }
                readCommands(fileMaster().commandFile());
                fileMaster().commandFile().clear();
                fileMaster().commandFile().seekg(0);
            }

            /*
             * Read and execute commands from a specified command file.
             */
            template <int D>
            void System<D>::readCommands(std::istream &in)
            {
                UTIL_CHECK(isAllocated_)
                std::string command;
                std::string filename;

                bool readNext = true;
                while (readNext)
                {
                    in >> command;
                    Log::file() << command << std::endl;
                    if (command == "FINISH")
                    {
                        Log::file() << std::endl;
                        break;
                        readNext = false;
                    }
                    else if (command == "READ_W_BASIS")
                    {
#if DFT == 0
                        in >> filename;
                        Log::file() << " " << Str(filename, 20) << std::endl;
                        fieldIo().readFieldsBasis(filename, wFields());
                        fieldIo().convertBasisToRGrid(wFields(), wFieldsRGridPh());
                        hasWFields_ = true;
#endif
#if DFT == 1
                        Log::file() << "Cannot read w in in symmetrized Fourier basis for fast cosine transform."
                                    << std::endl;
                        exit(1);
#endif
                    }
                    else if (command == "READ_W_RGRID")
                    {
                        in >> filename;
                        Log::file() << " " << Str(filename, 20) << std::endl;
                        fieldIo().readFieldsRGrid(filename, wFieldsRGridPh());
                        hasWFields_ = true;
                    }
                    else if (command == "ITERATE")
                    {
                        Log::file() << std::endl;

                        // Read w fields in grid format iff not already set.
                        if (!hasWFields_)
                        {
                            std::cout << "Read w field before iteration." << std::endl;
                            exit(1);
                        }

                        // Attempt to iteratively solve SCFT equations
                        int fail = iterator().solve();
                        hasCFields_ = true;

                        if (!fail)
                        {
                            computeFreeEnergy();
                            outputThermo(Log::file());
                        }
                        else
                        {
                            computeFreeEnergy();
                            outputThermo(Log::file());
                            Log::file() << "Iterate has failed. Exiting " << std::endl;
                        }
                    }
                    else if (command == "CHI_PATH")
                    {
                        Log::file() << std::endl;

                        if (!hasWFields_)
                        {
                            in >> filename;
                            Log::file() << "Reading w fields from file: "
                                        << Str(filename, 20) << std::endl;
                            fieldIo().readFieldsRGrid(filename, wFieldsRGridPh());
                            hasWFields_ = true;
                        }

                        int i, j;
                        // start point, current point, and end point.
                        // end ponit is read from command file, while
                        // start point is read from param file.
                        double start_chi, current_chi, end_chi;
                        // current increment, maximum increment, minimum increment,
                        // and adpative factor, which are read from command file
                        double inc, max_inc, min_inc, fac;
                        // Is finished? Set false initially
                        bool isFinished = false;
                        // Is chi increasing? Set true initially
                        bool dir = true;

                        std::ofstream outRunfile;

                        in >> i;
                        in >> j;
                        in >> end_chi >> inc >> max_inc >> min_inc >> fac >> filename;
                        // in.ignore(5, '(');
                        outRunfile.open(filename, std::ios::app);
                        outRunfile << std::endl;
                        outRunfile << "        chi                           fHelmholtz                               U                                  -TS               error          cell params" << std::endl;
                        outRunfile << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
                        
                        start_chi = interaction().chi(i, j);
                        // Log::file() << i << "   " << j << std::endl;
                        // // Log::file() << start_chi << std::endl;
                        // Log::file() << end_chi << std::endl;
                        // Log::file() << inc << std::endl;
                        // Log::file() << fac << std::endl;
                        // Log::file() << filename << std::endl;
                        // exit(1);

                        if (start_chi == end_chi)
                        {
                            Log::file() << "The start point equals to the stop point." << std::endl;
                            exit(1);
                        }

                        if (start_chi > end_chi)
                        {
                            inc *= -1;
                            dir = false;
                        }

                        current_chi = start_chi;

                        while (!isFinished)
                        {
                            current_chi += inc; // step forward
                            if ((dir && current_chi >= end_chi) || ((!dir) && current_chi <= end_chi))
                            {
                                current_chi = end_chi;
                                isFinished = true;
                            }
                            interaction().chi_(i, j) = current_chi;
                            interaction().chi_(j, i) = current_chi;
                            unitCell().setParameters(unitCell().parameters()); // Reset cell parameters
                            // iteration

                            // Attempt to iteratively solve SCFT equations
                            Log::file() << std::endl;
                            Log::file() << "================================================" << std::endl;
                            Log::file() << "*Current chi = " << current_chi << "*" << std::endl
                                        << std::endl;
                            int fail = iterator().solve();
                            hasCFields_ = true;

                            if (!fail)
                            {
                                computeFreeEnergy();
                                outputThermo(Log::file());
                                outRunfile << Dbl(current_chi, 21, 10)
                                           << Dbl(fHelmholtz_, 35, 14)
                                           << Dbl(E_, 35, 14)
                                           << Dbl(fHelmholtz_-E_, 35, 14)
                                           << Dbl(iterator().final_error, 17, 6);

                                for (int i = 0; i < unitCell().nParameter(); ++i)
                                    outRunfile << Dbl(unitCell().parameter(i), 19, 8);
                                outRunfile << std::endl;

                                if (abs(inc * fac) <= max_inc)
                                    inc *= fac;
                                else
                                {
                                    if (start_chi > end_chi)
                                        inc = max_inc;
                                    else
                                        inc = -max_inc;
                                }
                            }
                            else
                            {
                                Log::file() << "Iterate has failed." << std::endl;
                                if (abs(inc) < min_inc)
                                {
                                    Log::file() << "Smallest increment reached." << std::endl;
                                    outRunfile.close();
                                }
                                else
                                {
                                    current_chi -= inc;
                                    inc /= fac;
                                }
                            }
                        }
                    }
                    else if (command == "bA_PATH")
                    {
                        Log::file() << std::endl;

                        double start_bA, current_bA, end_bA;
                        // current increment, maximum increment, minimum increment,
                        // and adpative factor, which are read from command file
                        double inc, max_inc, min_inc, fac;
                        int id;
                        // Is finished? Set false initially
                        bool isFinished = false;
                        // Is chi increasing? Set true initially
                        bool dir = true;

                        std::ofstream outRunfile;

                        in >> end_bA >> id >> inc >> max_inc >> min_inc >> fac >> filename;

                        outRunfile.open(filename, std::ios::app);
                        outRunfile << std::endl;
                        outRunfile << "         kuhn      monomer     fHelmholtz                 U               -TS                 error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                        outRunfile << std::endl;
                        start_bA = mixture().monomer(id).step();
                        current_bA = start_bA;

                        std::cout << "Run file name is " << filename << std::endl;
                        Log::file() << "Calculation along the path with respect to bA:"
                                    << std::endl;
                        Log::file() << "Starting point of bA: "
                                    << start_bA << std::endl;
                        Log::file() << " Current point of bA: "
                                    << current_bA << std::endl;
                        Log::file() << "    Stop point of bA: "
                                    << end_bA << std::endl;
                        Log::file() << "    Initial increment: "
                                    << inc << std::endl;

                        if (start_bA == end_bA)
                        {
                            Log::file() << "The start point equals to the stop point."
                                        << std::endl;
                            exit(1);
                        }
                        if (start_bA > end_bA)
                        {
                            inc *= -1;
                            dir = false;
                        }
                        while (!isFinished)
                        {
                            current_bA += inc;
                            if ((dir && current_bA >= end_bA) || ((!dir) && current_bA <= end_bA))
                            {
                                current_bA = end_bA;
                                isFinished = true;
                            }
                            double kuhn;
                            int monomerId;
                            for (int i = 0; i < this->mixture().nPolymer(); ++i)
                            {
                                for (int j = 0; j < this->mixture().polymer(i).nBlock(); ++j)
                                {
                                    monomerId = this->mixture().polymer(i).block(j).monomerId();  
                                    if (monomerId == id)
                                    {
                                        this->mixture().polymer(i).block(j).setKuhn(current_bA);
                                    }
                                }
                            }
                            // Attempt to iteratively solve SCFT equations
                            Log::file() << std::endl;
                            Log::file() << "================================================" << std::endl;
                            Log::file() << "*Current bA = " << current_bA << "*" << std::endl
                                        << std::endl;
                            int fail = iterator().solve();
                            hasCFields_ = true;
                            if (!fail)
                            {
                                computeFreeEnergy();
                                outputThermo(Log::file());
                                outRunfile << Dbl(current_bA, 19, 10)
                                           << Int(id, 5)
                                           << Dbl(fHelmholtz_, 23, 14)
                                           << Dbl(E_, 23, 14)
                                           << Dbl(fHelmholtz_-E_, 23, 14)
                                           << Dbl(iterator().final_error, 11, 2);
                                for (int i = 0; i < mixture().nParameter(); ++i)
                                    outRunfile << Dbl(unitCell().parameter(i), 17, 8);
                                outRunfile << std::endl;
                                if (abs(inc * fac) <= max_inc)
                                    inc *= fac;
                                else
                                {
                                    if (dir)
                                        inc = max_inc;
                                    else
                                        inc = -max_inc;
                                }
                            }
                            else
                            {
                                Log::file() << "Iterate has failed." << std::endl;
                                if (inc > min_inc)
                                {
                                    current_bA -= inc;
                                    inc /= fac;
                                }
                                else
                                {
                                    Log::file() << "Smallest increment reached." << std::endl;
                                    exit(1);
                                }
                            }
                        }
                    }
                    else if (command == "WRITE_W_BASIS")
                    {
                        UTIL_CHECK(hasWFields_)
                        in >> filename;
                        Log::file() << "  " << Str(filename, 20) << std::endl;
                        fieldIo().convertRGridToBasis(wFieldsRGridPh(), wFields());
                        fieldIo().writeFieldsBasis(filename, wFields());
                    }
                    else if (command == "WRITE_W_RGRID")
                    {
                        UTIL_CHECK(hasWFields_)
                        in >> filename;
                        Log::file() << "  " << Str(filename, 20) << std::endl;
                        fieldIo().writeFieldsRGrid(filename, wFieldsRGridPh());
                    }
                    else if (command == "WRITE_C_BASIS")
                    {
                        UTIL_CHECK(hasCFields_)
                        in >> filename;
                        Log::file() << "  " << Str(filename, 20) << std::endl;
                        fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());
                        fieldIo().writeFieldsBasis(filename, cFields());
                    }
                    else if (command == "WRITE_C_RGRID")
                    {
                        UTIL_CHECK(hasCFields_)
                        in >> filename;
                        Log::file() << "  " << Str(filename, 20) << std::endl;
#if DFT == 1
                        writeFullCRGrid(filename);
#else
                        fieldIo().writeFieldsRGrid(filename, cFieldsRGrid());
#endif
                    }
                    else if (command == "BASIS_TO_RGRID")
                    {
                        hasCFields_ = false;

                        std::string inFileName;
                        in >> inFileName;
                        Log::file() << " " << Str(inFileName, 20) << std::endl;

                        // fieldIo().readFieldsBasis(inFileName, cFields());
                        // fieldIo().convertBasisToRGrid(cFields(), cFieldsRGrid());

                        std::string outFileName;
                        in >> outFileName;
                        Log::file() << " " << Str(outFileName, 20) << std::endl;
                        // fieldIo().writeFieldsRGrid(outFileName, cFieldsRGrid());
                    }
                    else if (command == "RGRID_TO_BASIS")
                    {
                        hasCFields_ = false;

                        std::string inFileName;

                        in >> inFileName;
                        Log::file() << " " << Str(inFileName, 20) << std::endl;
                        // fieldIo().readFieldsRGrid(inFileName, cFieldsRGrid());

                        // fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());

                        std::string outFileName;
                        in >> outFileName;
                        Log::file() << " " << Str(outFileName, 20) << std::endl;
                        // fieldIo().writeFieldsBasis(outFileName, cFields());
                    }
                    else if (command == "KGRID_TO_RGRID")
                    {
                        hasCFields_ = false;

                        // Read from file in k-grid format
                        std::string inFileName;
                        in >> inFileName;
                        Log::file() << " " << Str(inFileName, 20) << std::endl;
                        // fieldIo().readFieldsKGrid(inFileName, cFieldsKGrid());

                        // Use FFT to convert k-grid r-grid
                        /* for (int i = 0; i < mixture().nMonomer(); ++i)
                         {
                             fft().inverseTransform(cFieldKGrid(i), cFieldRGrid(i));
                         }*/

                        // Write to file in r-grid format
                        std::string outFileName;
                        in >> outFileName;
                        Log::file() << " " << Str(outFileName, 20) << std::endl;
                        // fieldIo().writeFieldsRGrid(outFileName, cFieldsRGrid());
                    }
                    else if (command == "RHO_TO_OMEGA")
                    {

                        // Read c field file in r-grid format
                        std::string inFileName;
                        in >> inFileName;
                        Log::file() << " " << Str(inFileName, 20) << std::endl;
                        // fieldIo().readFieldsRGrid(inFileName, cFieldsRGrid());

                        // Compute w fields, excluding Lagrange multiplier contribution
                        // code is bad here, `mangled' access of data in array
                        /*for (int i = 0; i < mixture().nMonomer(); ++i)
                        {
                            assignUniformReal <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                            (wFieldRGrid(i).cDField(), 0, mesh().size());
                        }
                        for (int i = 0; i < mixture().nMonomer(); ++i)
                        {
                            for (int j = 0; j < mixture().nMonomer(); ++j)
                            {
                                pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                                (wFieldRGrid(i).cDField(), cFieldRGrid(j).cDField(),
                                 interaction().chi(i,j), mesh().size());
                            }
                        }*/

                        // Write w fields to file in r-grid format
                        std::string outFileName;
                        in >> outFileName;
                        Log::file() << " " << Str(outFileName, 20) << std::endl;
                        // fieldIo().writeFieldsRGrid(outFileName, wFieldsRGrid());
                    }
                    else if (command == "WRITE_THERMO")
                    {
                        std::string FileName, ParamFileName;
                        in >> FileName >> ParamFileName;
                        Log::file() << " " << Str(FileName, 20) << Str(ParamFileName, 20) << std::endl;
                        std::ofstream outRunfile;
                        outRunfile.open(FileName, std::ios::app);
                        outRunfile << std::endl;
                        outRunfile << " Parameter file : " << ParamFileName << std::endl;
                        outRunfile << std::endl;
                        outRunfile << "         fHelmholtz                 pressure                    TS                       U                  error   " << std::endl;
                        outRunfile << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

                        outRunfile << Dbl(fHelmholtz_, 25, 14)
                                   << Dbl(pressure_, 25, 14)
                                   << Dbl(S_, 25, 14)
                                   << Dbl(E_, 25, 14);
                        outRunfile << Dbl(iterator().final_error, 15, 4);
                        outRunfile.close();
                    }
                    else
                    {
                        Log::file() << "  Error: Unknown command  " << command << std::endl;
                        readNext = false;
                    }
                }
            }

            /*
             * Initialize Pscf::Homogeneous::Mixture homogeneous_ member.
             */
            template <int D>
            void System<D>::initHomogeneous()
            {
                // Set number of molecular species and monomers
                int nm = mixture().nMonomer();
                int np = mixture().nPolymer();
                // int ns = mixture().nSolvent();
                int ns = 0;
                UTIL_CHECK(homogeneous_.nMolecule() == np + ns)
                UTIL_CHECK(homogeneous_.nMonomer() == nm)

                // Allocate c_ work array, if necessary
                if (c_.isAllocated())
                {
                    UTIL_CHECK(c_.capacity() == nm)
                }
                else
                {
                    c_.allocate(nm);
                }

                int i;  // molecule index
                int j;  // monomer index
                int k;  // block or clump index
                int nb; // number of blocks
                int nc; // number of clumps

                // Loop over polymer molecule species
                for (i = 0; i < np; ++i)
                {
                    // Initial array of clump sizes
                    for (j = 0; j < nm; ++j)
                    {
                        c_[j] = 0.0;
                    }

                    // Compute clump sizes for all monomer types.
                    nb = mixture().polymer(i).nBlock();
                    for (k = 0; k < nb; ++k)
                    {
                        Block<D> &block = mixture().polymer(i).block(k);
                        j = block.monomerId();
                        c_[j] += block.length();
                    }

                    // Count the number of clumps of nonzero size
                    nc = 0;
                    for (j = 0; j < nm; ++j)
                    {
                        if (c_[j] > 1.0E-10)
                        {
                            ++nc;
                        }
                    }
                    homogeneous_.molecule(i).setNClump(nc);

                    // Set clump properties for this Homogeneous::Molecule
                    k = 0; // Clump index
                    for (j = 0; j < nm; ++j)
                    {
                        if (c_[j] > 1.0E-10)
                        {
                            homogeneous_.molecule(i).clump(k).setMonomerId(j);
                            homogeneous_.molecule(i).clump(k).setSize(c_[j]);
                            ++k;
                        }
                    }
                    homogeneous_.molecule(i).computeSize();
                }
            }

            /*
             * Compute Helmholtz free energy and pressure
             */
            template <int D>
            void System<D>::computeFreeEnergy()
            {
#if DFT == 0
                int nx = mesh().size();
                // GPU resources
                int nBlocks, nThreads;
                ThreadGrid::setThreadsLogical(nx, nBlocks, nThreads);

                fHelmholtz_ = 0.0;
                UR_ = 0.0;
                UCMP_ = 0;
                S_ = 0.0;
                // Compute ideal gas contributions to f_Helmholtz
                Polymer<D> *polymerPtr;
                double phi, mu, length, r;

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), 0.0, nx);

                int nm = mixture().nMonomer();

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), 0.0, nx);

                for (int i = 0; i < nm; ++i)
                {
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wFieldRGridPh(i).cDField(),
                                                                          cFieldRGrid(i).cDField(),
                                                                          workArray.cDField(),
                                                                          nx);

                    fHelmholtz_ -= gpuSum(workArray.cDField(), nx) / double(nx);
                }

                C_ = 0.0;
                int np = mixture().nPolymer();
                for (int i = 0; i < np; ++i)
                {
                    polymerPtr = &mixture().polymer(i);
                    phi = polymerPtr->phi();
                    mu = polymerPtr->mu();
                    // Recall: mu = ln(phi/q)
                    length = polymerPtr->length();
                    r = polymerPtr->length()/mixture().polymer(0).length();
                    fHelmholtz_ += phi * mu / r;
                    C_ += (1.0 - log(phi))*phi/r;
                    // fHelmholtz_ -= phi * log(polymerPtr->Q()) / length;
                }
                
                S_ = fHelmholtz_;

                for (int i = 0; i < nm; ++i)
                {
                    fft_.forwardTransform(cFieldRGrid(i), cFieldKGrid(i));
                }
                for (int i = 0; i < D; ++i)
                {
                    if (i < D - 1)
                    {
                        kMeshDimensions_[i] = mesh().dimensions()[i];
                    }
                    else
                    {
                        kMeshDimensions_[i] = mesh().dimensions()[i] / 2 + 1;
                    }
                }
                int kSize = 1;
                for (int i = 0; i < D; ++i)
                {
                    kSize *= kMeshDimensions_[i];
                }

                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        cudaConv<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArrayDft.cDField(),
                                                                          cFieldKGrid(i).cDField(),
                                                                          mixture().bu0().cDField(),
                                                                          kSize);
                        fft_.inverseTransform(workArrayDft, workArray);
                        inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(),
                                                                                     cFieldRGrid(j).cDField(),
                                                                                     nx);
             
                        UR_ += 0.5 * interaction().chi(i, j) * gpuSum(workArray.cDField(), nx)/double(nx);
                    }
                }

                fHelmholtz_ += UR_;
#if CMP == 1
                // UCMP_ -= fHelmholtz_;
                // fHelmholtz_ += UCMP_;
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cFieldTot_.cDField(), 0.0, nx);
                for (int i = 0; i < nm; ++i)
                {
                    pointWiseAdd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cFieldTot_.cDField(), cFieldRGrid(i).cDField(), nx);
                }
                mixture().computeBlockCMP(mesh(), cFieldsRGrid_, cFieldsKGrid_, fft());
                for (int i = 0; i < mixture().nUCompCMP(); ++i)
                {
                    UCMP_ += mixture().uBlockCMP(i);
                }
                fHelmholtz_ += UCMP_;
#endif
                
                mixture().computeBlockRepulsion(mesh(), cFieldsRGrid_, cFieldsKGrid_, fft());
                for (int i = 0; i < np; ++i)
                {
                    mixture().polymer(i).computeVertex(mesh());
                }
                mixture().computeBlockEntropy(wFieldsRGrid_ph_, mesh());
                // Compute pressure
                pressure_ = -fHelmholtz_;
                for (int i = 0; i < np; ++i)
                {
                    polymerPtr = &mixture().polymer(i);
                    phi = polymerPtr->phi();
                    mu = polymerPtr->mu();
                    // Recall: mu = ln(phi/q)
                    length = polymerPtr->length();
                    pressure_ += mu * phi / length;
                    // fHelmholtz_ -= phi * log(polymerPtr->Q()) / length;
                }
                
/*               double *k1, *k2, *k3;
                MeshIterator<D> iter;
                kMeshDimensions_ = wavelist().dimensions();
                kMeshDimensions_[2] = kMeshDimensions_[2]/2+1;
                iter.setDimensions(kMeshDimensions_);
                
                int kSize = 1;
                for (int i = 0; i < D; ++i)
                    kSize *= kMeshDimensions_[i];
                k1 = new double [kSize];
                k2 = new double [kSize];
                k3 = new double [kSize];

                for (iter.begin(); !iter.atEnd(); ++iter)
                {
                    k1[iter.rank()] = wavelist().minImage(iter.rank())[0];
                    k2[iter.rank()] = wavelist().minImage(iter.rank())[1];
                    k3[iter.rank()] = wavelist().minImage(iter.rank())[2];
                }

                int ns = mixture().polymer(0).block(0).ns();
                RDField<D> qA, tmp_q, tmp_dq;
                RDField<D> dqA1,dqA2,dqA3;
                RDFieldDft<D> qAk;
                RDFieldDft<D> dqAk;
                double *k1d,*k2d,*k3d;
                qA.allocate(nx*ns);
                dqA1.allocate(nx*ns);
                dqA2.allocate(nx*ns);
                dqA3.allocate(nx*ns);
                tmp_q.allocate(nx);
                tmp_dq.allocate(nx);
                qAk.allocate(kSize);
                dqAk.allocate(kSize);
                cudaMalloc((void **)&k1d, kSize * sizeof(cudaReal));
                cudaMalloc((void **)&k2d, kSize * sizeof(cudaReal));
                cudaMalloc((void **)&k3d, kSize * sizeof(cudaReal));
                cudaMemcpy(k1d, k1, sizeof(cudaReal) * kSize, cudaMemcpyHostToDevice);
                cudaMemcpy(k2d, k2, sizeof(cudaReal) * kSize, cudaMemcpyHostToDevice);
                cudaMemcpy(k3d, k3, sizeof(cudaReal) * kSize, cudaMemcpyHostToDevice);
                
                cudaMemcpy(qA.cDField(), mixture().polymer(0).block(0).propagator(0).qhead(), sizeof(cudaReal) * nx, cudaMemcpyHostToDevice);
                for (int s = 0; s < ns; ++s)
                {
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp_q.cDField(), qA.cDField()+s*nx, nx);
                    fft_.forwardTransform(tmp_q, qAk);

                    cudaImMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAk.cDField(), qAk.cDField(), k1d, kSize);
                    fft_.inverseTransform(dqAk,tmp_dq);
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqA1.cDField()+s*nx, tmp_dq.cDField(), nx);

                    cudaImMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAk.cDField(), qAk.cDField(), k2d, kSize);
                    fft_.inverseTransform(dqAk,tmp_dq);
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqA2.cDField()+s*nx, tmp_dq.cDField(), nx);

                    cudaImMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAk.cDField(), qAk.cDField(), k3d, kSize);
                    fft_.inverseTransform(dqAk,tmp_dq);
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqA3.cDField()+s*nx, tmp_dq.cDField(), nx);
                    
                    if (s+1<ns)
                    {
                        mixture().polymer(0).block(0).step(qA.cDField()+s*nx, tmp_q.cDField());
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qA.cDField()+(s+1)*nx, tmp_q.cDField(), nx);
                    }
                }

                RDField<D> qAs;
                RDField<D> dqAs1,dqAs2,dqAs3;
                qAs.allocate(nx*ns);
                dqAs1.allocate(nx*ns);
                dqAs2.allocate(nx*ns);
                dqAs3.allocate(nx*ns);

                cudaMemcpy(qAs.cDField(), mixture().polymer(0).block(0).propagator(1).qhead(), sizeof(cudaReal) * nx, cudaMemcpyHostToDevice);
                for (int s = 0; s < ns; ++s)
                {
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp_q.cDField(), qAs.cDField()+s*nx, nx);
                    fft_.forwardTransform(tmp_q, qAk);

                    cudaImMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAk.cDField(), qAk.cDField(), k1d, kSize);
                    fft_.inverseTransform(dqAk,tmp_dq);
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAs1.cDField()+s*nx, tmp_dq.cDField(), nx);

                    cudaImMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAk.cDField(), qAk.cDField(), k2d, kSize);
                    fft_.inverseTransform(dqAk,tmp_dq);
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAs2.cDField()+s*nx, tmp_dq.cDField(), nx);

                    cudaImMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAk.cDField(), qAk.cDField(), k3d, kSize);
                    fft_.inverseTransform(dqAk,tmp_dq);
                    assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dqAs3.cDField()+s*nx, tmp_dq.cDField(), nx);
                    
                    if (s+1<ns)
                    {
                        mixture().polymer(0).block(0).step(qAs.cDField()+s*nx, tmp_q.cDField());
                        assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qAs.cDField()+(s+1)*nx, tmp_q.cDField(), nx);
                    }
                }

                cudaReal *px, *py, *pz;
                cudaMalloc((void **)&px, nx * sizeof(cudaReal));
                cudaMalloc((void **)&py, nx * sizeof(cudaReal));
                cudaMalloc((void **)&pz, nx * sizeof(cudaReal));
                cudaMemset(px, 0, nx*sizeof(cudaReal));
                cudaMemset(py, 0, nx*sizeof(cudaReal));
                cudaMemset(pz, 0, nx*sizeof(cudaReal));
                for (int s = 0; s < ns; ++s)
                {
                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(px, qAs.cDField()+s*nx, dqA3.cDField()+(ns-1-s)*nx,  -1.0/mixture().polymer(0).Q()/6.0, nx);
                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(px, qA.cDField()+s*nx,  dqAs3.cDField()+(ns-1-s)*nx,  1.0/mixture().polymer(0).Q()/6.0, nx);

                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(py, qAs.cDField()+s*nx, dqA2.cDField()+(ns-1-s)*nx,  -1.0/mixture().polymer(0).Q()/6.0, nx);
                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(py, qA.cDField()+s*nx,  dqAs2.cDField()+(ns-1-s)*nx,  1.0/mixture().polymer(0).Q()/6.0, nx);

                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(pz, qAs.cDField()+s*nx, dqA1.cDField()+(ns-1-s)*nx,  -1.0/mixture().polymer(0).Q()/6.0, nx);
                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(pz, qA.cDField()+s*nx,  dqAs1.cDField()+(ns-1-s)*nx,  1.0/mixture().polymer(0).Q()/6.0, nx);
                }

                cudaReal *pxc, *pyc, *pzc;
                pxc = new cudaReal [nx];
                pyc = new cudaReal [nx];
                pzc = new cudaReal [nx];
                cudaMemcpy(pxc, px, sizeof(cudaReal) * nx, cudaMemcpyDeviceToHost);
                cudaMemcpy(pyc, py, sizeof(cudaReal) * nx, cudaMemcpyDeviceToHost);
                cudaMemcpy(pzc, pz, sizeof(cudaReal) * nx, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 32*32*32; ++i)
                    std::cout << pxc[i] << "    " << pyc[i] << "    " << pzc[i] << "\n";
                delete [] k1;
                delete [] k2;
                delete [] k3;
                delete [] pxc;
                delete [] pyc;
                delete [] pzc;
                cudaFree (k1d);
                cudaFree (k2d);
                cudaFree (k3d);
                */ 
#endif

#if DFT == 1
                fHelmholtz_ = 0.0;
                E_ = 0.0;
                S_ = 0.0;
                // Compute ideal gas contributions to f_Helmholtz
                Polymer<D> *polymerPtr;
                double phi, mu, length, tmp;

                int nm = mixture().nMonomer();
                // std::cout << "nm = " << nm << "\n";
                int nx = mesh().size();
                ;

                // GPU resources
                int nBlocks, nThreads;
                ThreadGrid::setThreadsLogical(nx, nBlocks, nThreads);

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), 0.0, nx);

                for (int i = 0; i < nm; ++i)
                {
                    pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wFieldRGridPh(i).cDField(),
                                                                          cFieldRGrid(i).cDField(),
                                                                          workArray.cDField(),
                                                                          nx);

                    fHelmholtz_ -= gpuSum(workArray.cDField(), nx) / double(nx);
                }

                int np = mixture().nPolymer();
                for (int i = 0; i < np; ++i)
                {
                    polymerPtr = &mixture().polymer(i);
                    phi = polymerPtr->phi();
                    mu = polymerPtr->mu();
                    // Recall: mu = ln(phi/q)
                    // std::cout << "mu = " << mu << "\n";
                    length = polymerPtr->length();
                    fHelmholtz_ += phi * (mu - 1) / length;
                }

                S_ = fHelmholtz_;

                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), 0.5 * interaction().chi(i, j), nx);
                        inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), cFieldRGrid(i).cDField(), nx);
                        inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), cFieldRGrid(j).cDField(), nx);

                        E_ += gpuSum(workArray.cDField(), nx) / double(nx);
                    }
                }

                fHelmholtz_ += E_;

                // Compute pressure
                pressure_ = -fHelmholtz_;
                for (int i = 0; i < np; ++i)
                    pressure_ += mu * phi / length;

#endif
            }

            template <int D>
            void System<D>::outputThermo(std::ostream &out)
            {
                out << std::endl;
                out << "fHelmholtz  = " << Dbl(fHelmholtz(), 21, 16) << std::endl;
                out << "U_chi       = " << Dbl(UR_, 21, 16) << std::endl;
                out << "U_CMP       = " << Dbl(UCMP_, 21, 16) << std::endl;
                out << "-TS         = " << Dbl(fHelmholtz() - UR_ - UCMP_, 21, 16) << std::endl;
                int np = mixture().nPolymer();
                out << std::endl;
                out << "Components of U due to Flory-Huggins repulsion:" << std::endl;
                out << "    polymer1   block1   polymer2   block2      -U(block)" << std::endl;
                
                for (int i = 0; i < mixture().nUCompChi(); ++i)
                {
                    out << Int(mixture().uBlockList(i)[0],5) <<"  " 
                        << Int(mixture().uBlockList(i)[1],9) <<"  " 
                        << Int(mixture().uBlockList(i)[2],7) <<"  " 
                        << Int(mixture().uBlockList(i)[3],9) <<"  " 
                        <<  "   "
                        << Dbl(mixture().uBlockRepulsion(i), 28, 16)
                        << std::endl;
                }
#if CMP==1
                out << std::endl;
                out << "Components of U due to compressibility:" << std::endl;
                out << "    polymer1   block1   polymer2   block2      -U(block)" << std::endl;
                
                for (int i = 0; i < mixture().nUCompCMP(); ++i)
                {
                    out << Int(mixture().uBlockList(i)[0],5) <<"  " 
                        << Int(mixture().uBlockList(i)[1],9) <<"  " 
                        << Int(mixture().uBlockList(i)[2],7) <<"  " 
                        << Int(mixture().uBlockList(i)[3],9) <<"  " 
                        <<  "   "
                        << Dbl(mixture().uBlockCMP(i), 28, 16)
                        << std::endl;
                }
#endif
                out << std::endl;
                out << "Components of -TS:" << std::endl;
                out << "    polymer   vertex       -TS(vertex)" << std::endl;
                for (int i = 0; i < np; ++i)
                {
                    int nv = mixture().polymer(i).nVertex();
                    for (int v = 0; v < nv; ++v)
                    {
                        out << Int(i,5) <<"  " << Int(v,8)
                            <<  "   "
                            << Dbl(mixture().polymer(i).sV(v), 31, 16)
                            << std::endl;
                    }
                }
                out << std::endl;
                out << "    polymer   block        -TS(block)" << std::endl;
                int idx = 0;
                for (int i = 0; i < np; ++i)
                {
                    int nb = mixture().polymer(i).nBlock();
                    for (int b = 0; b < nb; ++b)
                    {
                        out << Int(i,5) <<"  " << Int(b,8)
                            <<  "   "
                            << Dbl(mixture().sBlock(idx), 31, 16)
                            << std::endl;
                        ++idx;
                    }
                }
                out << std::endl;

                out << std::endl;

                out << "Polymers:" << std::endl;
                out << "    i"
                    << "        phi[i]      "
                    << "        mu[i]       "
                    << std::endl;
                for (int i = 0; i < mixture().nPolymer(); ++i)
                {
                    out << Int(i, 5)
                        << "  " << Dbl(mixture().polymer(i).phi(), 18, 11)
                        << "  " << Dbl(mixture().polymer(i).mu(), 18, 11)
                        << std::endl;
                }
                out << std::endl;
            }

            template <int D>
            void System<D>::outputThermo(Json::Value &thermo)
            {   
                Json::Value out;

                for (int i = 0; i < mixture().nMonomer(); ++i)
                {
                    for (int j = i; j < mixture().nMonomer(); ++j)
                    {
                        Json::Value tmp;
                        tmp[0] = i;
                        tmp[1] = j;
                        if (interaction().chi(i,j) != 0)
                        {
                            tmp[2] = interaction().chi(i,j);  
                            out["Chi"].append(tmp); 
                        }
                    }
                }

                for (int i = 0; i < mixture().nMonomer(); ++i)
                {
                    
                    Json::Value tmp;
                    tmp[0] = i;
                    tmp[1] = double(mixture().monomer(i).step());  
                    out["SegmentLength"].append(tmp); 
                 
                }

                Json::Value HelmholtzFreeEnergy;

                Json::Value InternalEnergyContribution;

                HelmholtzFreeEnergy["Total"] = Json::Value(fHelmholtz());

                InternalEnergyContribution["Total"] = Json::Value(UR_ + UCMP_);

                Json::Value FloryHugginsRepulsion;

                FloryHugginsRepulsion["Total"] = Json::Value(UR_);

                for (int i = 0; i < mixture().nUCompChi(); ++i)
                {
                    Json::Value tmp;
                    tmp[0] = mixture().uBlockList(i)[0];
                    tmp[1] = mixture().uBlockList(i)[1];
                    tmp[2] = mixture().uBlockList(i)[2];
                    tmp[3] = mixture().uBlockList(i)[3];
                    tmp[4] = mixture().uBlockRepulsion(i);
                    FloryHugginsRepulsion["Components"].append(tmp);
                }
                
                Json::Value Compressibility  = Json::Value(UCMP_);

                InternalEnergyContribution["Compressibility"] = Json::Value(Compressibility);

                InternalEnergyContribution["FloryHugginsRepulsion"] = Json::Value(FloryHugginsRepulsion);

                HelmholtzFreeEnergy["InternalEnergyContribution"] = Json::Value(InternalEnergyContribution);

                Json::Value EntropyContribution;

                EntropyContribution["Total"] = Json::Value(S_);

                for (int i = 0; i < mixture().nPolymer(); ++i)
                {
                    int nv = mixture().polymer(i).nVertex();
                    for (int v = 0; v < nv; ++v)
                    {
                        Json::Value tmp;
                        tmp[0] = i;
                        tmp[1] = v;
                        tmp[2] = mixture().polymer(i).sV(v);
                        EntropyContribution["VertexComponent"].append(tmp);
                    }
                }
                int idx = 0;
                for (int i = 0; i < mixture().nPolymer(); ++i)
                {
                    int nb = mixture().polymer(i).nBlock();
                    for (int b = 0; b < nb; ++b)
                    {
                        Json::Value tmp;
                        tmp[0] = i;
                        tmp[1] = b;
                        tmp[2] = mixture().sBlock(idx);
                        EntropyContribution["BlockComponent"].append(tmp);
                        ++idx;
                    }
                }

                HelmholtzFreeEnergy["EntropyContribution"] = Json::Value(EntropyContribution);

                out["HelmholtzFreeEnergy"] = Json::Value(HelmholtzFreeEnergy);

                Json::Value C;
                out["C"] = Json::Value(C_);

                for (int p = 0; p < mixture().nPolymer(); ++p)
                {   
                    Json::Value tmp;
                    tmp[0] = p;
                    tmp[1] = mixture().polymer(p).mu();  
                    out["ChemicalPotential"].append(tmp); 
                }

                for (int p = 0; p < mixture().nPolymer(); ++p)
                {   
                    Json::Value tmp;
                    tmp[0] = p;
                    tmp[1] = mixture().polymer(p).phi();  
                    out["Phi"].append(tmp); 
                }

                Json::Value tmp;
                tmp[0] = groupName();
                for (int i = 1; i <= mixture().nParameter(); ++i)
                {
                    tmp[i] = unitCell().parameter(i-1);
                }
                out["Unitcell"].append(tmp);

                thermo.append(out);
            }

#if DFT == 1
            template <int D>
            void System<D>::writeFullCRGrid(std::string filename)
            {
                int nMonomer = mixture().nMonomer();
                int fullSize;
                DArray<CField> cFieldsRGridFull;
                cFieldsRGridFull.allocate(nMonomer);
                cudaReal *tmpFull, *tmp;

                IntVec<D> fullDim;
                IntVec<D> CrysDim;

                if (D == 3)
                {
                    CrysDim[0] = mesh().dimensions()[0];
                    CrysDim[1] = mesh().dimensions()[1];
                    CrysDim[2] = mesh().dimensions()[2];
                    fullDim[0] = mesh().dimensions()[0] * 2;
                    fullDim[1] = mesh().dimensions()[1] * 2;
                    fullDim[2] = mesh().dimensions()[2] * 2;
                    fullSize = fullDim[0] * fullDim[1] * fullDim[2];
                }
                else
                {
                    cFieldsRGridFull.deallocate();
                    std::cout << "Crystallographic is for 3D unitcell only so far. \n";
                    exit(1);
                }

                for (int i = 0; i < nMonomer; ++i)
                {
                    cFieldsRGridFull[i].allocate(fullDim);
                    tmpFull = new cudaReal[fullSize];
                    tmp = new cudaReal[CrysDim[0] * CrysDim[1] * CrysDim[2]];

                    cudaMemcpy(tmp, cFieldsRGrid_[i].cDField(), sizeof(cudaReal) * CrysDim[0] * CrysDim[1] * CrysDim[2], cudaMemcpyDeviceToHost);

                    for (int z = 0; z < CrysDim[2]; ++z)
                    {
                        for (int y = 0; y < CrysDim[1]; ++y)
                        {
                            for (int x = 0; x < CrysDim[0]; ++x)
                            {
                                int idx = x + CrysDim[0] * y + CrysDim[0] * CrysDim[1] * z;
                                int idx1 = x + fullDim[0] * y + fullDim[0] * fullDim[1] * z;
                                int idx2 = (fullDim[0] - x - 1) + fullDim[0] * y + fullDim[0] * fullDim[1] * z;
                                int idx3 = x + fullDim[0] * (fullDim[1] - y - 1) + fullDim[0] * fullDim[1] * z;
                                int idx4 = x + fullDim[0] * y + fullDim[0] * fullDim[1] * (fullDim[2] - z - 1);
                                int idx5 = x + fullDim[0] * (fullDim[1] - y - 1) + fullDim[0] * fullDim[1] * (fullDim[2] - z - 1);
                                int idx6 = (fullDim[0] - x - 1) + fullDim[0] * y + fullDim[0] * fullDim[1] * (fullDim[2] - z - 1);
                                int idx7 = (fullDim[0] - x - 1) + (fullDim[1] - y - 1) * fullDim[0] + fullDim[0] * fullDim[1] * z;
                                int idx8 = (fullDim[0] - x - 1) + (fullDim[1] - y - 1) * fullDim[0] + fullDim[0] * fullDim[1] * (fullDim[2] - z - 1);
                                tmpFull[idx1] = tmp[idx];
                                tmpFull[idx2] = tmp[idx];
                                tmpFull[idx3] = tmp[idx];
                                tmpFull[idx4] = tmp[idx];
                                tmpFull[idx5] = tmp[idx];
                                tmpFull[idx6] = tmp[idx];
                                tmpFull[idx7] = tmp[idx];
                                tmpFull[idx8] = tmp[idx];
                            }
                        }
                    }
                    cudaMemcpy(cFieldsRGridFull[i].cDField(), tmpFull, sizeof(cudaReal) * fullDim[0] * fullDim[1] * fullDim[2], cudaMemcpyHostToDevice);
                    delete[] tmpFull;
                    delete[] tmp;
                }
                mesh().setDimensions(fullDim);
                fieldIo().writeFieldsRGrid(filename, cFieldsRGridFull);
                mesh().setDimensions(CrysDim);
                cFieldsRGridFull.deallocate();
            }
#endif
        }
    }
}
#endif
