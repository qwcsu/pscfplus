#ifndef D_SYSTEM_TPP
#define D_SYSTEM_TPP

#include "System.h"

#include <pspg/math/GpuHeaders.h>

#include <pscf/crystal/shiftToMinimum.h>

#include <util/format/Str.h>
#include <util/format/Int.h>
#include <util/format/Dbl.h>

#include <string>
#include <getopt.h>
#include <algorithm>
#include <vector>

int THREADS_PER_BLOCK;
int NUMBER_OF_BLOCKS;

namespace Pscf
{
    namespace Pspg
    {
        namespace Discrete
        {
            using namespace Util;

            template <int D>
            System<D>::System()
                : DMixture_(),
                  mesh_(),
                  unitCell_(),
                  fileMaster_(),
                  interactionPtr_(nullptr),
                  iteratorPtr_(0),
                  basisPtr_(),
                  wavelistPtr_(0),
                  fieldIo_(),
                  wFields_(),
                  cFields_(),
                  hasMixture_(false),
                  hasUnitCell_(false),
                  isAllocated_(false),
                  hasWFields_(false),
                  hasCFields_(false)
            {
                setClassName("System");

                interactionPtr_ = new ChiInteraction();
                wavelistPtr_ = new WaveList<D>();
                basisPtr_ = new Basis<D>();
                iteratorPtr_ = new AmIterator<D>(this);
            }

            template <int D>
            System<D>::~System()
            {
                delete interactionPtr_;
                delete wavelistPtr_;
                delete basisPtr_;
                delete iteratorPtr_;
            }

            template <int D>
            void System<D>::setOptions(int argc, char **argv)
            {
                bool eFlag = false; // echo
                bool pFlag = false; // param file
                bool cFlag = false; // command file
                bool iFlag = false; // input prefix
                bool oFlag = false; // output prefix
                char *pArg = nullptr;
                char *cArg = nullptr;
                char *iArg = nullptr;
                char *oArg = nullptr;
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
                else
                {
                    fileMaster().setParamFileName(std::string("param"));
                }

                // If option -c, set command file name
                if (cFlag)
                {
                    fileMaster().setCommandFileName(std::string(cArg));
                }
                else
                {
                    fileMaster().setParamFileName(std::string("command"));
                }

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
                    fieldFileName += "_omega.bf";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().readFieldsBasis(fieldFileName, wFields());
                    fieldIo().convertBasisToRGrid(wFields(), wFieldsRGrid());
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
                    fieldFileName += "_omega.bf";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().convertRGridToBasis(wFieldsRGrid(), wFields());
                    fieldIo().writeFieldsBasis(fieldFileName, wFields());
                }
                else if (   io == "read"
                         && type == "omega"
                         && format == "real")
                {
                    Log::file() << std::endl;
                    Log::file() << "Reading omega field in real-space:" << std::endl;
                    std::string fieldFileName = dir;
                    fieldFileName += caseid;
                    fieldFileName += "_omega.rf";
                    Log::file() << " " << Str(fieldFileName, 20) << std::endl;
                    fieldIo().readFieldsRGrid(fieldFileName, wFieldsRGrid());
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
                    fieldFileName += "_omega.rf";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().writeFieldsRGrid(fieldFileName, wFieldsRGrid());
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
                    fieldFileName += "_phi.rf";
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
                    fieldFileName += "_phi.bf";
                    Log::file() << Str(fieldFileName, 20) << std::endl;
                    Log::file() << std::endl;
                    fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());
                    fieldIo().writeFieldsBasis(fieldFileName, cFields());
                }
                else
                {
                    std::cout << "      I/O type should be either read or write." << std::endl;
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
                        if (s == "chi")
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
                                                inc = max_inc;
                                            else
                                                inc = -max_inc;
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
                                
                                double kuhn1, kuhn2;
                                int monomerId1, monomerId2;
                                for (int p = 0; p < this->dmixture().nPolymer(); ++p)
                                {
                                    for (int b = 0; b < this->dmixture().polymer(p).nBond(); ++b)
                                    {
                                        monomerId1 = this->dmixture().polymer(p).bond(b).monomerId(0);
                                        monomerId2 = this->dmixture().polymer(p).bond(b).monomerId(1);
                                        if (monomerId1 == id)
                                        {
                                            kuhn1 = current_b;
                                            this->dmixture().monomer(id).setStep(current_b);
                                        }
                                        else
                                        {
                                            kuhn1 = this->dmixture().monomer(monomerId1).step();
                                        }
                                        if (monomerId2 == id)
                                        {
                                            kuhn2 = current_b;
                                            this->dmixture().monomer(id).setStep(current_b);
                                        }
                                        else
                                        {
                                            kuhn2 = this->dmixture().monomer(monomerId2).step();
                                        }
                                        this->dmixture().polymer(p).bond(b).setKuhn(kuhn1, kuhn2);
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
                                                inc = max_inc;
                                            else
                                                inc = -max_inc;
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
                            
                    }
                    else if(!proc[i]["PhaseBoundaryPoints"].empty())
                    {
                        break;
                    }
                    // else if(!proc[i]["PhaseBoundary"].empty())
                    // {
                    //     break;
                    // }
                    else
                    {
                        std::cout << "Unknown Command" << "\n";
                    }
                }
            }


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
                        readNext = false;
                    }
                    else if (command == "READ_W_BASIS")
                    {
                        in >> filename;
                        Log::file() << " " << Str(filename, 20) << std::endl;
                        fieldIo().readFieldsBasis(filename, wFields());
                        fieldIo().convertBasisToRGrid(wFields(), wFieldsRGrid());
                        hasWFields_ = true;

                        // cudaReal *a;
                        // a = new cudaReal [10];
                        // cudaMemcpy(a, wField(0).cDField(), sizeof(cudaReal)*10, cudaMemcpyDeviceToHost);
                        // for (int i = 0; i < 10; ++i)
                        //     std::cout << a[i] << "\n";
                        // std::cout << "\n";
                        // cudaMemcpy(a, wField(1).cDField(), sizeof(cudaReal)*10, cudaMemcpyDeviceToHost);
                        // for (int i = 0; i < 10; ++i)
                        //     std::cout << a[i] << "\n";
                        // exit(1);
                    }
                    else if (command == "READ_W_RGRID")
                    {
                        in >> filename;
                        Log::file() << " " << Str(filename, 20) << std::endl;
                        fieldIo().readFieldsRGrid(filename, wFieldsRGrid());
                        hasWFields_ = true;
                    }
                    else if (command == "ITERATE")
                    {
                        Log::file() << std::endl;

                        // Read w fields in grid format iff not already set.
                        if (!hasWFields_)
                        {
                            in >> filename;
                            Log::file() << "Reading w fields from file: "
                                        << Str(filename, 20) << std::endl;

                            fieldIo().readFieldsRGrid(filename, wFieldsRGrid());
                            hasWFields_ = true;
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
                            Log::file() << "Iterate has failed. Exiting " << std::endl;
                        }
                    }
                    else if (command == "Nkp_PATH")
                    {
                        Log::file() << std::endl;
                        if (!hasWFields_)
                        {
                            std::cout << "No wFields input"
                                      << "\n";
                            exit(1);
                        }
                        // start point, current point, and end point.
                        // end ponit is read from command file, while
                        // start point is read from param file.
                        double start_k, current_k, end_k;
                        // current increment, maximum increment, minimum increment,
                        // and adpative factor, which are read from command file
                        double inc, max_inc, min_inc, fac;
                        // Is finished? Set false initially
                        bool isFinished = false;
                        // Is chi increasing? Set true initially
                        bool dir = true;

                        std::ofstream outRunfile;

                        in >> end_k >> inc >> max_inc >> min_inc >> fac >> filename;

                        outRunfile.open(filename, std::ios::app);

                        outRunfile << std::endl;
                        outRunfile << "        kpN               fHelmholtz                 U                    UAB                    UCMP                   S                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                        outRunfile << std::endl;

                        start_k = this->dmixture().kpN();
                        current_k = start_k;

                        std::cout << "Run file name is " << filename << std::endl;

                        Log::file() << "Calculation along the path with respect to chiN:"
                                    << std::endl;
                        Log::file() << "Starting point of chiN: "
                                    << start_k << std::endl;
                        Log::file() << " Current point of chiN: "
                                    << current_k << std::endl;
                        Log::file() << "    Stop point of chiN: "
                                    << end_k << std::endl;
                        Log::file() << "     Initial increment: "
                                    << inc << std::endl;

                        if (start_k == end_k)
                        {
                            Log::file() << "The start point equals to the stop point."
                                        << std::endl;
                            exit(1);
                        }

                        if (start_k > end_k)
                        {
                            inc *= -1;
                            dir = false;
                        }
                        while (!isFinished)
                        {
                            // Step "forward" (of courese inc can be negative)
                            current_k += inc;
                            if ((dir && current_k >= end_k) || ((!dir) && current_k <= end_k))
                            {
                                current_k = end_k;
                                isFinished = true;
                            }

                            dmixture().setKpN(current_k);

                            // Attempt to iteratively solve SCFT equations
                            Log::file() << std::endl;
                            Log::file() << "================================================" << std::endl;
                            Log::file() << "*Current kappa = " << current_k << "*" << std::endl
                                        << std::endl;
                            int fail = iterator().solve();
                            hasCFields_ = true;

                            if (!fail)
                            {
                                computeFreeEnergy();
                                outputThermo(Log::file());

                                outRunfile << Dbl(current_k, 19, 10)
                                           << Dbl(fHelmholtz_, 23, 14)
                                           << Dbl(U_, 23, 14)
                                           << Dbl(UAB_, 23, 14)
                                           << Dbl(UCMP_, 23, 14)
                                           << Dbl(S_[0] + S_[1], 23, 14)
                                           << Dbl(S_[0], 23, 14)
                                           << Dbl(S_[1], 23, 14)
                                           << Dbl(iterator().final_error, 11, 2);

                                for (int i = 0; i < dmixture().nParameter(); ++i)
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
                                    current_k -= inc;
                                    inc /= fac;
                                }
                                else
                                {
                                    Log::file() << "Smallest increment reached." << std::endl;
                                    exit(1);
                                }
                            }
                        }
                        outRunfile.close();
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
                        start_bA = this->dmixture().monomer(id).step();
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
                            double kuhn1, kuhn2;
                            int monomerId1, monomerId2;
                            for (int i = 0; i < this->dmixture().nPolymer(); ++i)
                            {
                                for (int j = 0; j < this->dmixture().polymer(i).nBond(); ++j)
                                {
                                    monomerId1 = this->dmixture().polymer(i).bond(j).monomerId(0);
                                    monomerId2 = this->dmixture().polymer(i).bond(j).monomerId(1);
                                    if (monomerId1 == id)
                                    {
                                        kuhn1 = current_bA;
                                        this->dmixture().monomer(id).setStep(current_bA);
                                    }
                                    else
                                    {
                                        kuhn1 = this->dmixture().monomer(monomerId1).step();
                                    }
                                    if (monomerId2 == id)
                                    {
                                        kuhn2 = current_bA;
                                        this->dmixture().monomer(id).setStep(current_bA);
                                    }
                                    else
                                    {
                                        kuhn2 = this->dmixture().monomer(monomerId2).step();
                                    }
                                    this->dmixture().polymer(i).bond(j).setKuhn(kuhn1, kuhn2);
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
                                           << Dbl(U_, 23, 14)
                                           << Dbl(fHelmholtz_-U_, 23, 14)
                                           << Dbl(iterator().final_error, 11, 2);
                                for (int i = 0; i < dmixture().nParameter(); ++i)
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
                    else if (command == "CHI_PATH")
                    {
                        Log::file() << std::endl;
                        if (!hasWFields_)
                        {
                            std::cout << "No wFields input"
                                      << "\n";
                            exit(1);
                        }
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

                        in >> end_chi >> inc >> max_inc >> min_inc >> fac >> filename;

                        outRunfile.open(filename, std::ios::app);

                        outRunfile << std::endl;
                        outRunfile << "        chiN               fHelmholtz                 U                    UAB                    UCMP                   S                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                        outRunfile << std::endl;

                        start_chi = this->interactionPtr_->chi(0, 1);
                        current_chi = start_chi;

                        std::cout << "Run file name is " << filename << std::endl;

                        Log::file() << "Calculation along the path with respect to chiN:"
                                    << std::endl;
                        Log::file() << "Starting point of chiN: "
                                    << start_chi << std::endl;
                        Log::file() << " Current point of chiN: "
                                    << current_chi << std::endl;
                        Log::file() << "    Stop point of chiN: "
                                    << end_chi << std::endl;
                        Log::file() << "     Initial increment: "
                                    << inc << std::endl;

                        if (start_chi == end_chi)
                        {
                            Log::file() << "The start point equals to the stop point."
                                        << std::endl;
                            exit(1);
                        }

                        if (start_chi > end_chi)
                        {
                            inc *= -1;
                            dir = false;
                        }
                        while (!isFinished)
                        {
                            // Step "forward" (of courese inc can be negative)
                            current_chi += inc;
                            if ((dir && current_chi >= end_chi) || ((!dir) && current_chi <= end_chi))
                            {
                                current_chi = end_chi;
                                isFinished = true;
                            }
                            this->interactionPtr_->setChi(0, 1, current_chi);
                            this->interactionPtr_->setChi(1, 0, current_chi);
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

                                outRunfile << Dbl(current_chi, 19, 10)
                                           << Dbl(fHelmholtz_, 23, 14)
                                           << Dbl(U_, 23, 14)
                                           << Dbl(UAB_, 23, 14)
                                           << Dbl(UCMP_, 23, 14)
                                           << Dbl(S_[0] + S_[1], 23, 14)
                                           << Dbl(S_[0], 23, 14)
                                           << Dbl(S_[1], 23, 14)
                                           << Dbl(iterator().final_error, 11, 2);

                                for (int i = 0; i < dmixture().nParameter(); ++i)
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
                                    current_chi -= inc;
                                    inc /= fac;
                                }
                                else
                                {
                                    Log::file() << "Smallest increment reached." << std::endl;
                                    exit(1);
                                }
                            }
                        }
                        outRunfile.close();
                    }
                    else if (command == "WRITE_W_BASIS")
                    {
                        UTIL_CHECK(hasWFields_)
                        in >> filename;
                        Log::file() << "  " << Str(filename, 20) << std::endl;
                        fieldIo().convertRGridToBasis(wFieldsRGrid(), wFields());
                        fieldIo().writeFieldsBasis(filename, wFields());
                    }
                    else if (command == "WRITE_W_RGRID")
                    {
                        UTIL_CHECK(hasWFields_)
                        in >> filename;
                        Log::file() << "  " << Str(filename, 20) << std::endl;
                        fieldIo().writeFieldsRGrid(filename, wFieldsRGrid());
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
                        std::cout << "\n";
                        in >> filename;
                        Log::file() << "  " << Str(filename, 20) << std::endl;
                        fieldIo().writeFieldsRGrid(filename, cFieldsRGrid());
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
                        fieldIo().readFieldsRGrid(inFileName, cFieldsRGrid());

                        fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());

                        std::string outFileName;
                        in >> outFileName;
                        Log::file() << " " << Str(outFileName, 20) << std::endl;
                        fieldIo().writeFieldsBasis(outFileName, cFields());
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
                        for (int i = 0; i < dmixture().nMonomer(); ++i)
                        {
                            assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wFieldRGrid(i).cDField(), 0, mesh().size());
                        }
                        for (int i = 0; i < dmixture().nMonomer(); ++i)
                        {
                            for (int j = 0; j < dmixture().nMonomer(); ++j)
                            {
                                pointWiseAddScale<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(wFieldRGrid(i).cDField(), cFieldRGrid(j).cDField(),
                                                                                           interaction().chi(i, j), mesh().size());
                            }
                        }

                        // Write w fields to file in r-grid format
                        std::string outFileName;
                        in >> outFileName;
                        Log::file() << " " << Str(outFileName, 20) << std::endl;
                        fieldIo().writeFieldsRGrid(outFileName, wFieldsRGrid());
                    }
                    else
                    {
                        Log::file() << "  Error: Unknown command  " << command << std::endl;
                        readNext = false;
                    }
                }
            }

            template <int D>
            void System<D>::readParam()
            {
                readParam(fileMaster().paramFile());
            }

            template <int D>
            void System<D>::readParam(std::istream &in)
            {
                readBegin(in, className().c_str());
                readParameters(in);
                readEnd(in);
            }

            template <int D>
            void System<D>::readParameters(std::istream &in)
            {
                readParamComposite(in, dmixture());
                hasMixture_ = true;

                int nm = dmixture().nMonomer();
                int np = dmixture().nPolymer();

                initHomogeneous();

# if CHN == 0
                interaction().setNSegment(1);
# else
                int nSeg = 0;
                for (int p = 0 ; p < np; ++p)
                {
                    nSeg += dmixture().polymer(p).N();
                }
                interaction().setNSegment(nSeg);

                double kpN = nSeg/dmixture().kpN();
                dmixture().setKpN(kpN);
#endif                
                interaction().setNMonomer(dmixture().nMonomer());
                readParamComposite(in, interaction());
                dmixture().setInteraction(interaction());

                read(in, "unitCell", unitCell_);
                hasUnitCell_ = true;

                read(in, "mesh", mesh_);
                dmixture().setMesh(mesh(), unitCell());
                hasMesh_ = true;

                wavelist().allocate(mesh(), unitCell());
                wavelist().computeMinimumImages(mesh(), unitCell());
                dmixture().setupUnitCell(unitCell(), wavelist());

                read(in, "groupName", groupName_);
                basis().makeBasis(mesh(), unitCell(), groupName_);
                fieldIo_.associate(unitCell_, mesh_, fft_,
                                   groupName_, basis(), fileMaster_);
                dmixture().setBasis(basis());
                wavelist().computedKSq(unitCell());
                dmixture().setU0(unitCell(), wavelist());
                allocate();

                readParamComposite(in, iterator());
                iterator().allocate();
            }

            template <int D>
            void System<D>::allocate()
            {
                UTIL_CHECK(hasMixture_)
                UTIL_CHECK(hasMesh_)

                int nMonomer = dmixture().nMonomer();

                S_ = new cudaReal[nMonomer];

                wFields_.allocate(nMonomer);
                wFieldsRGrid_.allocate(nMonomer);
                wFieldsKGrid_.allocate(nMonomer);

                cFields_.allocate(nMonomer);
                cFieldsRGrid_.allocate(nMonomer);
                cFieldsKGrid_.allocate(nMonomer);

                for (int i = 0; i < nMonomer; ++i)
                {
                    wField(i).allocate(basis().nStar());
                    wFieldRGrid(i).allocate(mesh().dimensions());
                    wFieldKGrid(i).allocate(mesh().dimensions());
                    // std::cout << "KGrid mesh().dimensions() = " << mesh().dimensions() << "\n";
                    cField(i).allocate(basis().nStar());
                    cFieldRGrid(i).allocate(mesh().dimensions());
                    cFieldKGrid(i).allocate(mesh().dimensions());
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

                kSize_ = 1;

                for (int i = 0; i < D; ++i)
                {
                    kSize_ *= kMeshDimensions_[i];
                }

                // workArray.allocate(kSize_);
                workArray.allocate(mesh().size());

                cudaMalloc((void **)&d_kernelWorkSpace_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
                kernelWorkSpace_ = new cudaReal[NUMBER_OF_BLOCKS];

                isAllocated_ = true;
            }

            template <int D>
            void System<D>::computeFreeEnergy()
            {
                fHelmholtz_ = 0.0;

                int np = dmixture().nPolymer();
                int ns = basis().nStar();
                int nx = mesh().size();

                int NUMBER_OF_BLOCKS, THREADS_PER_BLOCK;
                ThreadGrid::setThreadsLogical(ns, NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (workArray.cDField(), 0.0, nx);

                int nm = dmixture().nMonomer();
                for (int i = 0; i < nm; ++i)
                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(),
                                                                                wFields_[i].cDField(),
                                                                                cFields_[i].cDField(),
                                                                                -0.5,
                                                                                ns);

                // cudaReal *d_tmp;
                // cudaMalloc(&d_tmp, sizeof(cudaReal));
                // cudaMemset(d_tmp, 0, sizeof(cudaReal));
                // sumArray<<<(basis().nStar() + 32 -1)/32, 32>>>
                // (d_tmp, workArray.cDField(), ns);
                // cudaMemcpy(&fHelmholtz_, d_tmp, sizeof(cudaReal), cudaMemcpyDeviceToHost);
                // cudaFree(d_tmp);
                fHelmholtz_ = gpuSum(workArray.cDField(), ns);

                double lnQ;

                for (int i = 0; i < np; ++i)
                {
                    lnQ = std::log(dmixture().polymer(i).Q());
                    fHelmholtz_ -= lnQ;
                }

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), 0.0, ns);

                for (int i = 0; i < nm; ++i)
                {
                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(),
                                                                                wFields_[i].cDField(),
                                                                                cFields_[i].cDField(),
                                                                                0.5,
                                                                                basis().nStar());
                    // cudaMalloc(&d_tmp, sizeof(cudaReal));
                    // cudaMemset(d_tmp, 0, sizeof(cudaReal));
                    // sumArray<<<(basis().nStar() + 32 -1)/32, 32>>>
                    // (d_tmp, workArray.cDField(), basis().nStar());
                    // cudaMemcpy(&U_, d_tmp, sizeof(cudaReal), cudaMemcpyDeviceToHost);
                    // cudaFree(d_tmp);
                    
                }
                U_ = gpuSum(workArray.cDField(), ns);

                for (int i = 0; i < np; ++i)
                {
                    dmixture().polymer(i).computeSegment(mesh(),wFieldsRGrid());
                }

                for(int i = 0; i < nm; ++i)
                {
                    assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (workArray.cDField(), 0.0, basis().nStar());
                    pointWiseAddScale2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (workArray.cDField(), 
                     wFields_[i].cDField(), 
                     cFields_[i].cDField(), 
                     -1.0,
                     basis().nStar());
                    // cudaMalloc(&d_tmp, sizeof(cudaReal));
                    // cudaMemset(d_tmp, 0, sizeof(cudaReal));
                    // sumArray<<<(basis().nStar() + 32 -1)/32, 32>>>
                    // (d_tmp, workArray.cDField(), basis().nStar());
                    // cudaMemcpy(&S_[i], d_tmp, sizeof(cudaReal), cudaMemcpyDeviceToHost);
                    // cudaFree(d_tmp);
                    S_[i] = gpuSum(workArray.cDField(), ns);
                    S_[i] -= lnQ * this->dmixture().polymer(0).bond(i).length() / this->dmixture().polymer(0).N();
                }
                
                
                UAB_ = 0.0;
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (workArray.cDField(), 0.0, nx);

                for (int i = 0; i < nm; ++i)
                {
                    for (int j = 0; j < nm; ++j)
                    {
                        assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), 0.5 * interaction().chi(i, j), nx);
                        inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), cFieldRGrid(i).cDField(), nx);
                        inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(workArray.cDField(), cFieldRGrid(j).cDField(), nx);

                        UAB_ += gpuSum(workArray.cDField(), nx) / double(nx);
                    }
                }
                UCMP_ = U_ - UAB_;

            }

            template <int D>
            void System<D>::outputThermo(std::ostream &out)
            {
                out << std::endl;
                out << " fc      = " << Dbl(fHelmholtz_, 21, 16) << std::endl;
                out << " uc      = " << Dbl(U_, 21, 16) << std::endl;
                out << "uc,AB    = " << Dbl(UAB_, 21, 16) << std::endl;
                out << "uc,CMP   = " << Dbl(UCMP_, 21, 16) << std::endl; 
                out << "-sc      = " << Dbl(fHelmholtz_ - U_, 21, 16) << std::endl;
                out << std::endl;
#if 0
            out << "uc,AB    = " << Dbl(UAB_, 21, 16) << std::endl;
            out << "uc,CMP   = " << Dbl(UCMP_, 21, 16) << std::endl;  
            out << "sc,A     = " << Dbl(-S_[0], 21, 16) << std::endl;
            out << "sc,B     = " << Dbl(-S_[1], 21, 16) << std::endl;
            out << "Q        = " << Dbl(dmixture().polymer(0).Q(), 21, 16) << std::endl;
            out << std::endl;
#endif
            }

            template <int D>
            void System<D>::outputThermo(Json::Value &thermo)
            {   
                Json::Value out;

                for (int i = 0; i < dmixture().nMonomer(); ++i)
                {
                    for (int j = i; j < dmixture().nMonomer(); ++j)
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

                for (int i = 0; i < dmixture().nMonomer(); ++i)
                {
                    
                    Json::Value tmp;
                    tmp[0] = i;
                    tmp[1] = double(dmixture().monomer(i).step());  
                    out["SegmentLength"].append(tmp); 
                 
                }

                Json::Value HelmholtzFreeEnergy;

                Json::Value InternalEnergyContribution;

                HelmholtzFreeEnergy["Total"] = Json::Value(fHelmholtz());

                InternalEnergyContribution["Total"] = Json::Value(UAB_ + UCMP_);

                Json::Value FloryHugginsRepulsion;

                FloryHugginsRepulsion["Total"] = Json::Value(UAB_);
                
                Json::Value Compressibility  = Json::Value(UCMP_);

                InternalEnergyContribution["Compressibility"] = Json::Value(Compressibility);

                InternalEnergyContribution["FloryHugginsRepulsion"] = Json::Value(FloryHugginsRepulsion);

                HelmholtzFreeEnergy["InternalEnergyContribution"] = Json::Value(InternalEnergyContribution);

                Json::Value EntropyContribution;

                double Stot = 0.0;

                for (int p = 0; p < dmixture().nPolymer(); ++p)
                {   
                    int nb = dmixture().polymer(p).nBond();
                    int s = 0;
                    for (int b = 0; b < nb; ++b)
                    {
                        if (dmixture().polymer(p).bond(b).bondtype())
                        {
                            int nbs = dmixture().polymer(p).bond(b).length();
                            for (int bs = 0; bs < nbs; ++bs)
                            {
                                Json::Value tmp;
                                tmp[0] = p;
                                tmp[1] = b;
                                tmp[2] = bs;
                                tmp[3] = dmixture().polymer(p).sS(s);
                                EntropyContribution["SegmentComponent"].append(tmp);
                                Stot += dmixture().polymer(p).sS(s);
                                ++s;
                            }
                        }
                    } 
                }

                EntropyContribution["Total"] = Json::Value(Stot);

                HelmholtzFreeEnergy["EntropyContribution"] = Json::Value(EntropyContribution);

                out["HelmholtzFreeEnergy"] = Json::Value(HelmholtzFreeEnergy);

                Json::Value tmp;
                tmp[0] = groupName();
                for (int i = 1; i <= dmixture().nParameter(); ++i)
                {
                    tmp[i] = unitCell().parameter(i-1);
                }
                out["Unitcell"].append(tmp);

                thermo.append(out);
            }


            template <int D>
            void System<D>::initHomogeneous()
            {
                // Set number of molecular species and monomers
                int nm = dmixture().nMonomer();
                int np = dmixture().nPolymer();

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
                int k;
                int nb; // number of blocks

                for (i = 0; i < np; ++i)
                {
                    // Initial array of clump sizes
                    for (j = 0; j < nm; ++j)
                    {
                        c_[j] = 0.0;
                    }

                    // Compute clump sizes for all monomer types.
                    nb = dmixture().polymer(i).nBond();
                    for (k = 0; k < nb; ++k)
                    {
                        if (dmixture().polymer(i).bond(k).bondtype())
                        {
                            j = dmixture().polymer(i).bond(k).monomerId(0);
                            c_[j] += dmixture().polymer(i).bond(k).length()/double(dmixture().polymer(i).N());
                        }
                    }
                }
            }

        }
    }
}

#endif // ! D_SYSTEM_TPP