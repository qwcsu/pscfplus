#include <pgc/System.h>
#define MAXIT 60
#define UNUSED (-1.11e30)

double sign(double a, double b)
{
    if (b >= 0.0)
        return std::fabs(a);
    else
        return -std::fabs(a);
}

template <int D>
void compute(Pscf::Pspg::Continuous::System<D> *system)
{
    // // Read parameters from default parameter file
    // system->readParam();

    // // Read command script to run system
    // system->readCommands();

    system->iterate();
}

template <int D>
double dfc(Pscf::Pspg::Continuous::System<D> *system)
{
    double fhomo = 0.0;
    int m = system->mixture().nMonomer();
    for (int i  = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            double chi = system->interaction().chi(i ,j);
            double phi1 = system->c(i);
            double phi2 = system->c(j);
            fhomo += 0.5*chi * phi1 * phi2;
        }
    }

    fhomo -= 1;

    system->iterate();

    return system->fHelmholtz()- fhomo;
}

template <int D1, int D2>
double dfc(Pscf::Pspg::Continuous::System<D1> *system1,
           Pscf::Pspg::Continuous::System<D2> *system2)
{
    system1->iterate();
    system2->iterate();
    // std::cout << "df = " << system1->fHelmholtz() - system2->fHelmholtz() << "\n";
    return system1->fHelmholtz() - system2->fHelmholtz();
}

template <int D1, int D2>
double d_dfc(double chi1,
             double chi2,
             Pscf::Pspg::Continuous::System<D1> *system1,
             Pscf::Pspg::Continuous::System<D2> *system2)
{
    double f1, f2;
    system1->interaction().setChi(0, 1, chi1);
    system2->interaction().setChi(0, 1, chi1);
    f1 = dfc(system1, system2);
    system1->interaction().setChi(0, 1, chi2);
    system2->interaction().setChi(0, 1, chi2);
    f2 = dfc(system1, system2);

    return (f2 - f1) / (chi2 - chi1);
}

template <int D>
double riddr(Pscf::Pspg::Continuous::System<D> *system,
             int monomer,
             double b,
             int monomer1, int monomer2,
             double chi1, double chi2,
             double (*dfc)(Pscf::Pspg::Continuous::System<D> *system),
             double acc,
             bool &flag)
{
    double ans, fh, fl, fm, fnew,
        chih, chil, chim, chinew,
        s;
    int monomerId;
    bool braket = false;

    flag = false;

    if (chi1 < chi2)
    {
        chil = chi1;
        chih = chi2;
    }
    else
    {
        chil = chi2;
        chih = chi1;
    }
    
    
    system->mixture().monomer(monomer).setStep(b);
    for (int i = 0; i < system->mixture().nPolymer(); ++i)
    {
        for (int j = 0; j < system->mixture().polymer(i).nBlock(); ++j)
        {
            monomerId = system->mixture().polymer(i).block(j).monomerId();  
            if (monomerId == monomer)
            {
                system->mixture().polymer(i).block(j).setKuhn(b);
            }
        }
    }

    while (!braket)
    {
        system->interaction().setChi(monomer1, monomer2, chil);
        fl = dfc(system);
        system->interaction().setChi(monomer1, monomer2, chih);
        fh = dfc(system);
        if ((fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0))
        {
            braket = true;
        }
        else
        {
            std::cout << "Root is not in the origin bracket: ["
                      << chil << " " << chih << "]" << std::endl;
            double l = abs(chih - chil); 
            if ((fl > 0.0 && fh > fl) || (fl < 0.0 && fh < fl))
            {
                chih = chil;
                chil -= 0.95*l; 
            }
            else
            {
                chil = chih;
                chih += l; 
            }
            std::cout << "Try new bracket: ["
                      << chil << " " << chih << "]" << std::endl;
        }
    }

    ans = UNUSED;
 
    for (int j = 1; j <= MAXIT; j++)
    {
        chim = 0.5 * (chil + chih);
        system->interaction().setChi(monomer1, monomer2, chim);
        fm = dfc(system);
        s = sqrt(fm * fm - fl * fh);

        chinew = chim + (chim - chil) * ((fl >= fh ? 1.0 : -1.0) * fm / s);
            
        ans = chinew;
        system->interaction().setChi(monomer1, monomer2, ans);
        fnew = dfc(system);
    
        if (sign(fm, fnew) != fm)
        {
            chil = chim;
            fl = fm;
            chih = ans;
            fh = fnew;
        }
        else if (sign(fl, fnew) != fl)
        {
            chih = ans;
            fh = fnew;
        }
        else if (sign(fh, fnew) != fh)
        {
            chil = ans;
            fl = fnew;
        }
        else
        {
            std::cout << " Ridders' method failed.\n";
            exit(1);
        }
        std::cout << "chiN(" << monomer1 << ", " << monomer2 << ") = "
                  << ans << "\n";
        std::cout << "dchi = " << fabs(chih - chil) << "\n";
        std::cout << "dfc = " << fabs(fnew) << "\n\n";

        if (s == 0.0 || fabs(chih - chil) <= acc || fnew == 0.0)
        {
            flag = true;
            return ans;
        }

    }
}

template <int D1, int D2>
double riddr(Pscf::Pspg::Continuous::System<D1> *system1,
             Pscf::Pspg::Continuous::System<D2> *system2,
             int monomer,
             double b,
             int monomer1, int monomer2,
             double chi1, double chi2,
             double (*dfc)(Pscf::Pspg::Continuous::System<D1> *system1,
                           Pscf::Pspg::Continuous::System<D2> *system2),
             double acc,
             bool &flag)
{
    double ans, fh, fl, fm, fnew,
        chih, chil, chim, chinew,
        s;
    int monomerId;
    bool braket = false;

    flag = false;

    if (chi1 < chi2)
    {
        chil = chi1;
        chih = chi2;
    }
    else
    {
        chil = chi2;
        chih = chi1;
    }

    system1->mixture().monomer(monomer).setStep(b);
    for (int i = 0; i < system1->mixture().nPolymer(); ++i)
    {
        for (int j = 0; j < system1->mixture().polymer(i).nBlock(); ++j)
        {
            monomerId = system1->mixture().polymer(i).block(j).monomerId();  
            if (monomerId == monomer)
            {
                system1->mixture().polymer(i).block(j).setKuhn(b);
            }
        }
    }
    system2->mixture().monomer(monomer).setStep(b);
    for (int i = 0; i < system2->mixture().nPolymer(); ++i)
    {
        for (int j = 0; j < system2->mixture().polymer(i).nBlock(); ++j)
        {
            monomerId = system2->mixture().polymer(i).block(j).monomerId();  
            if (monomerId == monomer)
            {
                system2->mixture().polymer(i).block(j).setKuhn(b);
            }
        }
    }

    while (!braket)
    {
        system1->interaction().setChi(monomer1, monomer2, chil);
        system2->interaction().setChi(monomer1, monomer2, chil);
        fl = dfc(system1, system2);
        system1->interaction().setChi(monomer1, monomer2, chih);
        system2->interaction().setChi(monomer1, monomer2, chih);
        fh = dfc(system1, system2);
        if ((fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0))
        {
            braket = true;
        }
        else
        {
            std::cout << "Root is not in the origin bracket: ["
                      << chil << " " << chih << "]" << std::endl;
            double l = abs(chih - chil); 
            if ((fl > 0.0 && fh > fl) || (fl < 0.0 && fh < fl))
            {
                chih = chil;
                chil -= 1.05*l; 
            }
            else
            {
                chil = chih;
                chih += 1.05*l; 
            }
            std::cout << "Try new bracket: ["
                      << chil << " " << chih << "]" << std::endl;
        }
    }
    
    ans = UNUSED;
    for (int j = 1; j <= MAXIT; j++)
    {
        chim = 0.5 * (chil + chih);
        system1->interaction().setChi(monomer1, monomer2, chim);
        system2->interaction().setChi(monomer1, monomer2, chim);
        fm = dfc(system1, system2);
        s = sqrt(fm * fm - fl * fh);
        chinew = chim + (chim - chil) * ((fl >= fh ? 1.0 : -1.0) * fm / s);
            
        ans = chinew;
        system1->interaction().setChi(monomer1, monomer2, ans);
        system2->interaction().setChi(monomer1, monomer2, ans);
        fnew = dfc(system1, system2);
    
        if (sign(fm, fnew) != fm)
        {
            chil = chim;
            fl = fm;
            chih = ans;
            fh = fnew;
        }
        else if (sign(fl, fnew) != fl)
        {
            chih = ans;
            fh = fnew;
        }
        else if (sign(fh, fnew) != fh)
        {
            chil = ans;
            fl = fnew;
        }
        else
        {
            std::cout << " Ridders' method failed.\n";
            exit(1);
        }
        std::cout << "chiN(" << monomer1 << ", " << monomer2 << "): ["
                  << chil << ", " << chih << "]" << "\n";
        std::cout << "dchi = " << fabs(chih - chil) << "\n";
        std::cout << "dfc = " << fabs(fnew) << "\n\n";
        if (s == 0.0 || fabs(chih - chil) <= acc || fnew == 0.0)
        {
            flag = true;
            return ans;
        }
    }
}
/*
// template <int D1, int D2>
// double newton(double chi0,
//               Pscf::Pspg::Continuous::System<D1> *system1,
//               Pscf::Pspg::Continuous::System<D2> *system2,
//               double acc)
// {
//     double chi = chi0,
//            df, fp,
//            delta = 0.1;
//     int iter = 0;

//     while (iter < 100)
//     {
//         system1->interaction().setChi(0, 1, chi);
//         system2->interaction().setChi(0, 1, chi);
//         df = dfc(system1, system2);
//         fp = d_dfc(chi, chi + delta, system1, system2);
//         delta = -df / fp;
//         chi += delta;
//         if (abs(delta) < acc)
//             break;
//         iter++;
//     }
//     return chi;
// }*/

template <int D1, int D2>
void computeTwoPhases(Pscf::Pspg::Continuous::System<D1> *system1,
                      Pscf::Pspg::Continuous::System<D2> *system2,
                      bool echo)
{
    // Process command line options
    system1->setOptionsOutside("param1", "command", echo, 1);
    system2->setOptionsOutside("param2", "command", echo, 2);
    
    system1->readParam();
    system2->readParam();

    system1->readCommandsJson(1);
    system2->readCommandsJson(2);

    std::ifstream procFile ("command", std::ifstream::binary);
    Json::Value proc;
    procFile >> proc;

    std::ofstream Out;
    Out.open("out/PhaseBoundaryPoints");

    for (int i = 1; i < proc.size(); ++i)
    {
        if (!proc[i]["PhaseBoundaryPoints"].empty())
        {
            int m, m1, m2;
            double eps, b, c1, c2;
            bool flag;

            eps = proc[i]["PhaseBoundaryPoints"]["epsilon"].asDouble();
            m = proc[i]["PhaseBoundaryPoints"]["b"][0].asInt();
            b = proc[i]["PhaseBoundaryPoints"]["b"][1].asDouble();
            m1 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][0].asInt();
            m2 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][1].asInt();
            c1 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][2].asDouble();
            c2 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][3].asDouble();

            double chiN = riddr(system1, system2, m, b, m1, m2, c1, c2, dfc, eps, flag);
            // double chiN = newton(2.2e+01, &system1, &system2, 0.01);
            std::cout << "The phase boundry point is ("
                      << chiN
                      << ", "
                      << system1->mixture().monomer(m).step() << ")"
                      << std::endl;
            Out << std::setw(25) << "chiN";
            Out << std::setw(25) << "kuhn" << std::endl;
            Out << std::setw(25) << std::scientific << std::setprecision(16) << chiN;
            Out << std::setw(25) << std::scientific << std::setprecision(16) << system1->mixture().monomer(m).step() << std::endl;
            if (!proc[i]["PhaseBoundaryPoints"]["ACAP"].empty())
            {

                double end,
                       inc,
                       smallestStep,
                       largestStep,
                       scale,
                       current;
                bool isFinished = false;
                // Is chi increasing? Set true initially
                bool dir = true;

                current = b;
                
                end = proc[i]["PhaseBoundaryPoints"]["ACAP"]["FinalValue"].asDouble();
                inc = proc[i]["PhaseBoundaryPoints"]["ACAP"]["InitialStep"].asDouble();
                smallestStep = proc[i]["PhaseBoundaryPoints"]["ACAP"]["SmallestStep"].asDouble();
                largestStep = proc[i]["PhaseBoundaryPoints"]["ACAP"]["LargestStep"].asDouble();
                scale = proc[i]["PhaseBoundaryPoints"]["ACAP"]["StepScale"].asDouble();

                if (b == end)
                {
                    Log::file() << "The start point equals to the stop point."
                                << std::endl;
                    exit(1);
                }
                if (b > end)
                {
                    inc *= -1;
                    dir = false;
                }
                current = b + inc;
                std::cout << std::endl;
                std::cout << "Initial b(" << m << ") = " << b << std::endl;
                std::cout << "Final b(" << m <<") value: "<< end << "\n";

                while (!isFinished)
                {
                    if ((dir && current >= end) || ((!dir) && current <= end))
                    {
                        current = end;
                        isFinished = true;
                    }
                    
                    std::cout << "Current b(" << m << ") = " << current << std::endl;

                    double l = 0.1*chiN;
                    c1 = chiN - l;
                    c2 = chiN + l;
                    chiN = riddr(system1, system2, m, current, m1, m2, c1, c2, dfc, eps, flag);
                    
                    if (flag)
                    {
                        std::cout << "The phase boundry point is ("
                        << chiN
                        << ", "
                        << system1->mixture().monomer(m).step() << ")"
                        << std::endl;

                        Out << std::setw(25) << std::scientific << std::setprecision(16) << chiN;
                        Out << std::setw(25) << std::scientific << std::setprecision(16) << system1->mixture().monomer(m).step() << std::endl;

                        if (abs(inc * scale) <= largestStep)
                            inc *= scale;
                        else
                        {
                            if (dir)
                                inc = largestStep;
                            else
                                inc = -largestStep;
                        }
                        current += inc; // step forward
                    }
                }
            }
        }
    }

    Out.close();
}

template <int D>
void computeTwoPhases(Pscf::Pspg::Continuous::System<D> *system, bool echo, int id)
{
    // Process command line options
    if (id == 1)
        system->setOptionsOutside("param1", "command", echo, 1);
    else
        system->setOptionsOutside("param2", "command", echo, 2);
    
    system->readParam();

    if (id == 1)
    system->readCommandsJson(1);
    else
    system->readCommandsJson(1);

    std::ifstream procFile ("command", std::ifstream::binary);
    Json::Value proc;
    procFile >> proc;

    std::ofstream Out;
    Out.open("out/PhaseBoundaryPoints");

    for (int i = 1; i < proc.size(); ++i)
    {
        if (!proc[i]["PhaseBoundaryPoints"].empty())
        {
            int m, m1, m2;
            double eps, b, c1, c2;
            bool flag;

            eps = proc[i]["PhaseBoundaryPoints"]["epsilon"].asDouble();
            m = proc[i]["PhaseBoundaryPoints"]["b"][0].asInt();
            b = proc[i]["PhaseBoundaryPoints"]["b"][1].asDouble();
            m1 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][0].asInt();
            m2 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][1].asInt();
            c1 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][2].asDouble();
            c2 = proc[i]["PhaseBoundaryPoints"]["InitialGuess(chiN)"][3].asDouble();

            double chiN = riddr(system, m, b, m1, m2, c1, c2, dfc, eps, flag);
            // double chiN = newton(2.2e+01, &system1, &system2, 0.01);
            std::cout << "The phase boundry point is ("
                      << chiN
                      << ", "
                      << system->mixture().monomer(m).step() << ")"
                      << std::endl;
            Out << std::setw(25) << "chiN";
            Out << std::setw(25) << "kuhn" << std::endl;
            Out << std::setw(25) << std::scientific << std::setprecision(16) << chiN;
            Out << std::setw(25) << std::scientific << std::setprecision(16) << system->mixture().monomer(m).step() << std::endl;

            if (!proc[i]["PhaseBoundaryPoints"]["ACAP"].empty())
            {
                double end,
                       inc,
                       smallestStep,
                       largestStep,
                       scale,
                       current;
                bool isFinished = false;
                // Is chi increasing? Set true initially
                bool dir = true;

                current = b;
                
                end = proc[i]["PhaseBoundaryPoints"]["ACAP"]["FinalValue"].asDouble();
                inc = proc[i]["PhaseBoundaryPoints"]["ACAP"]["InitialStep"].asDouble();
                smallestStep = proc[i]["PhaseBoundaryPoints"]["ACAP"]["SmallestStep"].asDouble();
                largestStep = proc[i]["PhaseBoundaryPoints"]["ACAP"]["LargestStep"].asDouble();
                scale = proc[i]["PhaseBoundaryPoints"]["ACAP"]["StepScale"].asDouble();

                if (b == end)
                {
                    Log::file() << "The start point equals to the stop point."
                                << std::endl;
                    exit(1);
                }
                if (b > end)
                {
                    inc *= -1;
                    dir = false;
                }
                current = b + inc;
                std::cout << std::endl;
                std::cout << "Initial b(" << m << ") = " << b << std::endl;
                std::cout << "Final b(" << m <<") value: "<< end << "\n";

                while (!isFinished)
                {
                    if ((dir && current >= end) || ((!dir) && current <= end))
                    {
                        current = end;
                        isFinished = true;
                    }
                    
                    std::cout << "Current b(" << m << ") = " << current << std::endl;

                    double l = 0.001*chiN;
                    c1 = chiN - l;
                    c2 = chiN + l;
                    chiN = riddr(system, m, current, m1, m2, c1, c2, dfc, eps, flag);
                    
                    if (flag)
                    {
                        std::cout << "The phase boundry point is ("
                        << chiN
                        << ", "
                        << system->mixture().monomer(m).step() << ")"
                        << std::endl;

                        Out << std::setw(25) << std::scientific << std::setprecision(16) << chiN;
                        Out << std::setw(25) << std::scientific << std::setprecision(16) << system->mixture().monomer(m).step() << std::endl;

                        if (abs(inc * scale) <= largestStep)
                            inc *= scale;
                        else
                        {
                            if (dir)
                                inc = largestStep;
                            else
                                inc = -largestStep;
                        }
                        current += inc; // step forward
                    }
                }
            }
        }
    }

    Out.close();
}
