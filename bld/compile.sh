#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -B CHN -C CMP -N NBP -K K -T DFT [-h]"
   echo ""
   echo "\t-B CHN specifying the model of chain connectivity
                    CGC : continuous Gaussian chain (default)
                    DGC : discrete Gaussian chain
                    FJC : freely jointed chain"
   echo ""
   echo "\t-B CMP specifying the system compressibility
                    I : incompressible system (default)
                    C : compressible system"
   echo ""
   echo "\t-N NBP specifying the non-bonded potential
                    DLT : Dirac delta-function potential (default)
                    G : Gaussian potential
                    DPD : dissipative particle dynamics potential
                    SS : soft sphere potential"
   echo ""
   echo "\t-K K specifying the value K of REPS-K method (when solving MDE)
                    0 : K = 0
                    1 : K = 1 (default)
                    2 : K = 2
                    3 : K = 3
                    4 : K = 4"
   echo ""
   echo "\t-T DFT specifying the discrete transform between real and reciprocal space
                    FFT : fast Fourier transform (default)
                    FCT : fast cosine transform"
   echo ""
   echo "\t-h help"
   exit 1 # Exit script after printing help
}
CHN="CGC";
CMP="I";
NBP="DLT";
K="1";
DFT="FFT";
while getopts "B:C:N:K:T:h" opt
do
   case "$opt" in
      B ) CHN="$OPTARG" ;;
      C ) CMP="$OPTARG" ;;
      N ) NBP="$OPTARG" ;;
      K ) K="$OPTARG" ;;
      T ) DFT="$OPTARG" ;;
      h ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Chain
if [[ "$CHN" == "CGC" ]]; then
  CHNs="continuous Gaussian chain"
fi
if [[ "$CHN" == "DGC" ]]; then
  CHNs="discrete Gaussian chain"
fi
if [[ "$CHN" == "FJC" ]]; then
  CHNs="freely jointed chain"
fi

# Compressibility
if [[ "$CMP" == "I" ]]; then
  CMPs="incompressible"
fi
if [[ "$CMP" == "C" ]]; then
  CMPs="compressible"
fi

# Non-bonded potential
if [[ "$NBP" == "DLT" ]]; then
  NBPs="Dirac delta-function potential"
fi
if [[ "$NBP" == "G" ]]; then
  NBPs="Gaussian potential"
fi
if [[ "$NBP" == "DPD" ]]; then
  NBPs="DPD potential"
fi
if [[ "$NBP" == "SS" ]]; then
  NBPs="soft sphere potential"
fi

# REPS-K
if [[ "$K" == "0" ]]; then
  REPSs="REPS-0 method"
fi
if [[ "$K" == "1" ]]; then
  REPSs="REPS-1 method"
fi
if [[ "$K" == "2" ]]; then
  REPSs="REPS-2 method"
fi
if [[ "$K" == "3" ]]; then
  REPSs="REPS-3 method"
fi
if [[ "$K" == "4" ]]; then
  REPSs="REPS-4 method"
fi

# REPS-K
if [[ "$DFT" == "FFT" ]]; then
  DFTs="fast Fourier transform"
fi
if [[ "$DFT" == "FCT" ]]; then
  DFTs="fast cosine transform"
fi

# Begin script in case all parameters are correct
# echo "$CHN"
# echo "$CMP"
# echo "$NBP"
# echo "$K"
# echo "$DFT"

if [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "4" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_CGC_DD_INCOMP_4_FFT
elif [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "3" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_CGC_DD_INCOMP_3_FFT
elif [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "2" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_CGC_DD_INCOMP_2_FFT
elif [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "1" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_CGC_DD_INCOMP_1_FFT
elif [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "4" ]] && [[ "$DFT" == "FCT" ]]; then
  bash compile_CGC_DD_INCOMP_4_FCT
elif [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "3" ]] && [[ "$DFT" == "FCT" ]]; then
  bash compile_CGC_DD_INCOMP_3_FCT
elif [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "2" ]] && [[ "$DFT" == "FCT" ]]; then
  bash compile_CGC_DD_INCOMP_2_FCT
elif [[ "$CHN" == "CGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$K" == "1" ]] && [[ "$DFT" == "FCT" ]]; then
  bash compile_CGC_DD_INCOMP_1_FCT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DLT" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_DD_INCOMP_FFT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "C" ]] && [[ "$NBP" == "DLT" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_DD_COMP_FFT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "G" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_G_INCOMP_FFT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "C" ]] && [[ "$NBP" == "G" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_G_COMP_FFT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DPD" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_DPD_INCOMP_FFT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "C" ]] && [[ "$NBP" == "DPD" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_DPD_COMP_FFT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "SS" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_SS_INCOMP_FFT
elif [[ "$CHN" == "FJC" ]] && [[ "$CMP" == "C" ]] && [[ "$NBP" == "SS" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_FJC_SS_COMP_FFT
elif [[ "$CHN" == "DGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "DPD" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_DGC_DPD_INCOMP_FFT
elif [[ "$CHN" == "DGC" ]] && [[ "$CMP" == "C" ]] && [[ "$NBP" == "DPD" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_DGC_DPD_COMP_FFT
elif [[ "$CHN" == "DGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "SS" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_DGC_SS_INCOMP_FFT
elif [[ "$CHN" == "DGC" ]] && [[ "$CMP" == "C" ]] && [[ "$NBP" == "SS" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_DGC_SS_COMP_FFT
elif [[ "$CHN" == "DGC" ]] && [[ "$CMP" == "I" ]] && [[ "$NBP" == "G" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_DGC_G_INCOMP_FFT
elif [[ "$CHN" == "DGC" ]] && [[ "$CMP" == "C" ]] && [[ "$NBP" == "G" ]] && [[ "$DFT" == "FFT" ]]; then
  bash compile_DGC_G_COMP_FFT
else
  echo "PSCF+ doesn't include this model system"
  exit 2
fi

echo "System model: "
if [[ "$CHN" == "DGC" ]] || [[ "$CHN" == "FJC" ]]; then
  echo " "
  echo "--------------------------------------------------------------------------------------"
  echo " "
  echo "$CMPs melts of $CHNs with $NBPs using $DFTs."
  echo "*--------------------------------------------------------------------------------------"
  echo " "
else
  echo " "
  echo "--------------------------------------------------------------------------------------"
  echo " "
  echo "$CMPs melts of $CHNs with $NBPs using $REPSs and $DFTs."
  echo "--------------------------------------------------------------------------------------"
  echo " "
fi