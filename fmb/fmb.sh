#!/bin/bash

echo "Compiling fmb.cu"
nvcc fmb.cu -O3 -gencode arch=compute_89,code=sm_89 -I /usr/local/include -Xcudafe="--diag_suppress=2886" -Xcudafe="--diag_suppress=2977" -Xcudafe="--diag_suppress=20012" --expt-relaxed-constexpr -o fmb

echo "Running fmb.cu"
./fmb
