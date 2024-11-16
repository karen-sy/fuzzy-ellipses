#!/bin/bash

echo "Compiling circles.cu"
nvcc circles.cu -O3 -gencode arch=compute_89,code=sm_89 -I /usr/local/include -o circles

echo "Running circles.cu"
./circles
