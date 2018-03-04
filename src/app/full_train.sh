#!/usr/bin/env bash

if [ -d "cmake-build-debug" ]; then
    echo "Starting CNN training..."
    ./cmake-build-debug/Bachelor --train $1
    echo "CNN training finished. Starting shape predictor training..."
    ./cmake-build-debug/Bachelor --train-sp $1 cmake-build-debug/TL_net.dat
    echo "Shape predictor training completed. Result is in cmake-build-debug/TL_net.dat file."
else
    echo "Starting CNN training..."
    ./build/Bachelor --train $1
    echo "CNN training finished. Starting shape predictor training..."
    ./build/Bachelor --train-sp $1 build/TL_net.dat
    echo "Shape predictor training completed. Result is in build/TL_net.dat file."
fi