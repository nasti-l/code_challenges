#!/bin/bash
# cuda_setup.sh

# Unset any existing CUDA-related variables that might be causing conflicts
unset CUDA_VISIBLE_DEVICES

# Set proper CUDA environment
export CUDA_HOME=/usr/local/cuda-12.0
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Print environment for debugging
echo "Environment set:"
echo "CUDA_HOME=$CUDA_HOME"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
