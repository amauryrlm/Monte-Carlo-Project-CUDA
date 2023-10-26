#!/usr/bin/bash

# Build this cuda project from the root directory of a google colab notebook.
mkdir -p build
cd build
cmake ../Monte-Carlo-Project-CUDA
cmake --build ./