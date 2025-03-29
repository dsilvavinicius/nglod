#!/bin/bash

# Install C++/CUDA extensions
for ext in mesh2sdf_cuda sol_nglod; do
    cd $ext

    # Clean previous builds
    python setup.py clean --all
    rm -rf build/ dist/ *.egg-info

    # Build and install the extension
    pip install . --user

    cd -
done

