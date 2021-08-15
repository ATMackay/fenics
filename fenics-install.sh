#!/bin/bash

# MIT License

# Copyright (c) 2020 Alexander Mackay

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


set -e

FENICS_INSTALL=${HOME}/fenics
FENICS_VERSION=2019.1.0
PYBIND11_VERSION=2.2.3
MODULES="hdf5-mpi/1.8.18\
 boost\
 eigen\
 python/3.6\
 scipy-stack/2017b\
 mpi4py/3.0.0\
 petsc/3.7.5\
 scotch/6.0.6\
 fftw-mpi/3.3.6\
 ipp/9.0.4"


module load $MODULES

main () {
    warning_install
    make_fenics_directory
    download_py_packages $FENICS_VERSION
    make_py_packages
    make_pybind11
    make_dolfin
    print_instructions
}

warning_install () {
    echo "---------------------------------------------------------------"
    echo "WARNING: THE FENICS/DOLFIN INSTALL WILL WIPE OUT THIS DIRECTORY"
    echo "     $FENICS_INSTALL "
    echo
    echo "IF YOU DON'T WANT THIS TO HAPPEN, PRESS CTRL-C TO ABORT"
    echo "PRESS ANY KEY TO CONTINUE"
    echo "---------------------------------------------------------------"
    read -n 1
}

print_instructions () {
    echo "---------------------------------------------------------------"
    echo "TO USE FENICS/DOLFIN, YOU NEED TO DO:"
    echo
    echo "module load $MODULES"
    echo "source $FENICS_INSTALL/bin/activate"
    echo "source $FENICS_INSTALL/share/dolfin/dolfin.conf"
    echo "---------------------------------------------------------------"
}

make_fenics_directory () {
    rm -rf $FENICS_INSTALL
    mkdir -p $FENICS_INSTALL && cd $FENICS_INSTALL
}

download_py_packages () {
    version=$1
    cd $FENICS_INSTALL
    git clone --branch=$version https://bitbucket.org/fenics-project/fiat.git
    git clone --branch=$version https://bitbucket.org/fenics-project/dijitso.git
    git clone --branch=$version https://bitbucket.org/fenics-project/ufl.git
    git clone --branch=$version https://bitbucket.org/fenics-project/ffc.git
    git clone --branch=$version https://bitbucket.org/fenics-project/dolfin.git
    git clone --branch=$version https://bitbucket.org/fenics-project/mshr.git
    git clone --branch=v$PYBIND11_VERSION \
        https://github.com/pybind/pybind11.git

    chmod u+w ~/fenics/*/.git/objects/pack/*

    mkdir -p $FENICS_INSTALL/pybind11/build
    mkdir -p $FENICS_INSTALL/dolfin/build
    mkdir -p $FENICS_INSTALL/mshr/build
}

make_pybind11 () {
    cd $FENICS_INSTALL/pybind11/build

    source $FENICS_INSTALL/bin/activate

    cmake -DPYBIND11_TEST=off \
          -DCMAKE_INSTALL_PREFIX=$HOME/fenics \
          -DPYBIND11_CPP_STANDARD=-std=c++11 ..
    nice make -j8 install
}

make_py_packages () {
    cd $FENICS_INSTALL
    virtualenv --no-download $FENICS_INSTALL
    source $FENICS_INSTALL/bin/activate
    pip3 install ply
    pip3 install numpy
    cd $FENICS_INSTALL/fiat    && pip3 install .
    cd $FENICS_INSTALL/dijitso && pip3 install .
    cd $FENICS_INSTALL/ufl     && pip3 install .
    cd $FENICS_INSTALL/ffc     && pip3 install .
}

make_dolfin () {
    cd $FENICS_INSTALL/dolfin/build

    source $FENICS_INSTALL/bin/activate

    cmake .. -DDOLFIN_SKIP_BUILD_TESTS=true \
          -DEIGEN3_INCLUDE_DIR=$EBROOTEIGEN/include \
          -DCMAKE_INSTALL_PREFIX=$HOME/fenics \
          -DCMAKE_SKIP_RPATH=ON \
          -DRT_LIBRARY=$EBROOTNIXPKGS/lib64/librt.so \
          -DHDF5_C_LIBRARY_dl=$EBROOTNIXPKGS/lib64/libdl.so \
          -DHDF5_C_LIBRARY_m=$EBROOTNIXPKGS/lib64/libm.so \
          -DHDF5_C_LIBRARY_pthread=$EBROOTNIXPKGS/lib64/libpthread.so \
          -DHDF5_C_LIBRARY_z=$EBROOTNIXPKGS/lib/libz.so \
          -DLIB_ifcore_pic=$EBROOTIFORT/lib/intel64/libifcore.so \
          -DLIB_ipgo=$EBROOTIPP/lib/intel64/libipgo.a \
          -DLIB_decimal=$EBROOTIPP/lib/intel64/libdecimal.a \
          -DLIB_irc_s=$EBROOTIPP/lib/intel64/libirc_s.a -DSCOTCH_DIR=$EBROOTSCOTCH -DSCOTCH_LIBRARIES=$EBROOTSCOTCH/lib -DSCOTCH_INCLUDE_DIRS=$EBROOTSCOTCH/include

    nice make -j 8 install
    cd $FENICS_INSTALL/dolfin/python && pip3 install .
}

exit 0

main
