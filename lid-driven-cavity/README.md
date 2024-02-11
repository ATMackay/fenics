# Visoelastic Flows in the Unit Square: Lid Driven Cavity Flow - Alex Mackay 2021


Python modules for computing viscoelastic lid-driven cavity flow in the unit square [0,1]x[0,1].

## Install FEniCS (Ubuntu)

Modules from this project require ```python 3.6``` and ```fenics``` as dependencies. To install FEniCS on Ubuntu, run the following commands:

```
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics
```

Alternatively if using Mac or Windows OS see installation instructions here: https://fenicsproject.org/download/.

## Test Installation with fenics_fem.py

```
lid-driven-cavity$ python3 fenics_fem.py 
```
