# Visoelastic Flows in the Unit Square: Lid Driven Cavity Flow - Alex Mackay 2021

## TODO

Python modules for computing non viscoelastic flow in the unit square [0,1]x[0,1].

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
alex@alex-XPS-13-7390:~/python/fenics/lid-driven-cavity$ python3 fenics_fem.py 
alex@alex-XPS-13-7390:~/python/fenics/lid-driven-cavity$
```

If packages have been installed correctly dgp_base.py should execute without errors.

## Flow between eccentrically rotating cylinders

### Extended White Metzner (EWM) flow simulation TODO

```
alex@alex-XPS-13-7390:~/python/fenics/flow between eccentrically rotating cylinders$ ...
```

Note: this program consumes parameters from `parameters-ewm.csv`. Output data are written to files stored in plots/ and results/.

### FENEP-MP flow simulation TODO

```
alex@alex-XPS-13-7390:~/python/fenics/flow between eccentrically rotating cylinders$ ...
```

Note: this program consumes parameters from `parameters-fenep-mp.csv`. Output data are written to files stored in plots/ and results/.