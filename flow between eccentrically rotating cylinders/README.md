# Flow Between Eccentrically Rotating Cylinders - Alex Mackay 2021

Python modules for computing non Newtonian flow between rotating cylinders using the finite element method.

## Install FEniCS (Ubuntu)

Modules from this project require ```python 3.6``` and ```fenics``` as dependencies. To install FEniCS on Ubuntu, run the following commands:

```
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics
```

Alternatively if using Mac or Windows OS see installation instructions here: https://fenicsproject.org/download/.

## Test Installation with fenics_base.py

```
alex@alex-XPS-13-7390:~/python/fenics/flow between eccentrically rotating cylinders$ python3 fenics_base.py 
alex@alex-XPS-13-7390:~/python/fenics/flow between eccentrically rotating cylinders$
```

If packages have been installed correctly fenics_base.py should execute without errors.

## Flow between eccentrically rotating cylinders

### Extended White Metzner (EWM) flow simulation

```
alex@alex-XPS-13-7390:~/python/fenics/flow between eccentrically rotating cylinders$ python3 comp_ewm_jpb.py
```

Note: this program consumes parameters from `parameters-ewm.csv`. Output data are written to files stored in plots/ and results/.

### FENEP-MP flow simulation

```
alex@alex-XPS-13-7390:~/python/fenics/flow between eccentrically rotating cylinders$ python3 fenepmp_jbp.py
```

Note: this program consumes parameters from `parameters-fenep-mp.csv`. Output data are written to files stored in plots/ and results/.
