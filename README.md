
This repository contains code for the following publication, please cite our work if you use this repo:

B. Riviere, W. Hönig, M. Anderson, S-J. Chung. "Neural Tree Expansion for Multi-Robot Planning in Non-Cooperative Environments" in IEEE Robotics and Automation Letters (RA-L) June 2021. 


## Dependencies:
Developed on Ubuntu 20.04. Python dependencies in `~/environment.yml`, can be batch installed with: 

```
conda env create -f environment.yml
```
If package install fails, try removing specific versions in `.yml` file: e.g. if 
```
ResolvePackageNotFound: 
  - libgfortran-ng==7.5.0=hdf63c60_6
```  
change `libgfortran-ng==7.5.0=hdf63c60_6` to `libgfortran-ng=7.5.0`. 

Then:
```
conda activate dm_env
```

## Compiling:
from `~/code/cpp`:
```
mkdir build
cd build
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release ..
make
```

## Script Examples:  
Run individual problems and solvers from `~\code` by modifying `code/param.py` and then:
```
python run.py
```
Run batch examples from `~\code\tests`: 
```
python regression.py
```
Run waypoint planning from `~\code\tests`: 
```
python waypoint_planning.py
```
Train neural networks by modifying parameters in `code/train.py` then, from `~\code`:
```
python train.py
```
