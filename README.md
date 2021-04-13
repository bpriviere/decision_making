

### Dependencies

 Python dependencies in '~/environment.yml', can be batch installed with: 
 	`conda env create -f environment.yml`
 or installed individually with pip, e.g.: 
 	`pip install numpy`
 cpp dependencies 
 	`sudo apt install -y libeigen3-dev libyaml-cpp-dev`


### Compiling 
 from ~/code/cpp:
  `mkdir build`
  `cd build`
  `cmake -DCMAKE_BUILD_TYPE=Release ..`
  `make`


### Script Examples 
 Run individual problems and solvers from ~\code by modifying 'code/param.py' and then:
  `python run.py`
 Batch runs from ~\code\tests : 
  `python regression.py`
 Waypoint planning from ~\code\tests : 
  `python waypoint_planning.py` 


### Training Neural Networks 
 In progress. Eventually:
  `python train.py`
