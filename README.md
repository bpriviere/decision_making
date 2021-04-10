

### Dependencies

 Python dependencies in '...todo.yml', can be batch installed with: 
 	`conda ...todo`
 or installed individually with pip, e.g.: 
 	`pip install numpy`
 cpp dependencies 
 	`todo`


### Compiling 
 from ~/code/cpp:
  `mkdir build`
  `cd build`
  `cmake -DCMAKE_BUILD_TYPE=Release ..`
  `make`


### Script Examples 
 Run individual problems and solvers from ~\code by specifying 'code/param.py':
  `python run.py`
 Batch runs from ~\code\tests : 
  `python regression.py`
 Waypoint planning from ~\code\tests : 
  `python waypoint_planning.py` 


### Training Neural Networks 
 In progress. Eventually:
  `python train.py`
