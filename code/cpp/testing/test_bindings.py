
import numpy as np 
import sys
sys.path.append("../")
import build.bindings as bindings

if __name__ == "__main__":
    # print('example.add(1,2)',example.add(1,2))

    # state = np.array( [[1.0],[1.0]] )
    state = np.array( [[ 7.61588577],[-1.7170523]] )
    print(state.shape)

    print('bindings.search(state)',bindings.cpp_search(state))
    

# import example 

# import numpy as np
# import build.bindings as bindings

# A = np.array([[1,2,1],
#               [2,1,0],
#               [-1,1,2]])

# print('A = \n'                   , A)
# print('bindings.det(A) = \n'      , bindings  .det(A))
# print('numpy.linalg.det(A) = \n' , np.linalg.det(A))
# print('bindings.inv(A) = \n'      , bindings  .inv(A))
# print('numpy.linalg.inv(A) = \n' , np.linalg.inv(A))

