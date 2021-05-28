
#pragma once
#include <iostream>
#include <memory>
#include "problem.hpp"
#include "example1.hpp" 
#include "example2.hpp" 
#include "example3.hpp" 
#include "example4.hpp" 
#include "example5.hpp" 
#include "example6.hpp" 
#include "example8.hpp" 
#include "example9.hpp" 
#include "example10.hpp" 
#include "example12.hpp" 

// Strategy Pattern: https://stackoverflow.com/questions/41220046/is-it-possible-to-change-a-c-objects-class-after-instantiation

class Problem_Wrapper
{
    public:
        Problem* problem;
        Problem_Wrapper(std::string string, Problem_Settings problem_settings) 
        {
            if (string == "example1"){
                problem = new Example1(); 
            } else if (string == "example2"){
                problem = new Example2(); 
            } else if (string == "example3"){
                problem = new Example3(); 
            } else if (string == "example4"){
                problem = new Example4(); 
            } else if (string == "example5"){
                problem = new Example5(); 
            } else if (string == "example6"){
                problem = new Example6(); 
            } else if (string == "example8"){
                problem = new Example8(); 
            } else if (string == "example9"){
                problem = new Example9(); 
            } else if (string == "example10"){
                problem = new Example10(); 
            } else if (string == "example11"){
                problem = new Example6(); // uses the same example because only difference is obstacle arangement 
            } else if (string == "example12"){
                problem = new Example12(); 
            }
            (*problem).set_params(problem_settings);
        }
};