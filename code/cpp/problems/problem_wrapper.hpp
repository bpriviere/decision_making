
#pragma once
#include <iostream>
#include <memory>
#include "problem.hpp"
#include "example1.hpp" 

// Strategy Pattern: https://stackoverflow.com/questions/41220046/is-it-possible-to-change-a-c-objects-class-after-instantiation

class Problem_Wrapper
{
    public:
        Problem* problem;
        Problem_Wrapper(std::string string, Problem_Settings problem_settings) 
        {
            problem = new Example1();
            (*problem).set_params(problem_settings);
        }
};