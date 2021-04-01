
#pragma once
#include <iostream>
#include "problem.hpp"
#include "example1.hpp" 
#include <memory>

// Strategy Pattern: https://stackoverflow.com/questions/41220046/is-it-possible-to-change-a-c-objects-class-after-instantiation

class Problem_Wrapper
{
    public:
        Problem* problem;
        Problem_Wrapper(std::string string) 
        {
            problem = new Example1();
        }
};