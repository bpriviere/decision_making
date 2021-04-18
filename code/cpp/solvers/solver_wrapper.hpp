#pragma once
#include <iostream>
#include <memory>
#include "solver.hpp"
#include "puct_v0.hpp" 

// Strategy Pattern: https://stackoverflow.com/questions/41220046/is-it-possible-to-change-a-c-objects-class-after-instantiation

class Solver_Wrapper
{
    public:
        Solver* solver;
        Solver_Wrapper(std::string string, Solver_Settings solver_settings) 
        {
            if (string == "C_PUCT_V0"){
                solver = new PUCT_V0(); 
            } 
            (*solver).set_params(solver_settings);
        }
};