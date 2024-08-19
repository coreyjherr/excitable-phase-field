#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include <cstdlib>

int main() {
    Param param;
    float simulation_time = 50;
    param.T = (int)(simulation_time / param.dt);
    // param.T = 1000000;
    param.blind = 400;
    param.xi = 3;
    // param.alpha = 3;
    param.alpha = 4.7;
    param.numCells = 128;
    param.initial_offset_x = 60;
    param.initial_offset_y = 60;
    param.m_repulsion = 10;

    {
        Model model(param, 1);
        model.simulate();
    }
    
    int call = system("/home/cherr/workspace/stochastic_phase_field_model/create_video.py");
    if (call != 0)
    {
        std::cerr << "Error creating video" << std::endl;
    }

    return 0;
}
