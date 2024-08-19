#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include <cstdlib>

int main() {
    Param param;
    param.T = 20/0.001;
    param.alpha = 5.0;
    param.numCells = 100;

    {
        Model model(param, 1);
        model.simulate();
    }
    
    // int call = system("/home/cherr/workspace/stochastic_phase_field_model/create_video.py");
    // if (call != 0)
    // {
    //     std::cerr << "Error creating video" << std::endl;
    // }

    return 0;
}
