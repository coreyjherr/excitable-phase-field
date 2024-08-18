#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include <cstdlib>

int main() {
    Param param;
    float simulation_time = 2000;
    param.T = (int)(simulation_time / param.dt);
    // param.T = 1000000;
    param.blind = 200;
    param.xi = 3;
    // param.alpha = 3;
    param.alpha = 4.7;
    param.numCells = 128;
    param.initial_offset_x = 60;
    param.initial_offset_y = 60;
    param.m_repulsion = 10;

    using namespace indicators;
    BlockProgressBar bar{
        option::BarWidth{80}, option::ForegroundColor{Color::white},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        option::MaxProgress{param.T}, option::ShowElapsedTime{true},option::ShowRemainingTime{true}};

    { // Scope for model so that it is destructed before the video is created
    Model model(param);
    for (int t = 0; t < param.T; t++)
    {
        if (t % param.blind == 0)
        {
            model.updateGridLocation();
            model.writeInBackground();
        }

        bar.tick();
        model.updateCollisionForce();
        model.updateArea();
        model.updatePhi();
        model.updateA();
        model.updateR();

    }
    }
    
    // int call = system("/home/cherr/workspace/stochastic_phase_field_model/create_video.py");
    // if (call != 0)
    // {
    //     std::cerr << "Error creating video" << std::endl;
    // }

    return 0;
}
