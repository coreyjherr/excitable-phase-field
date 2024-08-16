#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include <cstdlib>

int main() {
    Param param;
    // float simulation_time = 500;
    // param.T = (int)(simulation_time / param.dt);
    param.T = 1000000;
    param.blind = 100;
    param.xi = 3;
    param.alpha = 6;
    param.numCells = 1;

    using namespace indicators;
    BlockProgressBar bar{
        option::BarWidth{80}, option::ForegroundColor{Color::white},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        option::MaxProgress{param.T}, option::ShowElapsedTime{true},option::ShowRemainingTime{true}};

    Model model(param);
    for (int t = 0; t < param.T; t++)
    {
        if (t % param.blind == 0)
        {
            model.updateGridLocation();
            model.writeArrayToFileInBackground();
        }

        bar.tick();
        model.updateArea();
        model.updatePhi();
        model.updateA();
        model.updateR();

    }

    int call = system("/home/cherr/workspace/stochastic_phase_field_model/create_video.py");
    if (call != 0)
    {
        std::cerr << "Error creating video" << std::endl;
    }

    return 0;
}
