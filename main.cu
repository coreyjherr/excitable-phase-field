#include "model.cuh"
#include <iostream>
#include "indicators.hpp"
#include <cstdlib>

int main() {
    Param param;
    float simulation_time = 100;
    param.T = (int)(simulation_time / param.dt);
    param.blind = 100;
    param.xi = 4;
    param.alpha = 5;
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
            model.writeArrayToFileInBackground();
        }

        bar.tick();
        model.updateArea();
        model.updatePhi();
        model.updateA();
        model.updateR();

    }
    system("/home/cherr/workspace/stochastic_phase_field_model/create_video.py");
    return 0;
}
