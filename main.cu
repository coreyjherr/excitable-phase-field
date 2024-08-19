#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include <cstdlib>
#include <thread>
#include <vector>

int main() {
    using std::vector;
    using std::thread;
    vector<Real> adhesion = {5, 10, 15, 20};

    // create a vector of 4 threads, each thread will run a simulation
    vector<thread> threads;
    for (int i = 0; i < 4; i++)
    {
        threads.push_back(thread([i, adhesion](){
            Param param;
            param.T = 100/0.001;
            param.alpha = 6;
            param.numCells = 100;
            param.m_adhesion = adhesion[i];

            Model model(param, i);
            bool progressBar = true ? i == 0 : false;
            model.simulate(progressBar);
        }));
    }
    // join all threads
    for (auto& t : threads)
    {
        t.join();
    }

    // Param param;
    // param.T = 20/0.001;
    // param.alpha = 5.0;
    // param.numCells = 100;
    //
    // {
    //     Model model(param, 1);
    //     model.simulate();
    // }
    
    // int call = system("/home/cherr/workspace/stochastic_phase_field_model/create_video.py");
    // if (call != 0)
    // {
    //     std::cerr << "Error creating video" << std::endl;
    // }

    return 0;
}
