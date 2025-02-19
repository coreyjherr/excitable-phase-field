#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include "include/params.cuh"
#include <cstdlib>
#include <vector>
using std::vector;
#include <thread>
using std::thread;

int main(int argc, char const *argv[]) {

    if (argc > 2)
    {
        std::cout << "Usage: ./main [storage_location]" << std::endl;
        return 1;
    }
    std::string basePath = "./results/";
    if (argc == 2)
    {
        basePath = argv[1];
    }
    if (basePath.back() != '/')
        basePath += '/';

    std::cout << basePath << std::endl;

    // vector<Real> alpha = {5, 5.15, 5.3, 5.45};
    // vector<Real> adhesion = {0, 3, 6, 9, 12, 15};
    // vector<Real> adhesion = {15, 20, 25, 30, 35};
    vector<Real> xi = {5};
    vector<Real> alpha = range(5.0, 7.51, 0.1);
    vector<Real> adhesion = range(15, 30.1, 0.5);
    vector<Real> c2 = {31.5};//, 33};
    vector<Real> chem_strength = {.25};
    vector<Real> chem_force = {1};
    vector<Real> m_wall_repulsion = {30};
    // vector<Real> xi = range(4.5, 6.51, 0.1);
    // vector<Real> xi = {4.5, 5.5};
    // vector<Real> alpha = range(4.5, 13.01, 0.1);
    // vector<Real> alpha = range(5.5, 7.6, 0.5);
    // vector<Real> alpha = {5.5, 6.5, 7, 7.5};
    // vector<Real> alpha = range(6, 7.5, 0.1);
    // vector<Real> gamma = {1.5, 2};
    // vector<Real> seed = {3, 4, 5, 6, 7};
    // vector<Real> c2 = {15, 30};
    // vector<Real> packing_fraction = {0.9, .925, 0.95, .975, 1};
    // vector<Real> chem_force = {0.05, 0.01};
    // vector<Real> chem_strength = {0.5};
    // vector<Real> numCells = {1024, 512, 256, 128 };
    // vector<Real> adhesion = {0, 1, 3, 5, 7, 10};
    // vector<Real> B_s = {0.1, 1, 10};
    vector<Real> packing_fraction = {1};
    vector<Real> gamma = {2};
    // vector<vector<Real>> params = { packing_fraction,c2, adhesion, xi , gamma, alpha, };
    vector<vector<Real>> params = {alpha, adhesion};
    // vector<vector<Real>> params = {c2, sigma, alpha, adhesion};
    vector<vector<Real>> allCombinations = getAllCombinations(params);
    std::cout << "Number of combinations: " << allCombinations.size() << std::endl;
    std::cout << "Example combination: " << allCombinations[0][0] << " " << allCombinations[0][1] << " " << allCombinations[0][2] << std::endl;

    int numSimulations = allCombinations.size();
    int numThreads = 4;
    int numBlocks = numSimulations/numThreads + 1;
    std::cout << "Running " << numSimulations << " simulations" << std::endl;
    bool useCondensed = true;

    // create a vector of 4 threads, each thread will run a simulation
    for (int block_index = 0; block_index < numBlocks; block_index++)
    {
        vector<thread> threads;
        // skip the first two blocks to avoid running the same simulation
        for (int thread_index = 0; thread_index < numThreads; thread_index++)
        {
            int idx = block_index*numThreads + thread_index;
            if (idx >= numSimulations)
                break;

            Param param;
            param.r0 = 5.25;
            param.delta_r = 1.5;
            // param.r0 = 7;
            // param.delta_r = 0;
            param.blind = 25;
            param.T = 1000/0.001;
            param.m_repulsion = 30;
            // param.numCells = 512;
            param.numCells = 64;
            param.m_wall_repulsion = 30;
            // param.R_wall = sqrt(param.numCells/param.packing_fraction)*param.r0*2;

            unsigned long long seed = 0;
            param.sigma = 0.1;
            param.R_wall = 0;

            param.packing_fraction = .93;
            param.alpha = allCombinations[idx][0];
            param.m_adhesion = allCombinations[idx][1];
            // param.B_s = 0.01;

            // param.chem_force = 1.5;
            param.chem_force = 1;

            threads.push_back(thread([thread_index, param, basePath, seed, useCondensed](){
                Model model(param, thread_index, basePath, seed);
                bool progressBar = thread_index==0;
                model.simulate(progressBar, useCondensed);
            }));
        }
        // join all threads
        for (auto& t : threads)
        {
            t.join();
        }
        std::cout << "\n";

#ifdef USE_CHEMO
        std::string command = "scripts/create_video.py " + basePath + " -c -d";
#else
        std::string command = "scripts/create_video.py " + basePath + " -d";
#endif

        // if (useCondensed)
        //     command = "scripts/create_video.py " + basePath + " condensed";
        // else 
        //     command = "scripts/create_video.py " + basePath + " binary";

        int script_result = system(command.c_str());
    }

    return 0;
}
