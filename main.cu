#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include "include/params.cuh"
#include <cstdlib>
#include <thread>
#include <vector>
using std::vector;
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
    vector<Real> adhesion = {0,  9, 15, 30};
    // vector<Real> xi = range(4.5, 6.51, 0.1);
    // vector<Real> xi = {4.5, 5.5};
    // vector<Real> alpha = range(4.5, 13.01, 0.1);
    // vector<Real> alpha = range(5.5, 7.6, 0.5);
    vector<Real> alpha = {8.0};
    // vector<Real> alpha = range(6, 7.5, 0.1);
    // vector<Real> gamma = {1.5, 2};
    // vector<Real> seed = {3, 4, 5, 6, 7};
    // vector<Real> c2 = {15, 30};
    vector<Real> sigma = {0.1, 0.2, 0.3};
    vector<Real> packing_fraction = {0.9, .925, 0.95, .975, 1};
    // vector<Real> adhesion = {0, 1, 3, 5, 7, 10};
    vector<vector<Real>> params = {sigma, packing_fraction, adhesion, alpha};
    // vector<vector<Real>> params = {c2, sigma, alpha, adhesion};
    vector<vector<Real>> allCombinations = getAllCombinations(params);
    std::cout << "Number of combinations: " << allCombinations.size() << std::endl;
    std::cout << "Example combination: " << allCombinations[0][0] << " " << allCombinations[0][1] << " " << allCombinations[0][2] << std::endl;

    int numSimulations = allCombinations.size();
    int numConcurrentSimulations = numSimulations/4 + 1;
    std::cout << "Running " << numSimulations << " simulations" << std::endl;
    bool useCondensed = true;
    vector<thread> threads;

    // create a vector of 4 threads, each thread will run a simulation
    for (int block_index = 0; block_index < numConcurrentSimulations; block_index++)
    {
        // skip the first two blocks to avoid running the same simulation
        for (int thread_index = 0; thread_index < 4; thread_index++)
        {
            int idx = block_index*4 + thread_index;
            if (idx >= numSimulations)
                break;

            Param param;
            param.T = 1000/0.001;
            param.numCells = 64;
            param.m_repulsion = 30;
            // param.R_wall = param.r0*sqrt(param.numCells) + 1.0*param.r0;
            Real packing_fraction = allCombinations[idx][1];
            // Real packing_fraction = 0.844;
            param.R_wall= sqrt(param.numCells/packing_fraction)*param.r0;
            // param.R_wall = 40;
            param.m_wall_repulsion = 30;

            // param.alpha = allCombinations[idx][0];
            param.xi = 5;
            // param.gamma = allCombinations[idx][1];
            // param.m_adhesion = allCombinations[idx][1];
            param.m_adhesion = allCombinations[idx][2];
            param.alpha = allCombinations[idx][3];
            unsigned long long seed = 1;
            // param.c2 = allCombinations[idx][0];
            param.sigma = allCombinations[idx][0];

            threads.push_back(thread([thread_index, param, basePath, seed, useCondensed](){
                Model model(param, thread_index, basePath, seed);
                bool progressBar = thread_index==3;
                model.simulate(progressBar, useCondensed);
            }));
        }
        // join all threads
        for (auto& t : threads)
        {
            t.join();
        }
        std::cout << "\n";

        std::string command = "scripts/create_video.py " + basePath + " condensed delete";

        // if (useCondensed)
        //     command = "scripts/create_video.py " + basePath + " condensed";
        // else 
        //     command = "scripts/create_video.py " + basePath + " binary";

        system(command.c_str());
    }

    return 0;
}
