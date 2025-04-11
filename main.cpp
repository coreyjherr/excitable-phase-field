/*
MIT License

Copyright (c) 2025 Corey Herr
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
*/
#include "include/model.h"
#include <iostream>
#include "include/indicators.hpp"
#include "include/params.hpp"
#include <cstdlib>
#include <vector>
using std::vector;
#include <thread>
using std::thread;

int main(int argc, char const *argv[]) {

    if (argc > 4 || argc < 2)
    {
        std::cout << "Usage: ./run_phase_field config.toml [storage_location] [fullOutput]" << std::endl;
        return 1;
    }
    std::string configFilePath = argv[1];
    std::string basePath = "./results/";
    int writeFullOutput = 1;
    if (argc >= 3)
    {
        basePath = argv[2];
        if (basePath.back() != '/')
            basePath += '/';
    }
    if (argc >= 4) 
    {
        std::string fullOutputArg = argv[3];
        if (std::stoi(fullOutputArg) == WRITE_FULL_OUTPUT)
            writeFullOutput = WRITE_FULL_OUTPUT;
        else if (std::stoi(fullOutputArg) == WRITE_CONDENSED_OUTPUT)
            writeFullOutput = WRITE_CONDENSED_OUTPUT;
        else if (std::stoi(fullOutputArg) == WRITE_DYNAMICS_OUTPUT)
            writeFullOutput = WRITE_DYNAMICS_OUTPUT;
        else if (std::stoi(fullOutputArg) == WRITE_PARAMETERS_OUTPUT)
            writeFullOutput = WRITE_PARAMETERS_OUTPUT;
        else
            writeFullOutput = WRITE_NONE;
    }
    
    std::cout << basePath << std::endl;

    // load parameters from config file

    Param defaultParams;
    vector<vector<Real>> params = loadParamsFromTOML(configFilePath, defaultParams);
    vector<vector<Real>> allCombinations = getAllCombinations(params);

    std::cout << "Number of combinations: " << allCombinations.size() << std::endl;
    Param param0 = mapCombinationToParam(allCombinations[0]);
    printParam(param0);

    int numSimulations = allCombinations.size();
    int numThreads = 4;
    int numBlocks = numSimulations/numThreads + 1;
    std::cout << "Running " << numSimulations << " simulations" << std::endl;
    bool useCondensed = true;

    for (int block_index = 0; block_index < numBlocks; block_index++)
    {
        // create a vector of 4 threads, each thread will run a simulation
        vector<thread> threads;

        std::ostringstream oss;
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        std::string simulationID = oss.str();

        if (block_index == 0)
        {
            // copy the configuration file to the base path with the simulation ID in the string
            // this is only done once
            // should be using fsystem but this is easier
            std::string command = "cp " + configFilePath + " " + basePath + "config_" + simulationID + ".toml";
            int result = system(command.c_str());
        }

        for (int thread_index = 0; thread_index < numThreads; thread_index++)
        {
            int idx = block_index*numThreads + thread_index;
            if (idx >= numSimulations)
                break;

            Param param = mapCombinationToParam(allCombinations[idx]);
            // param.alpha = param.xi * param.alpha; // scale alpha by xi
            unsigned long long seed = 0;

            threads.push_back(thread([thread_index, param, basePath, seed, useCondensed, simulationID, writeFullOutput](){
                Model model(param, thread_index, basePath, seed, simulationID, writeFullOutput);
                bool progressBar = thread_index==0;
                try {
                    model.simulate(progressBar);
                } catch (const std::runtime_error& e) {
                    std::cerr << "Caught exception: " << e.what() << std::endl;
                }
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
        std::string command = "scripts/create_video.py " + basePath + " -v -d --glob " + simulationID;
#endif


        int script_result = system(command.c_str());
    }

    return 0;
}
