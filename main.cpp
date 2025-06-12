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
#include <cstdlib>
#include <iostream>
#include <vector>

#include "dependencies/cxxopts.hpp"
#include "dependencies/indicators.hpp"
#include "include/model.h"
#include "include/params.hpp"
using std::vector;
#include <thread>
using std::thread;

int main(int argc, char const* argv[]) {
  // Use cxxopts for command line parsing
  cxxopts::Options options("run_phase_field",
                           "Stochastic phase field simulation");
  options.add_options()("c,config", "TOML config file",
                        cxxopts::value<std::string>())(
      "s,storage", "Storage location (default: ./results/)",
      cxxopts::value<std::string>()->default_value("./results/"))(
      "o,output",
      "Output mode (0: none, 1: full, 2: condensed, 3: dynamics, 4: "
      "parameters)",
      cxxopts::value<int>()->default_value("1"))("k,keep",
                                                 "Keep intermediate files")(
      "g,gpuids", "GPU IDs to use",
      cxxopts::value<std::vector<int>>()->default_value("0,1,2"))("h,help",
                                                                  "Print help");

  options.parse_positional({"config", "storage", "output"});
  auto result = options.parse(argc, argv);

  if (result.count("help") || argc < 2) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (!result.count("config")) {
    std::cerr << "Error: config file is required." << std::endl;
    return 1;
  }

  std::string configFilePath = result["config"].as<std::string>();
  std::string basePath = result["storage"].as<std::string>();
  int writeFullOutput = result["output"].as<int>();

  // check that the number of threads is equal to the length of the gpuids
  // vector
  std::vector<int> gpuids = result["gpuids"].as<std::vector<int>>();
  int numThreads = gpuids.size();

  if (basePath.back() != '/')
    basePath += '/';

  std::cout << basePath << std::endl;

  // load parameters from config file

  Param defaultParams;
  vector<vector<Real>> params =
      loadParamsFromTOML(configFilePath, defaultParams);
  vector<vector<Real>> allCombinations = getAllCombinations(params);

  std::cout << "Number of combinations: " << allCombinations.size()
            << std::endl;
  Param param0 = mapCombinationToParam(allCombinations[0]);
  printParam(param0);

  int numSimulations = allCombinations.size();
  int numBlocks = numSimulations / numThreads + 1;
  std::cout << "Running " << numSimulations << " simulations" << std::endl;
  bool useCondensed = true;

  for (int block_index = 0; block_index < numBlocks; block_index++) {
    // create a vector of 4 threads, each thread will run a simulation
    vector<thread> threads;

    std::ostringstream oss;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string simulationID = oss.str();

    if (block_index == 0) {
      // copy the configuration file to the base path with the simulation ID in
      // the string this is only done once should be using fsystem but this is
      // easier
      std::string command = "cp " + configFilePath + " " + basePath +
                            "config_" + simulationID + ".toml";
      int result = system(command.c_str());
    }

    for (int thread_index = 0; thread_index < numThreads; thread_index++) {
      int idx = block_index * numThreads + thread_index;
      if (idx >= numSimulations)
        break;

      Param param = mapCombinationToParam(allCombinations[idx]);
      unsigned long long seed = (unsigned long long)param.seed;
      int deviceID = gpuids[thread_index];

      threads.push_back(thread(
          [deviceID, param, basePath, seed, simulationID, writeFullOutput]() {
            Model model(param, deviceID, basePath, seed, simulationID,
                        writeFullOutput);
            std::cout << deviceID << " " << basePath << " " << seed << " "
                      << simulationID << " " << writeFullOutput << std::endl;
            bool progressBar = deviceID == 0;
            try {
              model.simulate(progressBar);
            } catch (const std::system_error& e) {
              throw e;
            } catch (const std::runtime_error& e) {
              std::cerr << "Caught exception: " << e.what() << std::endl;
            }
          }));
    }
    // join all threads
    for (auto& t : threads) {
      std::cout << "Joining thread" << t.get_id() << std::endl;
      t.join();
    }
    std::cout << "\n";

#ifdef USE_CHEMO
    std::string command = "scripts/create_video.py " + basePath + " -c -d";
#else
    std::string command =
        "scripts/create_video.py " + basePath + " -v --glob " + simulationID;
    if (!result["keep"].as<bool>())
      command += " -d";
#endif

    int script_result = system(command.c_str());
  }

  return 0;
}
