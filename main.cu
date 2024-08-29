#include "include/model.cuh"
#include <iostream>
#include "include/indicators.hpp"
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char const *argv[]) {
    using std::vector;
    using std::thread;
    vector<Real> alpha = {3, 4, 5, 6};
    vector<Real> adhesion = {0, 1, 3, 5, 7, 10};

    if (argc != 2)
    {
        std::cout << "Usage: ./main <adhesion_index>" << std::endl;
        return 1;
    }
    int j = std::atoi(argv[1]);

    // create a vector of 4 threads, each thread will run a simulation
    vector<thread> threads;
    for (int i = 0; i < 4; i++)
    {
        threads.push_back(thread([i,j, adhesion, alpha](){
            Param param;
            param.T = 1000/0.001;
            // param.T = 10000;
            param.alpha = alpha[i];
            param.numCells = 128;
            param.m_repulsion = 30;
            param.m_adhesion = adhesion[j];

            Model model(param, i);
            bool progressBar = i==3;
            model.simulate(progressBar, true);
        }));
    }
    // join all threads
    for (auto& t : threads)
    {
        t.join();
    }

    return 0;
}
