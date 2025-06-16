#include <cublas_v2.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "dependencies/indicators.hpp"
#include "include/error_check.cuh"
#include "include/kernels.cuh"
#include "include/model.cuh"
#include "include/model.h"
#include "include/params.hpp"

void Model::makeCondensedOutput(Real* field1,
                                Real* field2,
                                std::ofstream& outputField,
                                bool overwriteFlag) {
  int2 min_grid_location;
  min_grid_location.x = cudaTypes->grid_location[0].x;
  min_grid_location.y = cudaTypes->grid_location[0].y;
  int2 max_grid_location;
  max_grid_location.x = cudaTypes->grid_location[0].x;
  max_grid_location.y = cudaTypes->grid_location[0].y;

  for (int i = 0; i < numCells; i++) {
    min_grid_location.x = cudaTypes->grid_location[i].x < min_grid_location.x
                              ? cudaTypes->grid_location[i].x
                              : min_grid_location.x;
    min_grid_location.y = cudaTypes->grid_location[i].y < min_grid_location.y
                              ? cudaTypes->grid_location[i].y
                              : min_grid_location.y;
    max_grid_location.x = cudaTypes->grid_location[i].x > max_grid_location.x
                              ? cudaTypes->grid_location[i].x
                              : max_grid_location.x;
    max_grid_location.y = cudaTypes->grid_location[i].y > max_grid_location.y
                              ? cudaTypes->grid_location[i].y
                              : max_grid_location.y;
  }

  int condensed_size_x = max_grid_location.x - min_grid_location.x + N;
  int condensed_size_y = max_grid_location.y - min_grid_location.y + N;

  // TODO: if a cell gets too far away, we need to make sure that the condensed
  // field is not too large. So instead, we should break the condensed field
  // into smaller chunks and write them to the file. This would avoid the need
  // to allocate a large amount of memory for the condensed field. Maybe some
  // sort of sparse matrix format would be better.

  // For now, we will just check the size of the condensed field and throw an
  // error if it is too large.
  if (condensed_size_x < 0 || condensed_size_y < 0) {
    throw std::runtime_error(
        "Error: condensed_size_x or condensed_size_y is negative");
  }
  if (condensed_size_x > 10000 || condensed_size_y > 10000) {
    throw std::runtime_error(
        "Error: condensed_size_x or condensed_size_y is too large");
  }

  Real* condensed_field = new Real[condensed_size_x * condensed_size_y];
  memset(condensed_field, 0,
         condensed_size_x * condensed_size_y * sizeof(Real));

  if (field2 == nullptr) {
    for (int k = 0; k < numCells; k++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          int xcoord = cudaTypes->grid_location[k].x + i - min_grid_location.x;
          int ycoord = cudaTypes->grid_location[k].y + j - min_grid_location.y;

          // only overwrite if the value is not already set
          if (overwriteFlag) {
            if (condensed_field[ycoord * condensed_size_x + xcoord] == 0) {
              condensed_field[ycoord * condensed_size_x + xcoord] =
                  field1[k * N * N + j * N + i];
            }
          } else {
            condensed_field[ycoord * condensed_size_x + xcoord] +=
                field1[k * N * N + j * N + i];
          }
        }
      }
    }
  } else {
    for (int k = 0; k < numCells; k++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          int xcoord = cudaTypes->grid_location[k].x + i - min_grid_location.x;
          int ycoord = cudaTypes->grid_location[k].y + j - min_grid_location.y;
          condensed_field[ycoord * condensed_size_x + xcoord] +=
              field1[k * N * N + j * N + i] * field2[k * N * N + j * N + i];
        }
      }
    }
  }

  outputField.write(reinterpret_cast<char*>(&condensed_size_x), sizeof(int));
  outputField.write(reinterpret_cast<char*>(&condensed_size_y), sizeof(int));
  outputField.write(reinterpret_cast<char*>(condensed_field),
                    condensed_size_x * condensed_size_y * sizeof(Real));
  delete[] condensed_field;
}

void Model::writeDynamics(int step) {
#ifdef USE_DEVICE_BACKLOG
  int threadsPerBlock = 256;
  int numBlocks = (numCells + threadsPerBlock - 1) / threadsPerBlock;
  static int backlog_step = 0;
  copyToBacklogKernel<<<numBlocks, threadsPerBlock>>>(
      d_output, cudaTypes->d_com, cudaTypes->d_comA, d_area, d_Aarea,
      cudaTypes->d_grid_location, numCells, step, backlog_step);
  cudaErrCheck(cudaGetLastError());

  if (backlog_step == BACKLOG_SIZE - 1) {
    cudaErrCheck(
        cudaMemcpy(h_output, d_output,
                   numCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS,
                   cudaMemcpyDeviceToHost));
    dynamics_output.write(
        reinterpret_cast<char*>(h_output),
        numCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS);
    // Error check for any nan values
    for (int i = 0; i < numCells * BACKLOG_SIZE * NUM_OUTPUT_COLUMNS; ++i) {
      if (std::isnan(h_output[i])) {
        std::cerr << "Error: Dynamics output" << i % NUM_OUTPUT_COLUMNS << ", "
                  << i / NUM_OUTPUT_COLUMNS << " is NaN"
                  << "\n";
        throw std::runtime_error("Dynamics output is NaN");
      }
    }
  }
  backlog_step = (backlog_step + 1) % BACKLOG_SIZE;

#else
  cudaErrCheck(cudaMemcpy(cudaTypes->com, cudaTypes->d_com,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(cudaTypes->comA, cudaTypes->d_comA,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(area, d_area, numCells * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(Aarea, d_Aarea, numCells * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaDeviceSynchronize());
  for (int i = 0; i < numCells; ++i) {
    cudaTypes->com[i].x /= area[i];
    cudaTypes->com[i].y /= area[i];
    if (Aarea[i] < 1e-6) {
      cudaTypes->comA[i].x = 0;
      cudaTypes->comA[i].y = 0;
    } else {
      cudaTypes->comA[i].x /= Aarea[i];
      cudaTypes->comA[i].y /= Aarea[i];
    }

    if (std::isnan(cudaTypes->com[i].x) || std::isnan(cudaTypes->com[i].y)) {
      std::cerr << "Error: COM" << i << " is NaN"
                << "\n";
      std::cerr << "Area: " << area[i] << "\n";
      std::cerr << "COM: " << cudaTypes->com[i].x << ", " << cudaTypes->com[i].y
                << "\n";
      throw std::runtime_error("COM is NaN");
    }
    if (std::isnan(cudaTypes->comA[i].x) || std::isnan(cudaTypes->comA[i].y)) {
      std::cerr << "Error: COMA is NaN"
                << "\n";
      std::cerr << "Area: " << Aarea[i] << "\n";
      std::cerr << "COMA: " << cudaTypes->comA[i].x << ", "
                << cudaTypes->comA[i].y << "\n";
      throw std::runtime_error("COMA is NaN");
    }
  }

  cudaErrCheck(cudaMemcpy(cudaTypes->grid_location, cudaTypes->d_grid_location,
                          numCells * sizeof(int2), cudaMemcpyDeviceToHost));
  Real real_i, real_step, real_x, real_y;
  for (int i = 0; i < numCells; ++i) {
    real_i = (Real)i;
    real_step = (Real)step;
    real_x = (Real)cudaTypes->grid_location[i].x;
    real_y = (Real)cudaTypes->grid_location[i].y;
    dynamics_output.write(reinterpret_cast<char*>(&real_i), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&real_step), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&real_x), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&real_y), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&area[i]), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&Aarea[i]), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->com[i].x),
                          sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->com[i].y),
                          sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->comA[i].x),
                          sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->comA[i].y),
                          sizeof(Real));
  }
#endif
}

void Model::writeFullOutput() {
  // Write phi and A to file
  // copy the grid location to the host
  cudaErrCheck(cudaMemcpy(cudaTypes->grid_location, cudaTypes->d_grid_location,
                          numCells * sizeof(int2), cudaMemcpyDeviceToHost));
  std::lock_guard<std::mutex> guard(fileMutex);  // Lock the mutex
  cudaErrCheck(cudaMemcpy(phi, d_phi, numCells * N * N * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(A, d_A, numCells * N * N * sizeof(Real),
                          cudaMemcpyDeviceToHost));

  if (writeCondensedOutput_) {
    makeCondensedOutput(phi, nullptr, phi_output);
    makeCondensedOutput(A, phi, A_output);
    makeLabelImage<<<cudaTypes->grid, cudaTypes->block>>>(
        d_phi, d_buffer, cudaTypes->d_grid_location, d_Param);
    cudaErrCheck(cudaMemcpy(phi, d_buffer, numCells * N * N * sizeof(Real),
                            cudaMemcpyDeviceToHost));
    makeCondensedOutput(phi, nullptr, label_output, true);
  } else {
    phi_output.write(reinterpret_cast<char*>(phi),
                     numCells * N * N * sizeof(Real));
    A_output.write(reinterpret_cast<char*>(A), numCells * N * N * sizeof(Real));
  }

  // Write the concentration field to file
#ifdef USE_CHEMO
  cudaErrCheck(cudaMemcpy(C, d_C, p.Nchem * p.Nchem * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  C_output.write(reinterpret_cast<char*>(C), p.Nchem * p.Nchem * sizeof(Real));
#endif
}

// Function to run the file writing operation in a background thread
void Model::writeInBackground(bool useCondensedOutput) {
  std::thread backgroundThread(&Model::writeFullOutput, this);
  backgroundThread.detach();  // Detach the thread to run in the background
}
