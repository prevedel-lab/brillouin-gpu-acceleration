/*****************************************************************//**
 * \file   errorHandling.cuh
 * \brief  Cuda Error handling
 *
 * Inspired by Cuda samples and https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api 
 *
 * \author Sebastian
 * \date   August 2020
 *********************************************************************/

#pragma once
#include <stdexcept>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
         if (abort) {
            exit(code);
        }
        else {
             throw std::exception((std::string(cudaGetErrorString(code)) + std::string(file) + std::to_string(line)).c_str());
        }
    }
}