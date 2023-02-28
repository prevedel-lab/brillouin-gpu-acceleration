/*****************************************************************//**
 * \file   errorHandling.cuh
 * \brief  Cuda Error handling
 *
 * Inspired by Cuda samples and https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api 
 *
 * \author Sebastian Hambura
 * \date   August 2020
 *********************************************************************/
 /* Copyright (C) 2020  Sebastian Hambura
  *
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation; either version 3 of the License, or
  * (at your option) any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <https://www.gnu.org/licenses/>.
  */

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