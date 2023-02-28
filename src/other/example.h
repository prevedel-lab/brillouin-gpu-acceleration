/*****************************************************************//**
 * \file   example.h
 * \brief  Quick example in C++
 *
 * Allows to compile the library, and to test it directly without need of Matlab.
 *
 * \author Sebastian Hambura
 * \date   Ferbruary 2022
 *********************************************************************/
 /* Copyright (C) 2022 Sebastian Hambura
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


#include "../cuda/kernel.cuh"
#include "../gpufit_wrapper/GPUFit_calls.h"
#include "../gpufit_wrapper/FunctionFitting.h"
#include "CImg.h"
#include "SyntheticSignal.h"
#include "../DLL_wrapper.h"

#include <random>



void main();
void debug_splines();
void check_poly2_fit(int n_fits, int n_points);
void check_lorentzian_fit(int n_fits, int n_points);
void check_spline_fit();
void check_invNorm();