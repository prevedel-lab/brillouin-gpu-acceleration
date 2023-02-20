/*****************************************************************//**
 * \file   example.h
 * \brief  Quick example in C++
 *
 * Allows to compile the library, and to test it directly without need of Matlab.
 *
 * \author Sebastian
 * \date   Ferbruary 2022
 *********************************************************************/

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