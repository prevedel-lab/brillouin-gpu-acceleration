/*****************************************************************//**
 * \file   kernel.cuh
 * \brief  Various Cuda function
 *
 * This file contains the various kernels called by the library.
 * It's organised in 4 parts :
 *  - GPUImage, which is a wrapper for a 2D texture
 *  - Kernel, containing the custom Cuda kernel to process the images
 *  - Functions, containing the host and device fitting functions
 *  - LaunchKernel, a wrapper to take care of the kernel's grid and block size.
 * This allows to launch kernela from regular C++ files.
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

/* Cuda Libraries*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include <cuComplex.h>

/* C++ Standard libraries*/
#include <stdio.h>
#include <stdint.h>     //uint16_t
#include <string.h>     //memset
#include <algorithm>    //min
#include <iostream>     //output in file
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <random>

/* Custom Libraries*/
#include "errorHandling.cuh"
//#include "GPUFitting_constants.h"
#include "../DLL_struct.h"

/* GPUFit*/
#include "gpufit.h"


/* Functions and structure to handle data transfer to GPU buffer and texture */

struct GPU_Image {
    cudaArray* array;
    cudaExtent size;
    cudaChannelFormatDesc channelDesc;
    cudaTextureObject_t texture;
    cudaSurfaceObject_t surface;
};
void init_GPUImage(uint16_t* pData, cudaExtent dim, GPU_Image* pDest);
void update_GPUImage(uint16_t* pData, GPU_Image* pDest);
void deinit_GPUImage(GPU_Image* image);

namespace Kernel
{
    /* Pixel space to freqency space conversion*/

    __global__ void create_frq_lut_extrapolation(int max_n_peaks, size_t width, size_t height,
        int* peak_numbers, float* original_peak_positions, float* p_a, float* p_b, float* p_c, float* frq_lut,
        float starting_freq, float ending_freq);


    /* Region of interest */

    __global__ void create_ROI(int max_n_peaks, int* peak_numbers, float* original_peak_positions, int width,
        int height, float* p_a, float* p_b, float* p_c, int starting_order, int n_orders, int* start_ROI, int* end_ROI);


    /* Combining the different orders into on single curve */

    __global__ void combine_peaks(int n_points, float starting_freq, float step, cudaTextureObject_t average,
        size_t width, size_t height, int* peak_numbers, float* frq_lut, int* start_ROI, int* end_ROI, float* dest);
    __global__ void combine_peaks_spline(int n_points, float starting_freq, float step,
        cudaTextureObject_t average, size_t width, size_t height, int* peak_numbers, 
        float* frq_lut, int* start_ROI, int* end_ROI, float* dest, Spline_Buffers spline_buffer);


    /* Extracting from the summed curve the data for the fit of only one function*/

    __global__ void extract_fitting_data(float starting_freq, float step, float* summed_curves, size_t width, size_t height,
        int* translation_lut, int start_y, int end_y, float* data_X, float* data_Y);
    __global__ void extract_fitting_data_dynamic_recentering(float starting_freq, float step, float* summed_curves, size_t width, size_t height,
        int* translation_lut, int start_y, int end_y, float* data_X, float* data_Y, float* rayleigh_fit, int* rayleigh_sanity);
    __global__ void apply_rayleigh_X_offset(int n_fits, float* stokes_data_X, int n_stokes_points, float* rayleigh_parameters, int* rayleigh_sanity);


    /* Determining some parameters for the function fit */

    __global__ void get_initial_parameters(int n_fits, float* data_X, float* data_Y, int n_points, float* parameters, bool use_abs_value);
    __global__ void estimate_noise_signal(int width, int height, float* summed_curves, int* translation_lut,
        int start_stokes_y, int end_antistokes_y, float* output_noise_value);


    /* Unused function : for testing or depreciated
    TODO : remove and cleanup, or implement fully */

    __global__ void get_Rayleigh_peaks(int max_n_peaks, cudaTextureObject_t thresholded, size_t width, size_t height,
        int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions);
    __global__ void get_phasor(float* f, float* I, int n_points, float* shift, float* width, float* amplitude, cuFloatComplex* phasor);


    /* Different sanity checks */

    __device__ void sanity_check(float signal, float noise_level, float threshold, float* SNR,
        int fitting_state, int fitting_n_iterations, int max_n_iterations, int* sanity, int index);
    __global__ void sanity_check_antistokes(float* noise_level, float threshold, int n_fits, float* param,
        int* gof_state, int* gof_n_iterations, int max_n_iterations, int angle_distribution_n, float* angle_distribution, float* SNR, int* sanity);
    __global__ void sanity_check_stokes(float* noise_level, float threshold, int n_fits, float* param, 
        int* gof_state, int* gof_n_iterations, int max_n_iterations, int angle_distribution_n, float* angle_distribution, float* SNR, int* sanity);
    __global__ void sanity_check_lorentzian(float* noise_level, float threshold, int n_fits, float* param, 
        int* gof_state, int* gof_n_iterations, int max_n_iterations, float* SNR, int* sanity);
    __global__ void sanity_check_gaussian(float* noise_level, float threshold, int n_fits, float* param,
        int* gof_state, int* gof_n_iterations, int max_n_iterations, float* SNR, int* sanity);


    /* Other */

    __global__ void find_maxima(int n_fits, int n_points, float* data_X, float* data_Y, float* maximum_position, float* maximum_amplitude);
}

namespace Functions {
    
    /* scalar computation functions(CPU & GPU) */

    __host__ __device__ float poly2(float x, float a, float b, float c);
    __host__ __device__ float from_linear_to_original_space(float pixel_l, float a, float b, float c);
    __host__ __device__ float lorentzian(float x, float amplitude, float center, float gamma);
    __host__ __device__ float gaussian(float x, float amplitude, float center, float width);
    __host__ __device__ float anti_stokes(float x, float amplitude, float shift, float width,
        float* angle_distrib, int angle_distrib_length);
    __host__ __device__ float stokes(float x, float amplitude, float shift, float width,
        float* angle_distrib, int angle_distrib_length);


     /*batch computation functions (GPU only) */

    __global__ void batch_lorentzian(float* data_x, float* data_y, int n_fits, int n_points,
        float* amplitude, float* shift, float* width, float* offset);
    __global__ void batch_gaussian(float* data_x, float* data_y, int n_fits, int n_points,
        float* amplitude, float* shift, float* width, float* offset);
    __global__ void batch_stokes(float* data_x, float* data_y, int n_fits, int n_points,
        float* amplitude, float* shift, float* width, float* offset,
        float* angle_distrib, int angle_distrib_length);
    __global__ void batch_antistokes(float* data_x, float* data_y, int n_fits, int n_points,
        float* amplitude, float* shift, float* width, float* offset,
        float* angle_distrib, int angle_distrib_length);


    /* Spline interpolation */

    __host__ __device__ void spline_coefficients(int points, float* x, float* y, float* a, float* b, float* c, float* d, Spline_Buffers spline_buffer, int buffer_offset);
}

namespace LaunchKernel {

    /* Pixel space to freqency space conversion*/

    void create_frq_lut_extrapolation(Curve_Extraction_Context* cec, cudaExtent dim,
        int* peak_numbers, float* original_peak_positions, float* p_a, float* p_b, float* p_c, float* frq_lut);


    /* Region of interest */

    void create_ROI(Curve_Extraction_Context* cec, int* peak_numbers, float* original_peak_positions, int width,
        int height, float* p_a, float* p_b, float* p_c, int starting_order, int n_orders, int* start_ROI, int* end_ROI);


    /* Combining the different orders into on single curve */

    void combine_peaks(Curve_Extraction_Context* cec, cudaTextureObject_t average, cudaExtent dim, 
        int* peak_numbers, float* frq_lut, int* start_ROI, int* end_ROI, float* dest);
    void combine_peaks_spline(Curve_Extraction_Context* cec, cudaTextureObject_t average, cudaExtent dim,
        int* peak_numbers, float* frq_lut, int* start_ROI, int* end_ROI, float* dest, Spline_Buffers gpu_buffers);


    /* Extracting from the summed curve the data for the fit of only one function*/

    void extract_fitting_data(Curve_Extraction_Context* cec, float* summed_curves, size_t width, int* translation_lut, int start_y,
        int end_y, float* data_X, float* data_Y);
    void extract_fitting_data_dynamic_recentering(Curve_Extraction_Context* cec, float* summed_curves, size_t width, int* translation_lut,
        int start_y, int end_y, float* data_X, float* data_Y, float* rayleigh_fit, int* rayleigh_sanity);
    void apply_rayleigh_X_offset(int n_fits, float* stokes_data_X, int n_stokes_points, float* rayleigh_parameters, int* rayleigh_sanity);


    /* Determining some parameters for the function fit */

    void get_initial_parameters(int n_fits, float* data_X, float* data_Y, int n_points, float* parameters, bool use_abs_value);
    void estimate_noise_signal(Curve_Extraction_Context* cec, size_t width, float* summed_curves,
        int* translation_lut, int start_stokes_y, int end_antistokes_y, float* output_noise_value);


    /* Postfit functions : compute fitted curve, or perform a sanity check*/

    void batch_evaluation(ModelID model, float* data_x, float* data_y, int n_fits, int n_points,
        float* amplitude, float* shift, float* width, float* offset,
        float* angle_distrib, int angle_distrib_length);
    void sanity_check(float* gpu_noise_level, float threshold, int n_fits, float* param,
        int* gof_state, int* gof_n_iterations, int max_n_iterations, int* sanity, ModelID model,
        float* SNR, int angle_distribution_n, float* angle_distribution);

    /*  Unused function : for testing or depreciated
        TODO : remove and cleanup, or implement fully */

    void get_Rayleigh_peaks(Curve_Extraction_Context* cec, dim3 grid_dim, dim3 block_dim, cudaTextureObject_t thresholded, cudaExtent dim,
        int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions);
    void get_phasor(dim3 grid_dim, dim3 block_dim, float* f, float* I, int n_points, float* shift, float* width, float* amplitude,
        cuFloatComplex* phasor);


    /* Other */

    void find_maxima(int n_fits, int n_points, float* data_X, float* data_Y, float* maximum_position, float* maximum_amplitude);


}