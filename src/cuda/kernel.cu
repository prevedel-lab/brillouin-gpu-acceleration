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


#include "kernel.cuh"

/* Functions to handle data transfer to GPU buffer and texture */

void init_GPUImage(uint16_t* pData, cudaExtent dim, GPU_Image  *pDest) {
    /* Allocating memory */
    pDest->channelDesc = cudaCreateChannelDesc<uint16_t>();
    gpuErrchk(cudaMallocArray(&pDest->array, &pDest->channelDesc, dim.width, dim.height));
    pDest->size = dim;

    /* Copying data */
    gpuErrchk(cudaMemcpy2DToArray(pDest->array, 0, 0,  //destination array & offsete
        pData, dim.width * sizeof(uint16_t), dim.width*sizeof(uint16_t), dim.height, //pitch, width (in bytes), height
        cudaMemcpyHostToDevice));

    /* Create 2D surface
        Currently not used, but could be a solution if better performance is needed */
    

    /* Create 2D texture */
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray; //the texture draws from a cudaArray
    texRes.res.array.array = pDest->array;

    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false; //coord are not [0;1]
    texDescr.filterMode = cudaFilterModePoint; //no Interpolation between points
    texDescr.addressMode[0] = cudaAddressModeClamp; //out-of-boundary coordinates are clamped
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType; //returns as uint16_t (as opposed to normalized float)
    

    gpuErrchk(cudaCreateTextureObject(&pDest->texture, &texRes, &texDescr, NULL));


}
void update_GPUImage(uint16_t* pData, GPU_Image* pDest) {
    gpuErrchk(cudaMemcpy2DToArray(pDest->array, 0, 0,  //destination array & offset
        pData, pDest->size.width * sizeof(uint16_t), pDest->size.width * sizeof(uint16_t), pDest->size.height, //pitch, width (in bytes), height
        cudaMemcpyHostToDevice));
}
void deinit_GPUImage(GPU_Image* image) {
    gpuErrchk(cudaDestroyTextureObject(image->texture));
    //gpuErrchk(cudaDestroySurfaceObject(image->surface));
    gpuErrchk(cudaFreeArray(image->array));
}


namespace Kernel {

    /* Given the position of the Rayleigh peaks and the poly2 fit, it computes the frequency look-up-table for all the pixels. 
    It extrapolates : pixels before the first and after the last rayleigh peak also get a frq value */
    __global__ void create_frq_lut_extrapolation(int max_n_peaks, size_t width, size_t height,
        int* peak_numbers, float* original_peak_positions, float* p_a, float* p_b, float* p_c,
        float* frq_lut, float starting_freq, float ending_freq) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x < width && y < height) {

            double frq = 0;
            if (peak_numbers[x] >= 3) {
                float a = p_a[x];
                float b = p_b[x];
                float c = p_c[x];


                double step_size = (original_peak_positions[x * max_n_peaks + peak_numbers[x] - 1] - original_peak_positions[x * max_n_peaks + 0]) / (float)(peak_numbers[x] - 1);
                //float linear_pixel_space = (a * y + b) * y + c - original_peak_positions[x * max_n_peaks + 0];
                //float linear_pixel_space = (a * (y) + b) * (y) + c; 
                double linear_pixel_space = a * ((y) * (y)) + b * (y) + c;
                double first_peak_in_linear_space = a * original_peak_positions[x * max_n_peaks + 0] * original_peak_positions[x * max_n_peaks + 0]  + b * original_peak_positions[x * max_n_peaks + 0] + c;
                double distance_to_first_peak = (linear_pixel_space - first_peak_in_linear_space);
                double pos = fmod(distance_to_first_peak, step_size);

                frq = pos / step_size;                  // [0 ; 1 [ or ]-1; 0]
                if (frq > 0.5)
                    frq -= 1;                           // [-0.5 ; 0.5 ]
                if (frq < -0.5)
                    frq += 1;
                frq += 0.5;                             // [0; 1 ]
                frq *= ending_freq - starting_freq;     // [0; ending_freq - starting_freq]
                frq += starting_freq;                   // [starting_freq ; ending_freq]

#ifdef _DEBUG
                // Sanity check
                 if (abs(frq) > 7.5 ) {
                    printf("FRQLLUT_ERR @ %d %d | %f %f %f | %f %f | %f %f  | %f %f\n",
                        x, y,
                        a, b, c,
                        original_peak_positions[x * max_n_peaks + peak_numbers[x] - 1], original_peak_positions[x * max_n_peaks + 0], 
                        starting_freq, ending_freq, 
                        frq, pos);
                }
#endif
            }
            frq_lut[y * width + x] = frq;

        }
    }

    /*  Creates the region-of-interest (ROI) on the image.
    Starts at half a period before the starting's rayleigh peak position, and end half a period after the ending's one. */
    __global__ void create_ROI(int max_n_peaks, int* peak_numbers, float* original_peak_positions, int width,
        int height, float* p_a, float* p_b, float* p_c, int starting_order, int n_orders, int* start_ROI, int* end_ROI) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if (x < width) {
            if (peak_numbers[x] >= 3) {

                //In the linearized space
                float start = original_peak_positions[x * max_n_peaks + 0];
                float end = original_peak_positions[x * max_n_peaks + peak_numbers[x] - 1];
                float order_size = (original_peak_positions[x * max_n_peaks + peak_numbers[x] - 1] - original_peak_positions[x * max_n_peaks + 0]) / (float)(peak_numbers[x] - 1);

                start -= 0.5 * order_size; // Order starts at the stokes : shifting half a period

                end = start + (n_orders)*order_size;    //start is at order 0
                start += (starting_order)*order_size;   //move start to starting_order

                //Converting to original pixel space and clipping to boundary if needed
                float a = p_a[x];
                float b = p_b[x];
                float c = p_c[x];

                int start_original = ceil(Functions::from_linear_to_original_space(start, a, b, c));
                start_original = (start_original < 0) ? 0 : start_original;
                start_original = (start_original > height) ? height : start_original;
                int end_original = floor(Functions::from_linear_to_original_space(end, a, b, c));
                end_original = (end_original < 0) ? 0 : end_original;
                end_original = (end_original > height) ? height : end_original;


                start_ROI[x] = start_original;
                end_ROI[x] = end_original;

            }
            else {
                start_ROI[x] = 0;
                end_ROI[x] = 0;
            }
        }
    }

    /*  Takes a frq_lut and the image to combine all the different orders into 1 curve.
    *   The sampling between pixels is done through LINEAR interpolation 
    */
    __global__ void combine_peaks(int n_points, float starting_freq, float step, cudaTextureObject_t average, size_t width,
        size_t height, int* peak_numbers, float* frq_lut, int* start_ROI, int* end_ROI, float* dest) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int frq_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < width && frq_y < n_points) {
            float target_frq = starting_freq + frq_y * step;
            double value = 0;
            double up_val, low_val, alpha;
            int count = 0;

            //If at least 3 Rayleigh peaks (poly2 fit is meaningless else)
            if (peak_numbers[x] >= 3) {
                int start = start_ROI[x];
                int end = end_ROI[x];

                for (int y = start; y < end - 1 ; y++) {
                    if ((target_frq > frq_lut[y * width + x]) && (target_frq <= frq_lut[(y + 1) * width + x])) {
                        up_val = tex2D<uint16_t>(average, x + 0.5, y + 1 + 0.5); //TEST
                        low_val = tex2D<uint16_t>(average, x + 0.5, y + 0.5); //TEST
                        alpha = (up_val - low_val) / (frq_lut[(y + 1) * width + x] - frq_lut[y * width + x]);

                        value += alpha * (target_frq - frq_lut[y * width + x]) + low_val; //+1 here ?
                        count++;

                    }
                }

                //Avoid division by 0 and NaN problems
                if (count > 0) { 
                    value = value / count;
                }
                else {
                    value = 0;
                }
                
            }
            dest[x + frq_y * width] = value;
        }
    }

    /*  Takes a frq_lut and the image to combine all the different orders into 1 curve.
    *   The sampling between pixels is done through SPLINE interpolation.
    *   Currently less use of parallel processing here - room for optimization if performance increase is needed
    */
    __global__ void combine_peaks_spline(int n_points, float starting_freq, float step, cudaTextureObject_t average, size_t width, 
        size_t height, int* peak_numbers, float* frq_lut, int* start_ROI, int* end_ROI, float* dest, Spline_Buffers spline_buffer)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x; //which line
        //int y = threadIdx.y + blockIdx.y * blockDim.y; //

        if (x < width) { //Inside image

            for (int i = 0; i < n_points; i++)
                dest[x + i * width] = 0;

            if (peak_numbers[x] >= 3) { //We could do a polynomial fit, so frq_lut is sane
                int start = start_ROI[x];
                int end = end_ROI[x];

                /*  We proceed order by order
                     For each order, we do a spline interpolation to be able to sample as we need in the order
                     We then add all the sample points together, and divide by the number of orders used */
                int order_start = start;
                int order_end = order_start;
                int used_orders = 0;

                for (int y = start; y < end; y++) {

                    // Detected the end of the current order
                    if (frq_lut[(y + 1) * width + x] < frq_lut[y * width + x] || (y + 1 == end)) {
                        order_end = y + 1;

                        int points_in_order = order_end - order_start;
                        if (points_in_order > 2) { //Spline interpolation needs at least 3 points
                            
                            //Memory was preallocated on the GPU before the call
                            float* data_x, * data_y, * a, * b, * c, * d;
                            /**  Minimum memory requirement *
                            data_x = (float*) malloc(sizeof(float) * points_in_order);
                            data_y = (float*) malloc(sizeof(float) * points_in_order);
                            a = (float*) malloc(sizeof(float) * points_in_order);
                            b = (float*) malloc(sizeof(float) * points_in_order);
                            c = (float*) malloc(sizeof(float) * points_in_order);
                            d = (float*) malloc(sizeof(float) * points_in_order);
                            /**/

                            int position = x * height;
                            data_x = spline_buffer.data_x + position;
                            data_y = spline_buffer.data_y + position;
                            a = spline_buffer.a + position;
                            b = spline_buffer.b + position;
                            c = spline_buffer.c + position;
                            d = spline_buffer.d + position;

                            // Reorganize data for spline interpolation
                            for (int i = order_start; i < order_end; i++) {
                                data_x[i - order_start] = frq_lut[i * width + x];
                                data_y[i - order_start] = tex2D<uint16_t>(average, x + 0.5, i + 0.5);;
                            }

                            //Compute Spline coefficients
                            Functions::spline_coefficients(points_in_order, data_x, data_y, a, b, c, d, spline_buffer, position);

                            //Compute value in sample points
                            int spline = 0;
                            for (int i = 0; i < n_points; i++) {
                                float frq = starting_freq + i * step;

                                //outside interpolation
                                if (frq < data_x[0] || frq > data_x[points_in_order - 1]) {
                                    dest[x + i * width] += 0;
                                }
                                else {

                                    //jump to next part of the spline ?
                                    while (frq >= data_x[spline + 1])
                                        spline++;

                                    //Compute interpolation value
                                    float spline_x = frq - data_x[spline];
                                    dest[x + i * width] += d[spline] * spline_x * spline_x * spline_x
                                        + c[spline] * spline_x * spline_x
                                        + b[spline] * spline_x
                                        + a[spline];
                                }
                            }



                            used_orders++;
                        }

                        //Prepare for next order
                        order_start = y + 1;
                    }
                }

                //Average result
                if (used_orders > 0) {
                    for (int i = 0; i < n_points; i++)
                        dest[x + i * width] /= used_orders;

                }
            }
        }


    }

    /* Extract from the summed curve a subsection. 
    *  Used to get only (X,Y) for a specific function (Stokes, antiStokes or Rayleigh). */
    __global__ void extract_fitting_data(float starting_freq, float step, float* summed_curves, size_t width, size_t height,
            int* translation_lut, int start_y, int end_y, float* data_X, float* data_Y) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < width && y < height) {

            int target_x = translation_lut[x]; //-1 if not used
            //int n = translation_lut[width]; //last value is width  of new arrays (i.e n_fits)
            if (target_x >= 0) {
                float frq = starting_freq + y * step;
                int target_y;
                int n_points;
                if (y >= start_y && y < end_y) {
                    target_y = (y - start_y);
                    n_points = end_y - start_y;
                    
                    data_X[target_x * n_points + target_y] = frq;
                    data_Y[target_x * n_points + target_y] = summed_curves[x + y * width];
                }
                
            }
        }
 
    }

    /** Extract from the summed curve a subsection. 
     Used to get only (X,Y) for a specific function (Stokes, antiStokes or Rayleigh).
     This version also relabels the X data, by shifting it according to the rayleigh shift (only if a sane fit is detected).
     */
     __global__ void extract_fitting_data_dynamic_recentering(float starting_freq, float step, float* summed_curves, size_t width,
 size_t height, int* translation_lut, int start_y, int end_y, float* data_X, float* data_Y, float* rayleigh_fit, int* rayleigh_sanity) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < width && y < height) {

            int target_x = translation_lut[x]; //-1 if not used

            if (target_x >= 0) {

                //Dynamic recentering
                float rayleigh_center = 0;
                if (rayleigh_sanity[target_x]) //if sane fit
                    rayleigh_center = rayleigh_fit[4 * target_x + 1];
                int y_offset = (rayleigh_center < 0) ? floor((rayleigh_center) / step) : ceil((rayleigh_center) / step);  //round away from 0
                int data_y = y + y_offset;

                //Extracting the data
                /** OLD way : change the frq axis, and take the points corresponding to the range in the new x-axis *
                float frq = starting_freq + data_y * step;
                int target_y;
                int n_points;
                if ((y >= start_y) && (y < end_y)
                    && (data_y >= start_y) && (data_y < end_y)) {
                    target_y = (y - start_y);
                    n_points = end_y - start_y;
                    data_X[target_x * n_points + target_y] = frq - rayleigh_center;
                    data_Y[target_x * n_points + target_y] = summed_curves[x + data_y * width];
                }
                /**/

                /**  NEW way : take the same points as without dynamic recentering, but change their frq label*/
                float frq = starting_freq + y * step;
                int target_y;
                int n_points;
                if ((y >= start_y) && (y < end_y)) {
                    target_y = (y - start_y);
                    n_points = end_y - start_y;
                    data_X[target_x * n_points + target_y] = frq - rayleigh_center;
                    data_Y[target_x * n_points + target_y] = summed_curves[x + y * width];

                }
                /**/


            }
        }
        

    }

    /* TODO : is this function used ?*/
    __global__ void apply_rayleigh_X_offset(int n_fits, float* stokes_data_X, int n_stokes_points, float* rayleigh_parameters, int* rayleigh_sanity) {
        constexpr int rayleigh_n_parameters = 4;
        constexpr int center_parameter = 1;
        
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x < n_stokes_points && y < n_fits) {
            if(rayleigh_sanity[y]){
                stokes_data_X[x + y * n_stokes_points] -= rayleigh_parameters[y * rayleigh_n_parameters + center_parameter];
            }
            
        }
    }

    /* Used to determine the initial paramaters passed to Gpufit.
       Absolute value can be forced for the shift and the width : useful for dealing with symetrical functions*/
    __global__ void get_initial_parameters(int n_fits, float* data_X, float* data_Y, int n_points, float* parameters, bool use_abs_value) {
        int y = threadIdx.x + blockIdx.x * blockDim.x;

        if (y < n_fits) {

            parameters[y * 4 + 0] = data_Y[y * n_points + n_points / 2]; //amplitude
            parameters[y * 4 + 1] = data_X[y * n_points + n_points / 2]; //shift or center
            parameters[y * 4 + 2] = (data_X[y * n_points + 0] - data_X[y * n_points + n_points - 1]) / 3; //width
            parameters[y * 4 + 3] = (data_Y[y * n_points + 0] + data_Y[y * n_points + n_points - 1]) / 2; //offset

            if (use_abs_value) {
                parameters[y * 4 + 1] = abs(parameters[y * 4 + 1]); //shift (Antistokes shift is positiv even if the shift is in <0GHz range)
                parameters[y * 4 + 2] = abs(parameters[y * 4 + 2]); //width
            }
        }

    }

    /* Does an estimation of the SNR. Used to determine is the fit is sane or not.*/
    __global__ void estimate_noise_signal(int width, int height, float* summed_curves, int* translation_lut,
        int start_stokes_y, int end_antistokes_y, float* output_noise_value ){
        
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if ((x < width) && (translation_lut[x] >= 0)) {
            float s1 = 0;
            float s2 = 0;
            int N = 0;
            float val;
            for(int y = 0 ; y < start_stokes_y ; y++){
                val = summed_curves[x + y * width];
                if(!isnan(val)){ //Edges may not have a correct value 
                    s1 += val;
                    s2 += val * val;
                    N++;
                }
            }

            for(int y = end_antistokes_y + 1; y < height ; y++){
                val = summed_curves[x + y * width];
                if (!isnan(val)) { //Edges may not have a correct value
                    s1 += val;
                    s2 += val * val;
                    N++;
                }
            }

            output_noise_value[translation_lut[x]] = sqrt((N * s2 - s1 * s1) / (N * (N - 1)));
        }    
    }

    /* Given a signal, it gives the original and te remapped position of the rayleigh peaks.
    * It finds the peaks at roughly equidistance to the previous and the next one.
    * Function used during first developement stages - DEPRECIATED */
    __global__ void get_Rayleigh_peaks(int max_n_peaks, cudaTextureObject_t thresholded, size_t width,
        size_t height, int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions) {
        const int eps = 2;
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int peaks = 0;
        uint16_t val;
        int min, max;
        int d_left = 0;
        int d_right = 0;
        int previous_peaks[2] = { -1 };
        if (x < width) {
            for (int y = 0; y < height; y++) {
                val = tex2D<uint16_t>(thresholded, x, y);
                if (val > 0) {
                    if (previous_peaks[0] > 0 && previous_peaks[1] > 0) {
                        d_left = previous_peaks[1] - previous_peaks[0];
                        d_right = y - previous_peaks[1];
                        if (abs(d_left - d_right) <= eps) { //Is the peak in the middle equidistance to the other 2 ?
                            if (peaks == 0)
                                min = previous_peaks[1];
                            original_peak_positions[x * max_n_peaks + peaks] = previous_peaks[1];
                            peaks++;
                            max = previous_peaks[1];
                        }
                    }
                    previous_peaks[0] = previous_peaks[1];
                    previous_peaks[1] = y;

                }
            }
            peak_numbers[x] = peaks;

            int len = max - min;
            for (int i = 0; i < max_n_peaks; i++) {
                remapped_peak_positions[x * max_n_peaks + i] = (i < peaks) ? (i * len / (float)(peaks - 1) + min) : 0;
            }
        }
    }

    /* Function to do a phasor analysis on the image. Wasn't used in the end - DEPRECIATED
    * f is dim n_points, I is dim n_points*n_fits
    * f is the x - axis and I is the y - axis
    * the spectrum must contain a single peak, so must be cut before passing it to the function
    * TODO : implement it fully*/
    __global__ void get_phasor(float* f, float* I, int n_points, float* shift, float* width, float* amplitude, cuFloatComplex* phasor) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int const order = 1;
        cuFloatComplex phase = make_cuComplex(0, 0);
        cuFloatComplex local_phasor = make_cuComplex(0, 0);
        float exposant = 0;
        float sum_I = 0;
        float max_I = I[x * n_points];
        for (int i = 0; i < n_points; i++) {
            exposant = 2 * order * 3.1415f * (f[i] - f[0]) / abs(f[n_points - 1] - f[0]);
            sincosf(exposant, &phase.y, &phase.x);
            local_phasor = cuCaddf(local_phasor, cuCmulf(phase, make_cuComplex(I[x * n_points + i], 0)));
            sum_I += I[x * n_points + i];
            max_I = (I[x * n_points + i] > max_I) ? I[x * n_points + i] : max_I;
        }
        local_phasor = cuCdivf(local_phasor, make_cuComplex(sum_I, 0));
        float angle = atan2f(local_phasor.y, local_phasor.x);
        angle = (angle < 0) ? angle + 2 * 3.1415f : angle;
        float local_shift = f[0] + (f[n_points - 1] - f[0]) * angle / (2 * 3.1415f);
        float local_width = cuCabsf(local_phasor);

        //Export results;
        shift[x] = local_shift;
        width[x] = local_width;
        amplitude[x] = max_I;
        phasor[x] = local_phasor;

    }

    /* Performs different checks to know if the fit was sane or not.*/
    __device__ void sanity_check(float signal, float noise_level, float threshold, float* SNR, 
        int fitting_state, int fitting_n_iterations, int max_n_iterations, int* sanity, int index) {
        bool sane = true;

        //SNR sanity check
        float snr = signal / noise_level;
        SNR[index] = snr;
        sane = sane && (snr > threshold); //The signal is above the noise level
        
        //GoF sanity check
        sane = sane && (fitting_state == 0); // 0 : The fit converged, tolerance is satisfied, the maximum number of iterations is not exceeded (GPUFit API)
        sane = sane && (fitting_n_iterations < max_n_iterations); //Not bloqued by the number of iterations
        
        //result
        sanity[index] = sane;
    }


    /* Performs a sanity check on the antistokes fit */
    __global__ void sanity_check_antistokes(float* noise_level, float threshold, int n_fits, float* param, 
        int* gof_state, int* gof_n_iterations, int max_n_iterations, int angle_distribution_n, 
        float* angle_distribution, float* SNR, int* sanity) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if (x < n_fits) {
            float a = param[4 * x + 0];
            float s = param[4 * x + 1];
            float width = param[4 * x + 2];
            //float offset = param[4 * x + 3];
            float offset = 0;

            //Calculate value of the peak 
            float val = Functions::anti_stokes(s, a, s, width, angle_distribution, angle_distribution_n);
            sanity_check(val, noise_level[x], threshold, SNR, gof_state[x], gof_n_iterations[x], max_n_iterations, sanity, x);
        }
    }

    /* Performs a sanity check on the stokes fit */
    __global__ void sanity_check_stokes(float* noise_level, float threshold, int n_fits,
        float* param, int* gof_state, int* gof_n_iterations, int max_n_iterations, 
        int angle_distribution_n, float* angle_distribution, float* SNR, int* sanity) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if (x < n_fits) {
            float a = param[4 * x + 0];
            float s = param[4 * x + 1];
            float width = param[4 * x + 2];
            //float offset = param[4 * x + 3];
            float offset = 0;

            //Calculate value of the peak 
            float val = Functions::stokes(-s, a, s, width, angle_distribution, angle_distribution_n);
            sanity_check(val, noise_level[x], threshold, SNR, gof_state[x], gof_n_iterations[x], max_n_iterations, sanity, x);
        }
    }

    /* Performs a sanity check on the Lorentzian fit */
    __global__ void sanity_check_lorentzian(float* noise_level, float threshold, int n_fits, float* param, 
        int* gof_state, int* gof_n_iterations, int max_n_iterations, float* SNR, int* sanity) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if (x < n_fits) {
            float a = param[4 * x + 0];
            float x0 = param[4 * x + 1];
            float width = param[4 * x + 2];
            //float offset = param[4 * x + 3];
            float offset = 0;

            //Calculate value of the peak 
            float val = Functions::lorentzian(x0, a, x0, width); 
            sanity_check(val, noise_level[x], threshold, SNR, gof_state[x], gof_n_iterations[x], max_n_iterations, sanity, x);
        }
    }

    /* Performs a sanity check on the Gaussian fit */
    __global__ void sanity_check_gaussian(float* noise_level, float threshold, int n_fits, float* param, 
        int* gof_state, int* gof_n_iterations, int max_n_iterations, float* SNR, int* sanity) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if (x < n_fits) {
            float a = param[4 * x + 0];
            float x0 = param[4 * x + 1];
            float width = param[4 * x + 2];
            //float offset = param[4 * x + 3];
            float offset = 0;

            //Calculate value of the peak 
            float val = Functions::gaussian(x0, a, x0, width); 
            sanity_check(val, noise_level[x], threshold, SNR, gof_state[x], gof_n_iterations[x], max_n_iterations, sanity, x);
        }
    }

    /* Finds a Y maxima in (X,Y) data. Return the maximum and its position.
    Not optimized : Performance gain could probably be done here. */
    __global__ void find_maxima(int n_fits, int n_points, float* data_X, float* data_Y, float* maximum_position, float* maximum_amplitude){
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if(x < n_fits){
            float max = data_Y[x* n_points + 0];
            int pos = 0;
            for(int y = 1 ; y < n_points; y++){
                if (data_Y[x * n_points + y] > max){
                    max = data_Y[x * n_points + y];
                    pos = y;
                }
                    
            }

            //return result
            maximum_amplitude[x] = max;
            maximum_position[x] = data_X[x* n_points + pos];
        }
    }

}

/* Common functions used by the CPU and the GPU */
namespace Functions {
    __host__ __device__ float poly2(float x, float a, float b, float c){
        return ((a * x) + b) * x + c;
    }

    /* Inverse transformation of poly2 : linear -> image space*/
    __host__ __device__ float from_linear_to_original_space(float pixel_l, float a, float b, float c) {
        float delta = b * b - 4 * a * (c - pixel_l);
        float pixel_o2 = (-b + sqrt(delta)) / (2 * a); //take biggest root
        return (pixel_o2);
    }

    __host__ __device__ float lorentzian(float x, float amplitude, float center, float gamma) {
        return (amplitude * gamma * gamma / (gamma * gamma + (x - center) * (x - center)) );
    }

    __host__  __device__ float gaussian(float x, float amplitude, float center, float width) {
        float const argx = (x - center) * (x - center) / (2 * width * width);
        float const ex = exp(-argx);
        return (amplitude * ex);
    }

    __host__ __device__ float anti_stokes(float x, float amplitude, float shift,
        float width, float* angle_distrib, int angle_distrib_length) {
        double psi, alpha, beta, gamma;
        double val = 0;
        for (int i = 0; i < angle_distrib_length; i++) {
            psi = angle_distrib[i] / 2.f;
            alpha = x - shift * sin(psi);
            beta = width * sin(psi) * sin(psi);
            gamma = 2 * alpha / beta;
            gamma = gamma * gamma;

            val += 1 / (1 + gamma);
            //val += alpha * alpha / gamma;
        }
        return (amplitude * val );
    }

    __host__ __device__ float stokes(float x, float amplitude, float shift,
        float width, float* angle_distrib, int angle_distrib_length) {
        float psi, alpha, beta, gamma;
        float val = 0;
        for (int i = 0; i < angle_distrib_length; i++) {
            psi = angle_distrib[i] / 2.f;
            alpha = x + shift * sin(psi);
            beta = width * sin(psi) * sin(psi);
            gamma = 2 * alpha / beta;
            gamma = gamma * gamma;

            val += 1 / (1 + gamma);
        }
        return (amplitude * val );
    }


    __global__ void batch_lorentzian(float* data_x, float* data_y, int n_fits, int n_points, 
        float* amplitude, float* shift, float* width, float* offset) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x < n_points && y < n_fits) {
            data_y[y * n_fits + x] = lorentzian(data_x[y * n_fits + x], amplitude[y], shift[y], width[y]) + offset[y];
        }
    }

    __global__ void batch_gaussian(float* data_x, float* data_y, int n_fits, int n_points, 
        float* amplitude, float* shift, float* width, float* offset) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x < n_points && y < n_fits) {
            data_y[y * n_fits + x] = gaussian(data_x[y * n_fits + x], amplitude[y], shift[y], width[y]) + offset[y];
        }
    }

    __global__ void batch_stokes(float* data_x, float* data_y, int n_fits,
        int n_points, float* amplitude, float* shift, float* width, float* offset,
        float* angle_distrib, int angle_distrib_length) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x < n_points && y < n_fits) {
            data_y[y * n_fits + x] = stokes(data_x[y * n_fits + x], amplitude[y], shift[y],
                width[y], angle_distrib, angle_distrib_length) + offset[y];
        }
    }

    __global__ void batch_antistokes(float* data_x, float* data_y, int n_fits,
        int n_points, float* amplitude, float* shift, float* width, float* offset, 
        float* angle_distrib, int angle_distrib_length) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x < n_points && y < n_fits) {
            data_y[y * n_fits + x] = anti_stokes(data_x[y * n_fits + x], amplitude[y], shift[y],
            width[y], angle_distrib, angle_distrib_length) + offset[y];
        }
    }

    /**
     * Numerical Analysis 9th ed - Burden, Faires (Ch. 3 Natural Cubic Spline, Pg. 149) 
     * https://faculty.ksu.edu.sa/sites/default/files/numerical_analysis_9th.pdf (p.149)
     *  Reference found thanks to this : https://gist.github.com/svdamani/1015c5c4b673c3297309
     * 
     * \param points (= n + 1)
     * \param x x[0] to x[points]
     * \param y y[0] to y[points]
     * \param a size : n + 1
     * \param b size : n
     * \param c size : n + 1
     * \param d size : n 
     * \return 
     */
    __host__ __device__ void spline_coefficients(int points, float* x, float* y, float* a, float* b, float* c, float* d, Spline_Buffers spline_buffer, int buffer_offset)
    {
        // Using same names as in the reference
        int n = points - 1;
        float* alpha = spline_buffer.A + buffer_offset;
        float* h = spline_buffer.h + buffer_offset;
        float* l = spline_buffer.l + buffer_offset;
        float* mu = spline_buffer.u + buffer_offset;
        float* z = spline_buffer.z + buffer_offset;


        for (int i = 0; i < n + 1; i++)
            a[i] = y[i];

        for (int i = 0; i < n; i++) {
            h[i] = x[i + 1] - x[i];
        }

        for (int i = 1; i < n ; i++) {
            alpha[i] = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1]);
        }

        // Solving tridiagonal linear system
        l[0] = 1; mu[0] = 0; z[0] = 0;

        for (int i = 1; i < n; i++) {
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n] = 1; z[n] = 0; c[n] = 0;

        for (int i = n - 1; i >= 0; i--) {
            c[i] = z[i] - mu[i] * c[i + 1];
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3;
            d[i] = (c[i + 1] - c[i]) / (3 * h[i]);
        }
    }
}

/* Functions to call from outside a Cuda code. Uses a default size of grid and block : if more performance is needed, this could be a start. */
namespace LaunchKernel {
   
    void create_frq_lut_extrapolation(Curve_Extraction_Context* cec, cudaExtent dim,
        int* peak_numbers, float* original_peak_positions, float* p_a, float* p_b, float* p_c, float* frq_lut) {        
        dim3 block_dim(16, 16);
        dim3 grid_dim((dim.width + block_dim.x - 1) / block_dim.x, (dim.height + block_dim.y - 1) / block_dim.y);
        Kernel::create_frq_lut_extrapolation << <grid_dim, block_dim >> > (cec->max_n_peaks, dim.width, dim.height, peak_numbers,
            original_peak_positions, p_a, p_b, p_c, frq_lut, cec->starting_freq, cec->starting_freq + cec->step * ( cec->n_points - 1)); 
       
    }

    void create_ROI(Curve_Extraction_Context* cec, int* peak_numbers, float* original_peak_positions, int width,
        int height, float* p_a, float* p_b, float* p_c, int starting_order, int n_orders, int* start_ROI, int* end_ROI) {
        dim3 block_dim(32, 1, 1);
        dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 1, 1);
        Kernel::create_ROI << <grid_dim, block_dim >> > (cec->max_n_peaks, peak_numbers, original_peak_positions, width,
            height, p_a, p_b, p_c, starting_order, n_orders, start_ROI, end_ROI);
    }

    void combine_peaks(Curve_Extraction_Context* cec, cudaTextureObject_t average, cudaExtent dim, 
        int* peak_numbers, float* frq_lut, int* start_ROI, int* end_ROI, float* dest) {
        dim3 block_dim = dim3(16, 16);
        dim3 grid_dim = dim3((dim.width + block_dim.x - 1) / block_dim.x, (cec->n_points + block_dim.y - 1) / block_dim.y);
        Kernel::combine_peaks << <grid_dim, block_dim >> > (cec->n_points, cec->starting_freq, cec->step, average, dim.width, dim.height, peak_numbers,
            frq_lut, start_ROI, end_ROI, dest);
            
    }

    void combine_peaks_spline(Curve_Extraction_Context* cec, cudaTextureObject_t average, cudaExtent dim, int* peak_numbers, float* frq_lut, 
        int* start_ROI, int* end_ROI, float* dest, Spline_Buffers gpu_buffer)
    {
        //1D kernel launch
        dim3 block_dim = dim3(32); 
        dim3 grid_dim = dim3((dim.width + block_dim.x - 1) / block_dim.x);
        Kernel::combine_peaks_spline << <grid_dim, block_dim >> > (cec->n_points, cec->starting_freq, cec->step, average, dim.width,
            dim.height, peak_numbers, frq_lut, start_ROI, end_ROI, dest, gpu_buffer);
    }
   
    void extract_fitting_data(Curve_Extraction_Context* cec, float* summed_curves, size_t width, int* translation_lut,
        int start_y, int end_y, float* data_X, float* data_Y) {
        dim3 block_dim = dim3(16, 16);
        dim3 grid_dim = dim3((width + block_dim.x - 1) / block_dim.x, (cec->n_points + block_dim.y - 1) / block_dim.y);
        Kernel::extract_fitting_data << <grid_dim, block_dim >> > (cec->starting_freq, cec->step, summed_curves, width, cec->n_points, translation_lut, start_y, end_y,
            data_X, data_Y);
    }

    void extract_fitting_data_dynamic_recentering(Curve_Extraction_Context* cec, float* summed_curves, size_t width, int* translation_lut,
        int start_y, int end_y, float* data_X, float* data_Y, float* rayleigh_fit, int* rayleigh_sanity) {
        dim3 block_dim = dim3(16, 16);
        dim3 grid_dim = dim3((width + block_dim.x - 1) / block_dim.x, (cec->n_points + block_dim.y - 1) / block_dim.y);
        Kernel::extract_fitting_data_dynamic_recentering << <grid_dim, block_dim >> > (cec->starting_freq, cec->step, summed_curves, width, cec->n_points, translation_lut, start_y, end_y,
            data_X, data_Y, rayleigh_fit, rayleigh_sanity);
    }
    
    void apply_rayleigh_X_offset(int n_fits, float* stokes_data_X, int n_stokes_points, float* rayleigh_parameters, int* rayleigh_sanity) {
        dim3 block_dim(16, 16, 1);
        dim3 grid_dim((n_stokes_points + block_dim.x - 1) / block_dim.x, (n_fits + block_dim.y - 1) / block_dim.y, 1);
        Kernel::apply_rayleigh_X_offset << <grid_dim, block_dim >> > (n_fits, stokes_data_X, n_stokes_points, rayleigh_parameters, rayleigh_sanity);
    }

    void get_initial_parameters(int n_fits, float* data_X, float* data_Y, int n_points, float* parameters, bool use_abs_value) {
        dim3 block_dim(32, 1, 1);
        dim3 grid_dim((n_fits + block_dim.x - 1) / block_dim.x, 1, 1);
        Kernel::get_initial_parameters << <grid_dim, block_dim >> > (n_fits, data_X, data_Y, n_points, parameters, use_abs_value);
    }

    void estimate_noise_signal(Curve_Extraction_Context* cec, size_t width, float* summed_curves,
        int* translation_lut, int start_stokes_y, int end_antistokes_y, float* output_noise_value){
        dim3 block_dim(32, 1, 1);
        dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 1, 1);
        Kernel::estimate_noise_signal << <grid_dim, block_dim >> > (width, cec->n_points,
            summed_curves, translation_lut, start_stokes_y, end_antistokes_y, output_noise_value);
    }
        
    void batch_evaluation(ModelID model, float* data_x, float* data_y, int n_fits,
        int n_points, float* amplitude, float* shift, float* width, float* offset,
        float* angle_distrib, int angle_distrib_length) {
        dim3 block_dim(16, 16, 1);
        dim3 grid_dim((n_points + block_dim.x - 1) / block_dim.x, (n_fits + block_dim.y - 1) / block_dim.y, 1);
        switch (model)
        {
        case GAUSS_1D:
            Functions::batch_gaussian<<<grid_dim, block_dim >>>(data_x, data_y, n_fits, n_points, amplitude, shift, width, offset);
            break;
        case ANTI_STOKES:
            Functions::batch_antistokes<<<grid_dim, block_dim >>>(data_x, data_y, n_fits, n_points, amplitude, shift, width, offset,
                angle_distrib, angle_distrib_length);
            break;
        case STOKES:
            Functions::batch_stokes<<<grid_dim, block_dim >>>(data_x, data_y, n_fits, n_points, amplitude, shift, width, offset,
                angle_distrib, angle_distrib_length);
            break;

        case CAUCHY_LORENTZ_1D:
            Functions::batch_lorentzian<<<grid_dim, block_dim >>>(data_x, data_y, n_fits, n_points, amplitude, shift, width, offset);
            break;
        default:
            break;
        }

    }

    void sanity_check(float* gpu_noise_level, float threshold, int n_fits,
        float* param, int* gof_state, int* gof_n_iterations, int max_n_iterations, int* sanity,
        ModelID model, float* SNR, int angle_distribution_n, float* angle_distribution) {

        dim3 block_dim(32, 1, 1);
        dim3 grid_dim((n_fits + block_dim.x - 1) / block_dim.x, 1, 1);
        switch (model)
        {
        case GAUSS_1D:
            Kernel::sanity_check_gaussian << <grid_dim, block_dim >> > (gpu_noise_level, threshold, n_fits, param, gof_state, gof_n_iterations, max_n_iterations, SNR, sanity);
            break;
        case ANTI_STOKES:
            Kernel::sanity_check_antistokes << <grid_dim, block_dim >> > (gpu_noise_level, threshold, n_fits, param, gof_state, gof_n_iterations, max_n_iterations, angle_distribution_n, angle_distribution, SNR, sanity);
            break;
        case STOKES:
            Kernel::sanity_check_stokes << <grid_dim, block_dim >> > (gpu_noise_level, threshold, n_fits, param, gof_state, gof_n_iterations, max_n_iterations, angle_distribution_n, angle_distribution, SNR, sanity);
            break;
        case CAUCHY_LORENTZ_1D:
            Kernel::sanity_check_lorentzian << <grid_dim, block_dim >> > (gpu_noise_level, threshold, n_fits, param, gof_state, gof_n_iterations, max_n_iterations, SNR, sanity);
            break;
        default:
            break;
        }
    }

    void get_Rayleigh_peaks(Curve_Extraction_Context* cec, dim3 grid_dim, dim3 block_dim, cudaTextureObject_t thresholded,
        cudaExtent dim, int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions) {
        Kernel::get_Rayleigh_peaks << <grid_dim, block_dim >> > (cec->max_n_peaks, thresholded, dim.width, dim.height,
            peak_numbers, original_peak_positions, remapped_peak_positions);
    }

    void get_phasor(dim3 grid_dim, dim3 block_dim, float* f, float* I, int n_points, float* shift, float* width, float* amplitude, cuFloatComplex* phasor) {
        Kernel::get_phasor << <grid_dim, block_dim >> > (f, I, n_points, shift, width, amplitude, phasor);
    }

    void find_maxima(int n_fits, int n_points, float* data_X, float* data_Y, float* maximum_position, float* maximum_amplitude) {
        dim3 block_dim(32, 1, 1);
        dim3 grid_dim((n_fits + block_dim.x - 1) / block_dim.x, 1, 1);
        Kernel::find_maxima << <grid_dim, block_dim >> > (n_fits, n_points, data_X, data_Y, maximum_position, maximum_amplitude);
    }

}


