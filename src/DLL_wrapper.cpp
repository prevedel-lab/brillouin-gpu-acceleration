#define DLL_COMPILATION
#include "DLL_wrapper.h"

namespace DLL_wrapper {
	extern "C" {

		/**
		 *	A function to linearize the space between rayleigh peaks for each column independently.
		 *	A quadratic function (a*x^2 + b*x + c) is best fitted for each column. Because of that, the fit is only meaningfull
		 *	if there are more than *3* rayleigh peaks in the columns.
		 *
		 *
		 *
		 *	 \param[in] cec The curve extraction context, containing the information about the max number of rayleigh peaks.
		 *	 \param[in] width Width of the image (number of columns).
		 *   \param[in] peak_numbers 1D array containing for each column the number of rayleigh peaks detected.
		 *		If there are < 2, no summation will be done on this column. dim : image.width
		 *   \param[in] original_peak_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The positions are given in the pixel of the image, without linearisation.
		 *		dimension : image.width * cec.max_n_peaks
		 *   \param[in] remapped_peak_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The positions are given in the linearized pixelspace of the image,
		 *		i.e the target location of each rayleigh peak.
		 *		dimension : image.width * cec.max_n_peaks
		 *   \param[out] a 1D array containing the first coefficient for the linearization of pixel of the column. dimension : image.width
		 *   \param[out] b 1D array containing the second coefficient for the linearization of pixel of the column. dimension : image.width
		 *   \param[out] c 1D array containing the third coefficient for the linearization of pixel of the column. dimension : image.width
		 *
		 * \returns
		 *		a, b and c : the arrays must already by allocated to the correct size.
		 *
		 *
		 * \ingroup standalone
		 *
		 */
		void linearize_pixelspace(Curve_Extraction_Context* cec, int width, int* peak_numbers, float* original_peak_positions,
			float* remapped_peak_positions, float* a, float* b, float* c) {

			Poly2Fitting poly2(width, cec->max_n_peaks, 20, 1e-5);
			poly2.fit(peak_numbers, original_peak_positions, remapped_peak_positions);
			poly2.get_fitted_parameters(a, b, c);
		}

		/**
		 *	A function to compute the corresponding freqeuncy for each pixel in the image. The pixelspace needs to be linearized beforehand.
		 *	All the inputs and outputs are expected to be CPU variables, the tranfers to and from the GPU is done internally.
		 *
		 *  \param[in] cec The curve extraction context used for this run. Needs to be consistent with the previous calls.
		 *  \param[out]  frq_lut A 2D array where each value represents the frequency in linearized space of the corresponding pixel in the image.
		 *		dimension : image.width * image.height
		 *  \param[in] height Height of the image
		 *  \param[in] width Width of the image
		 *  \param[in] peak_numbers 1D array containing for each column the number of rayleigh peaks detected.
		 *		If there are <2, no summation will be done on this column. dim : Image.width
		 *  \param[in] original_peak_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The position is given in the pixel of the image, without linearisation.
		 *		dimension : image.width * cec.max_n_peaks
		 *  \param[in] a 1D array containing the first coefficient for the linearization of pixel of the column. dimension : image.width
		 *		(a*x*x + b*x + c)
		 *  \param[in] b 1D array containing the second coefficient for the linearization of pixel of the column. dimension : image.width
		 *		(a*x*x + b*x + c)
		 *  \param[in] c 1D array containing the third coefficient for the linearization of pixel of the column. dimension : image.width
		 *		(a*x*x + b*x + c)
		 *
		 * \return frq_lut : passed as a pointer. Must already be allocated before calling this function.
		 *
		 *
		 *
		 * \ingroup standalone
		 *
		 */
		void create_frq_lut(Curve_Extraction_Context* cec, float* frq_lut, int height, int width,
			int* peak_numbers, float* original_peak_positions, float* a, float* b, float* c) {

			//Allocating GPU memory
			float* gpu_frq_lut;
			size_t image_size = sizeof(float) * height * width;
			float* gpu_a, * gpu_b, * gpu_c;
			size_t array_size = sizeof(float) * width;
			int* gpu_peak_numbers;
			float* gpu_original_peak_positions;
			gpuErrchk(cudaMalloc(&gpu_frq_lut, image_size));
			gpuErrchk(cudaMalloc(&gpu_a, array_size));
			gpuErrchk(cudaMalloc(&gpu_b, array_size));
			gpuErrchk(cudaMalloc(&gpu_c, array_size));
			gpuErrchk(cudaMalloc(&gpu_peak_numbers, array_size * 1));
			gpuErrchk(cudaMalloc(&gpu_original_peak_positions, array_size * cec->max_n_peaks));

			//Sending data to GPU
			gpuErrchk(cudaMemcpy(gpu_peak_numbers, peak_numbers, array_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_original_peak_positions, original_peak_positions, array_size * cec->max_n_peaks, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_a, a, array_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_b, b, array_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_c, c, array_size, cudaMemcpyHostToDevice));

			//Calling the kernel
			cudaExtent dim{ width, height, 1 };
			LaunchKernel::create_frq_lut_extrapolation(cec, dim, gpu_peak_numbers, gpu_original_peak_positions, gpu_a, gpu_b, gpu_c, gpu_frq_lut);

			//Retrieving data from GPU
			gpuErrchk(cudaMemcpy(frq_lut, gpu_frq_lut, image_size, cudaMemcpyDeviceToHost));

			//Freeing gpu memory
			gpuErrchk(cudaFree(gpu_frq_lut));
			gpuErrchk(cudaFree(gpu_a));
			gpuErrchk(cudaFree(gpu_b));
			gpuErrchk(cudaFree(gpu_c));
			gpuErrchk(cudaFree(gpu_peak_numbers));
			gpuErrchk(cudaFree(gpu_original_peak_positions));

		}

		/**
		 *	A function to create the Region Of Interest which is going to be used for the summation of the curves.
		 *	By default, the region for each column (with >= 3 peaks) starts at the first rayleigh peak detected, and
		 *	ends at the last rayleigh peak detected.
		 *
		 *	You can extend that region by a number of periods before the first peak and after the last peak.
		 *	A period is the distance between 2 consecutives rayleigh peaks in linearized space. This is usefull
		 *	if you didn't detect a peak, but you know it's there.
		 *
		 *
		 *   \param[in] cec The curve extraction context. Needs to be consistent with the previous calls.
		 *   \param[in] height Height of the image
		 *   \param[in] width Width of the image
		 *   \param[in] peak_numbers 1D array containing for each column the number of rayleigh peaks detected.
		 *		If there are < 2, no summation will be done on this column. dim : image.width
		 *   \param[in] original_peak_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The positions are given in the pixel of the image, without linearisation.
		 *		dimension : image.width * cec.max_n_peaks
		 *   \param[in] a 1D array containing the first coefficient for the linearization of pixel of the column. dimension : image.width
		 *		(a*x*x + b*x + c)
		 *   \param[in] b 1D array containing the second coefficient for the linearization of pixel of the column. dimension : image.width
		 *		(a*x*x + b*x + c)
		 *   \param[in] c 1D array containing the third coefficient for the linearization of pixel of the column. dimension : image.width
		 *		(a*x*x + b*x + c)
		 *	 \param[out] start_ROI 1D array containing the index of the start of the region of interest, i.e. the region of the image which
		 *		is taken into account for the summation of the curves
		 *		dimension : image.width
		 *	 \param[out] end_ROI 1D array containing the index of the end of the region of interest, i.e. the region of the image which
		 *		is taken into account for the summation of the curves
		 *		dimension : image.width
		 *
		 * \returns
		 *	 start_ROI and end_ROI : passed as pointers. Must already be allocated before calling this function.
		 *
		 *
		 *
		 * \ingroup standalone
		 *
		 */
		void create_ROI(Curve_Extraction_Context* cec, int height, int width, int* peak_numbers, float* original_peak_positions, float* a,
			float* b, float* c, int* start_ROI, int* end_ROI) {

			//Allocating memory and sending everything to the GPU
			float* gpu_a, * gpu_b, * gpu_c;
			gpuErrchk(cudaMalloc(&gpu_a, sizeof(float) * width));
			gpuErrchk(cudaMalloc(&gpu_b, sizeof(float) * width));
			gpuErrchk(cudaMalloc(&gpu_c, sizeof(float) * width));
			gpuErrchk(cudaMemcpy(gpu_a, a, sizeof(float) * width, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_b, b, sizeof(float) * width, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_c, c, sizeof(float) * width, cudaMemcpyHostToDevice));

			int* gpu_peak_numbers;
			float* gpu_original_peak_positions;
			gpuErrchk(cudaMalloc(&gpu_peak_numbers, sizeof(int) * width * 1));
			gpuErrchk(cudaMalloc(&gpu_original_peak_positions, sizeof(int) * width * cec->max_n_peaks));
			gpuErrchk(cudaMemcpy(gpu_peak_numbers, peak_numbers, sizeof(int) * width * 1, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_original_peak_positions, original_peak_positions, sizeof(float) * width * cec->max_n_peaks, cudaMemcpyHostToDevice));

			int* gpu_start_ROI, * gpu_end_ROI;
			gpuErrchk(cudaMalloc(&gpu_start_ROI, sizeof(int) * width));
			gpuErrchk(cudaMalloc(&gpu_end_ROI, sizeof(int) * width));

			//Calculation
			LaunchKernel::create_ROI(cec, gpu_peak_numbers, gpu_original_peak_positions, width,
				height, gpu_a, gpu_b, gpu_c, cec->starting_order, cec->ending_order, gpu_start_ROI, gpu_end_ROI);

			//Getting results back
			gpuErrchk(cudaMemcpy(start_ROI, gpu_start_ROI, sizeof(int) * width, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(end_ROI, gpu_end_ROI, sizeof(int) * width, cudaMemcpyDeviceToHost));

			//Cleanup
			gpuErrchk(cudaFree(gpu_a));
			gpuErrchk(cudaFree(gpu_b));
			gpuErrchk(cudaFree(gpu_c));
			gpuErrchk(cudaFree(gpu_start_ROI));
			gpuErrchk(cudaFree(gpu_end_ROI));
			gpuErrchk(cudaFree(gpu_peak_numbers));
			gpuErrchk(cudaFree(gpu_original_peak_positions));
		}

		/**
		 *	A function to go from the image and the frq LUT to the summed curves. The summed curves start at -7.5 GHz and end at 7.5 GHz.
		 *	Only columns with more than 3 rayleigh peaks are taken into account. For each column, the summation starts at
		 *	the pixel index indicated in start_ROI and end at the pixel index indicated in end_ROI.
		 *
		 *	The edges may have NaN, because the extrem values may not be found in the frq_lut
		 *	All the inputs and outputs are expected to be CPU variables, the tranfers to/from the GPU is done internally.
		 *
		 * The interpolation between 2 pixels can be done with Linear or natural Spline interpolation.
		 *
		 *
		 *
		 *   \param[in] cec The curve extraction context. Needs to be consistent with the previsous calls.
		 *   \param[in] raw_data 16-bit image containing the rayleighs, stokes and antistokes peaks. dim : image.width * image.height
		 *   \param[in] height Height of the image
		 *   \param[in] width Width of the image
		 *   \param[in] start_ROI 1D array containing the pixel for each column where the summation of the curves starts. dim : Image.width
		 *   \param[in] end_ROI 1D array containing the pixel for each column where the summation of the curves ends. dim : Image.width
		 *   \param[in] peak_numbers 1D array containing for each column the number of rayleigh peaks detected.
		 *		If there are < 2, no summation will be done on this column. dim : Image.width
		 *   \param[in] frq_lut A 2D array where each value represents the frequency in linearized space of the corresponding pixel in the image.
		 *		dimension : image.width * image.height
		 *   \param[out] dest A 2D array containing the summed curves for each column. dimension : image.width * cec.n_points
		 *
		 *
		 * \returns
		 *	dest : the array must already by allocated to the correct size.
		 *
		 *
		 *
		 * \ingroup standalone
		 *
		 */
		void create_summed_curves(Curve_Extraction_Context* cec, uint16_t* raw_data, int height, int width, int* start_ROI, int* end_ROI,
			int* peak_numbers, float* frq_lut, float* dest) {

			//Allocating memory
			cudaExtent dim{ width, height, 1 };
			float* gpu_frq_lut;
			size_t lut_size = sizeof(float) * height * width;

			GPU_Image image;
			init_GPUImage(raw_data, dim, &image);

			float* gpu_dest;
			size_t dest_size = sizeof(float) * width * cec->n_points;

			int* gpu_peak_numbers;
			size_t array_size = sizeof(int) * width;

			gpuErrchk(cudaMalloc(&gpu_frq_lut, lut_size));
			gpuErrchk(cudaMalloc(&gpu_dest, dest_size));
			gpuErrchk(cudaMalloc(&gpu_peak_numbers, array_size * 1));


			//Sending data to GPU
			gpuErrchk(cudaMemcpy(gpu_frq_lut, frq_lut, lut_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_peak_numbers, peak_numbers, array_size, cudaMemcpyHostToDevice));


			int* gpu_start_ROI, * gpu_end_ROI;
			gpuErrchk(cudaMalloc(&gpu_start_ROI, sizeof(int) * dim.width));
			gpuErrchk(cudaMalloc(&gpu_end_ROI, sizeof(int) * dim.width));
			gpuErrchk(cudaMemcpy(gpu_start_ROI, start_ROI, sizeof(int) * dim.width, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_end_ROI, end_ROI, sizeof(int) * dim.width, cudaMemcpyHostToDevice));


			//Calling the kernel		
			size_t buffer_size;
			Spline_Buffers gpu_buffers;
			switch (cec->interpolation)
			{
			case LINEAR:

				LaunchKernel::combine_peaks(cec, image.texture, dim,
					gpu_peak_numbers, gpu_frq_lut, gpu_start_ROI, gpu_end_ROI, gpu_dest);


				break;
			case SPLINE:
				//Allocate spline buffers

				buffer_size = dim.width * dim.height * sizeof(float); //Oversized, in theory could be smaller if required
				gpuErrchk(cudaMalloc(&gpu_buffers.data_x, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.data_y, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.a, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.b, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.c, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.d, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.A, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.h, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.l, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.u, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_buffers.z, buffer_size));

				LaunchKernel::combine_peaks_spline(cec, image.texture, dim,
					gpu_peak_numbers, gpu_frq_lut, gpu_start_ROI, gpu_end_ROI, gpu_dest, gpu_buffers);

				//Free spline buffers
				gpuErrchk(cudaFree(gpu_buffers.data_x));
				gpuErrchk(cudaFree(gpu_buffers.data_y));
				gpuErrchk(cudaFree(gpu_buffers.a));
				gpuErrchk(cudaFree(gpu_buffers.b));
				gpuErrchk(cudaFree(gpu_buffers.c));
				gpuErrchk(cudaFree(gpu_buffers.d));
				gpuErrchk(cudaFree(gpu_buffers.A));
				gpuErrchk(cudaFree(gpu_buffers.h));
				gpuErrchk(cudaFree(gpu_buffers.l));
				gpuErrchk(cudaFree(gpu_buffers.u));
				gpuErrchk(cudaFree(gpu_buffers.z));

				break;
			default: //nothing
				break;
			}

			//Retrieving data from GPU
			gpuErrchk(cudaMemcpy(dest, gpu_dest, dest_size, cudaMemcpyDeviceToHost));

			//Freeing gpu memory
			gpuErrchk(cudaFree(gpu_frq_lut));
			gpuErrchk(cudaFree(gpu_dest));
			gpuErrchk(cudaFree(gpu_peak_numbers));
			gpuErrchk(cudaFree(gpu_start_ROI));
			gpuErrchk(cudaFree(gpu_end_ROI));
			deinit_GPUImage(&image);
		}

		/**
		 *	This function makes the bridge between the summed_curves and the fitting of the functions. It extracts the
		 *	sample points inside [start_frq, end_frq[ and puts them in arrays, those format is compatible with the fitting
		 *	functions.
		 *
		 *	All the inputs and outputs are expected to be CPU variables, the tranfers to/from the GPU is done internally.
		 *
		 *
		 *   \param[in] cec The curve extraction context. Needs to be consistent with the previsous calls.
		 *   \param[in] width Width of the image (=number of columns)
		 *   \param[in] summed_curves A 2D array containing the summed curves for each column. dimension : image.width * cec.n_points
		 *   \param[in] translation_lut A 1D array to do the translation between column number and fitting number. Used to pack the data,
		 *		so that fits are only done on columns with real data.
		 *		-1 indicates we don't use the column, a positive value is the fitting number (fit #0, #1, ... ,#(n_fits-1)).
		 *		 n_fits is appended at the end.
		 *		dimension : image.width + 1
		 *   \param[in] start_frq The lower bound of the frequency range which is going to be extracted (included).
		 *   \param[in] end_frq The upper bound of the frequency range which is going to be extracted (excluded).
		 *   \param[out] n_points The number of sample points extracted
		 *   \param[out] data_X 1D array containting all the x values of the extracted sample points.
		 *		dimension : n_points * n_fits
		 *   \param[out] data_Y 1D array containting all the y values of the extracted sample points.
		 *		dimension : n_points * n_fits
		 *
		 * \returns
		 * n_points, data_X and data_Y : the arrays must already by allocated to the correct size (or to a bigger size).
		 *
		 *
		 *
		 * \ingroup standalone
		 *
		 */
		void extract_data_for_fitting(Curve_Extraction_Context* cec, int width, float* summed_curves, int* translation_lut, float start_frq, float end_frq,
			int* n_points, float* data_X, float* data_Y) {

			//Determine the number of points going to be used
			int start_y = round((start_frq - cec->starting_freq) / cec->step);
			int end_y = round((end_frq - cec->starting_freq) / cec->step);
			n_points[0] = end_y - start_y;

			// Allocate and send to GPU memory
			float* gpu_summed_curves, * gpu_data_X, * gpu_data_Y;
			int* gpu_translation_lut;
			gpuErrchk(cudaMalloc(&gpu_summed_curves, sizeof(float) * width * cec->n_points));
			gpuErrchk(cudaMalloc(&gpu_data_X, sizeof(float) * translation_lut[width] * n_points[0]));
			gpuErrchk(cudaMalloc(&gpu_data_Y, sizeof(float) * translation_lut[width] * n_points[0]));
			gpuErrchk(cudaMalloc(&gpu_translation_lut, sizeof(int) * (width + 1)));

			gpuErrchk(cudaMemcpy(gpu_summed_curves, summed_curves, sizeof(float) * width * cec->n_points, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_translation_lut, translation_lut, sizeof(int) * (width + 1), cudaMemcpyHostToDevice));


			//Launch the GPU kernel
			LaunchKernel::extract_fitting_data(cec, gpu_summed_curves, width, gpu_translation_lut, start_y,
				end_y, gpu_data_X, gpu_data_Y);

			//Retrieve GPU data and cleanup
			gpuErrchk(cudaMemcpy(data_X, gpu_data_X, sizeof(float) * translation_lut[width] * n_points[0], cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(data_Y, gpu_data_Y, sizeof(float) * translation_lut[width] * n_points[0], cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(gpu_summed_curves));
			gpuErrchk(cudaFree(gpu_data_X));
			gpuErrchk(cudaFree(gpu_data_Y));
			gpuErrchk(cudaFree(gpu_translation_lut));
		}


		/**
		 * This function behaves the same as the @ref DLL_wrapper::extract_data_for_fitting "extract_data_for_fitting," except it
		 * does a recentering of the data based on the fit of the Rayleigh peak. The recentering is done before the data is
		 * extracted from the summed curve.
		 *
		 * Usefull in case the Rayleigh peak's center is not exactly at 0 Ghz. After some changes to the code, this function wasn't used anymore
		 * and may not work all the time. We stopped using it because it propagates the error from the rayleigh fit to the other fits,
		 * as well as having some definition problems about how to recenter in extrem cases.
		 *
		 *
		 *   \param[in] cec The curve extraction context. Needs to be consistent with the previsous calls.
		 *   \param[in] width Width of the image (=number of columns)
		 *   \param[in] summed_curves A 2D array containing the summed curves for each column. dimension : image.width * cec.n_points
		 *   \param[in] translation_lut A 1D array to do the translation between column number and fitting number. Used to pack the data,
		 *		so that fits are only done on columns with real data.
		 *		-1 indicates we don't use the column, a positive value is the fitting number (fit #0, #1, ... ,#(n_fits-1)).
		 *		 n_fits is appended at the end.
		 *		dimension : image.width + 1
		 *   \param[in] start_frq The lower bound of the frequency range which is going to be extracted (included).
		 *   \param[in] end_frq The upper bound of the frequency range which is going to be extracted (excluded).
		 *   \param[out] n_points The number of sample points extracted
		 *   \param[out] data_X 1D array containting all the x values of the extracted sample points.
		 *		dimension : n_points * n_fits
		 *   \param[out] data_Y 1D array containting all the y values of the extracted sample points.
		 *		dimension : n_points * n_fits
		 *
		 * \returns
		 * n_points, data_X and data_Y : the arrays must already by allocated to the correct size (or to a bigger size).
		 *
		 *
		 *
		 * \ingroup standalone
		 *
		 */
		void extract_data_for_fitting_recentering(Curve_Extraction_Context* cec, int width, float* summed_curves, int* translation_lut, float start_frq, float end_frq,
			int* n_points, float* data_X, float* data_Y, Fitted_Function fitted_rayleigh) {

			//Determine the number of points going to be used
			int start_y = round((start_frq - cec->starting_freq) / cec->step);
			int end_y = round((end_frq - cec->starting_freq) / cec->step);
			n_points[0] = end_y - start_y;

			// Allocate and send to GPU memory
			float* gpu_summed_curves, * gpu_data_X, * gpu_data_Y;
			int* gpu_translation_lut;
			gpuErrchk(cudaMalloc(&gpu_summed_curves, sizeof(float) * width * cec->n_points));
			gpuErrchk(cudaMalloc(&gpu_data_X, sizeof(float) * translation_lut[width] * n_points[0]));
			gpuErrchk(cudaMalloc(&gpu_data_Y, sizeof(float) * translation_lut[width] * n_points[0]));
			gpuErrchk(cudaMalloc(&gpu_translation_lut, sizeof(int) * (width + 1)));

			gpuErrchk(cudaMemcpy(gpu_summed_curves, summed_curves, sizeof(float) * width * cec->n_points, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_translation_lut, translation_lut, sizeof(int) * (width + 1), cudaMemcpyHostToDevice));

			/* Putting the Rayleigh fit into the correct format for the dynamic recentering */
			int n_fits = translation_lut[width];
			int* gpu_rayleigh_sanity;
			float* gpu_rayleigh_param, * rayleigh_param = new float[4 * n_fits];
			for (int i = 0; i < n_fits; i++) {
				rayleigh_param[4 * i + 0] = fitted_rayleigh.amplitude[i];
				rayleigh_param[4 * i + 1] = fitted_rayleigh.shift[i];
				rayleigh_param[4 * i + 2] = fitted_rayleigh.width[i];
				rayleigh_param[4 * i + 3] = fitted_rayleigh.offset[i];
			}
			gpuErrchk(cudaMalloc(&gpu_rayleigh_sanity, n_fits * sizeof(int)));
			gpuErrchk(cudaMalloc(&gpu_rayleigh_param, 4 * n_fits * sizeof(float)));
			gpuErrchk(cudaMemcpy(gpu_rayleigh_sanity, fitted_rayleigh.sanity, n_fits * sizeof(int), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_rayleigh_param, rayleigh_param, 4 * n_fits * sizeof(float), cudaMemcpyHostToDevice));

			//Launch the GPU kernel
			LaunchKernel::extract_fitting_data_dynamic_recentering(cec, gpu_summed_curves, width, gpu_translation_lut,
				start_y,
				end_y, gpu_data_X, gpu_data_Y, gpu_rayleigh_param, gpu_rayleigh_sanity);

			//Retrieve GPU data and cleanup
			gpuErrchk(cudaMemcpy(data_X, gpu_data_X, sizeof(float) * translation_lut[width] * n_points[0], cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(data_Y, gpu_data_Y, sizeof(float) * translation_lut[width] * n_points[0], cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(gpu_summed_curves));
			gpuErrchk(cudaFree(gpu_data_X));
			gpuErrchk(cudaFree(gpu_data_Y));
			gpuErrchk(cudaFree(gpu_translation_lut));

			gpuErrchk(cudaFree(gpu_rayleigh_sanity));
			gpuErrchk(cudaFree(gpu_rayleigh_param));



		}

		/** Returns an estimation of the background noise, used for the SNR check.
		 *
		 * To do so, it only uses the part of the curve which is not used for the fits : [start_frequency ; stokes_start_frequency [
		 * and [antistokes_end_frequency ; end_frequency [. The noise deviation is estimated as the standard deviation of the signal
		 * in these ranges.
		 *
		 * \param[in] cec The curve extraction context. Needs to be consistent with the previsous calls.
		 * \param[in] width Width of the image (=number of columns)
		 * \param[in] summed_curves A 2D array containing the summed curves for each column. dimension : image.width * cec.n_points
		 * \param[in] translation_lut A 1D array to do the translation between column number and fitting number. Used to pack the data,
		 *		so that fits are only done on columns with real data.
		 *		-1 indicates we don't use the column, a positive value is the fitting number (fit #0, #1, ... ,#(n_fits-1)).
		 *		 n_fits is appended at the end.
		 *		dimension : image.width + 1
		 * \param[in] stokes_start_frq The frequency where the stokes signal starts.
		 * \param[in] antistokes_end_frq The frequency where the antistokes signal ends.
		 * \param[out] noise_deviation A 1D array containing the noise deviation value for each colum to fit.
		 *		dimension : n_fits
		 *
		 * \ingroup standalone
		 */
		void estimate_noise_deviation(Curve_Extraction_Context* cec, int width, float* summed_curves, int* translation_lut,
			float stokes_start_frq, float antistokes_end_frq, float* noise_deviation) {

			//Allocate and send to GPU memory
			int start_stokes_y = round((stokes_start_frq - cec->starting_freq) / cec->step);
			int end_antistokes_y = round((antistokes_end_frq - cec->starting_freq) / cec->step);
			float* gpu_noise_deviation, * gpu_summed_curves;
			int* gpu_translation_lut;
			gpuErrchk(cudaMalloc(&gpu_noise_deviation, sizeof(float) * translation_lut[width] * 1));
			gpuErrchk(cudaMalloc(&gpu_translation_lut, sizeof(int) * (width + 1)));
			gpuErrchk(cudaMalloc(&gpu_summed_curves, sizeof(float) * width * cec->n_points));
			gpuErrchk(cudaMemcpy(gpu_translation_lut, translation_lut, sizeof(int) * (width + 1), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_summed_curves, summed_curves, sizeof(float) * width * cec->n_points, cudaMemcpyHostToDevice));

			//Launch Cuda kernel
			LaunchKernel::estimate_noise_signal(cec, width, gpu_summed_curves, gpu_translation_lut, start_stokes_y, end_antistokes_y, gpu_noise_deviation);

			//Copy result from GPU to CPU and cleanup
			gpuErrchk(cudaMemcpy(noise_deviation, gpu_noise_deviation, sizeof(float) * translation_lut[width] * 1, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaFree(gpu_noise_deviation));
			gpuErrchk(cudaFree(gpu_translation_lut));
			gpuErrchk(cudaFree(gpu_summed_curves));
		}


		/** The wrapper function to call the GPUFit library and to do the fit.
		 *
		 *
		 * \param[in] data_X 1D array containting all the x values of the extracted sample points.
		 *		dimension : n_points * n_fits
		 * \param[in] data_Y 1D array containting all the y values of the extracted sample points.
		 *		dimension : n_points * n_fits
		 * \param[in] error_deviation A 1D array containing the noise deviation value for each colum to fit.
		 *		dimension : n_fits
		 * \param[in] n_points The number of points in the extracted samples used for the fits.
		 * \param[in] fitting_function An enumeration to choose which function to use for the fitting.
		 * \param[in] fit_co Information relating to how the fitting algorithm should behave and how to know if a fit is sane.
		 * \param[in] angle_context Information relating to how to compute Stokes/antiStokes function.
		 *		It will have no impact of the result if you use another function.
		 * \param[out] fitted_function A struct containing the fitted parameters of each fit.
		 *		The memory needs to be pre-allocated.
		 *
		 * \ingroup standalone
		 */
		void fitting(float* data_X, float* data_Y, float* error_deviation, int n_points,
			Fitting_Function_t fitting_function,
			Fitting_Context* fit_co,
			Stokes_antiStokes_Context* angle_context,
			Fitted_Function* fitted_function) {

			//Prepare for a fit to gpufit
			PeakFitting* fitting = create_PeakFitting_no_extraction(fitting_function, n_points, fit_co, angle_context);


			//Creating initialized parameters
			fitting->get_initial_parameters(data_X, data_Y, fitting->initial_parameters, HOST);
			//if(fitting->parameterContext->use_constraints)
			fitting->determine_fitting_constraints(data_X, data_Y);

			//Fitting the function
			fitting->no_copy_fit(data_X, data_Y, fitting->initial_parameters, NULL);
			fitting->sanity_check(error_deviation);

			//Getting the results
			fitting->export_fitted_parameters(fitted_function->amplitude, fitted_function->shift, fitted_function->width, fitted_function->offset);
			fitting->copy_array(fitting->get_sanity(), fitting->storage, fitted_function->sanity, HOST, sizeof(int) * fitting->n_fits);

#ifndef _WINDLL //todo
			fitting->save_to_file(data_X, data_Y, fitting->fit_parameters, fitting->output_states, fitting->output_n_iterations, fitting->get_SNR(),
				fitting->get_sanity(), "Temp_results/GPU_Standalone_stokes.txt");
#endif //_WINDLL
			delete fitting;
		}

		/** Given a fitted function, returns the Y-value of that function for given X-coordinates.
		 *
		 *
		 * \param[in] data_X 1D array containting all the x values of the extracted sample points.
		 *		dimension : n_points * n_fits
		 * \param[in] n_points The number of points in the extracted samples used for the fits.
		 * \param[in] n_fits The number of fits done in the batch.
		 * \param[in] function An enumaration to choose which function model should be used for the computation.
		 * \param[in] fitted_function A struct containing the fitted parameters of each fit.
		 * \param[in] angle_context Information relating to how to compute Stokes/antiStokes function.
		 *		It will have no impact of the result if you use another function.
		 * \param[out] function_Y 1D array containting all the computed y values of the functions for the given x-coordinates
		 *		dimension : n_points * n_fits
		 * \return
		 *
		 * \ingroup standalone
		 */
		void calculate_fitted_curve(float* data_X, int n_points, int n_fits,
			Fitting_Function_t function, Fitted_Function* fitted_function, Stokes_antiStokes_Context* angle_context, float* function_Y) {
			Fitting_Context fit_co;
			fit_co.max_iteration = 30;
			fit_co.n_fits = n_fits;
			fit_co.SNR_threshold = 2;
			fit_co.tolerance = 1e-5;
			PeakFitting* func = create_PeakFitting_no_extraction(function, n_points, &fit_co, angle_context);
			func->compute_fitted_curve(data_X, fitted_function, function_Y);


		}

		// ==========
		//  PIPELINE
		// ==========

		/* Global variables : information stored inside the GPU*/
		GPU_Image* image;								//Image wrapper for GPU: 2D array + Texture
		float* gpu_summed_curv_buf, * gpu_frq_lut;		//Buffers to store the summed curves and the frequency look-up-table
		int* gpu_peak_numbers, * gpu_translation_lut;	//Informations about the spotted rayleigh peaks
		float* gpu_original_peak_positions;
		int* gpu_start_ROI, * gpu_end_ROI;				//Where to start and end the summation of the curves
		float* gpu_error_deviation;						//for the sanity check of the fits
		Spline_Buffers gpu_spline_buffers;				//In case of Spline interpolation

		//living in GPU space
		PeakFitting* rayleigh;
		PeakFitting* stokes;
		PeakFitting* antistokes;

		Curve_Extraction_Context* curve_extraction_context;


		/*
			* Internal function
			* is used by pipeline_sum_and_fit and pipeline_sum_and_fit_to_array
			*
			* *not* used by pipeline_sum_and_fit_timed
			*/
		void pipeline_sum_and_fit_internal(uint16_t* cpu_raw_data,
			bool dynamic_recentering) {
			// Sending new image to the GPU
			update_GPUImage(cpu_raw_data, image);

			// Creating summed up curves
			switch (curve_extraction_context->interpolation) {
			case LINEAR:
				LaunchKernel::combine_peaks(
					curve_extraction_context, image->texture, image->size,
					gpu_peak_numbers, gpu_frq_lut, gpu_start_ROI,
					gpu_end_ROI, gpu_summed_curv_buf);
				break;
			case SPLINE:
				LaunchKernel::combine_peaks_spline(
					curve_extraction_context, image->texture, image->size,
					gpu_peak_numbers, gpu_frq_lut, gpu_start_ROI,
					gpu_end_ROI, gpu_summed_curv_buf, gpu_spline_buffers);
				break;
			default:  // nothing - could be a good idea to throw an
						// error, or to return with nothing
				break;
			}

			LaunchKernel::estimate_noise_signal(
				curve_extraction_context, image->size.width,
				gpu_summed_curv_buf, gpu_translation_lut,
				stokes->get_start_y(), antistokes->get_end_y(),
				gpu_error_deviation);
			cudaDeviceSynchronize();

			// Fitting the rayleigh peak
			rayleigh->extract_data(gpu_summed_curv_buf, image->size.width,
				gpu_translation_lut, DEVICE);
			rayleigh->get_initial_parameters();
			if (rayleigh->use_fit_constraints())
				rayleigh->determine_fitting_constraints();
			rayleigh->fit();
			rayleigh->sanity_check(gpu_error_deviation);
			cudaDeviceSynchronize();

			// Extracting data of Stokes and antistokes peak (taking into
			// account recentering)
			if (dynamic_recentering) {
				stokes->extract_recentered_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE, rayleigh);
				antistokes->extract_recentered_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE, rayleigh);
			}
			else {
				stokes->extract_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE);
				antistokes->extract_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE);
			}

			// Fitting the antistokes peak
			stokes->get_initial_parameters();
			if (stokes->use_fit_constraints())
				stokes->determine_fitting_constraints();
			stokes->fit();
			stokes->sanity_check(gpu_error_deviation);

			cudaDeviceSynchronize();

			// Fitting the stokes peak
			antistokes->get_initial_parameters();
			if (antistokes->use_fit_constraints())
				antistokes->determine_fitting_constraints();
			antistokes->fit();
			antistokes->sanity_check(gpu_error_deviation);

			cudaDeviceSynchronize();
		}

		/**
			* The main function to call.
			* Once the gpu buffers are initialized and full, you can call this to fit the stokes,
			* rayleigh and antistokes peaks. You get the parameters for the functions as a result.
			*
			* \param[in] cpu_raw_data The image which is going to be used as basis for the fitting. dimension : width*height
			*
			* \param[in] dynamic_recentering Boolean to choose whether to automatically recenter the x-axis of the curve or not.
			*		If *true*, the fitted center of the rayleigh peak is used to shift the stokes peak and the antistokes peak
			*		(x -= rayleigh_x0). This is done before the data extraction, so it may affect the frequency range choosen.
			*		If *false*, the x-axis of the peaks aren't changed. *For the moment, please use False*
			* \param[out] fitted_stokes A structure containing the fitted parameters for each fit of the Stokes peak.
			* \param[out] fitted_rayleigh A structure containing the fitted parameters for each fit of the Rayleigh peak.
			* \param[out] fitted_antistokes A structure containing the fitted parameters for each fit of the antiStokes peak.
			*
			* \returns
			*	*fitted_stokes*, *fitted_rayleigh* and *fitted_antistokes* are the return variables.
			*	They need to be allocated before being passed to this function.
			*
			* \ingroup pipeline
			*/
		void pipeline_sum_and_fit(uint16_t* cpu_raw_data,
			bool dynamic_recentering,
			Fitted_Function* fitted_stokes,
			Fitted_Function* fitted_rayleigh,
			Fitted_Function* fitted_antistokes) {

			// Process the image
			pipeline_sum_and_fit_internal(cpu_raw_data, dynamic_recentering);

			// Return the results
			rayleigh->export_fitted_parameters(
				fitted_rayleigh->amplitude, fitted_rayleigh->shift,
				fitted_rayleigh->width, fitted_rayleigh->offset);
			rayleigh->export_sanity(fitted_rayleigh->sanity);

			stokes->export_fitted_parameters(
				fitted_stokes->amplitude, fitted_stokes->shift,
				fitted_stokes->width, fitted_stokes->offset);
			stokes->export_sanity(fitted_stokes->sanity);

			antistokes->export_fitted_parameters(
				fitted_antistokes->amplitude, fitted_antistokes->shift,
				fitted_antistokes->width, fitted_antistokes->offset);
			antistokes->export_sanity(fitted_antistokes->sanity);
		}

		/**
			* The main function to call. Once the gpu buffers are
			* initialized and full, you can call this to fit the stokes,
			* rayleigh and antistokes peaks. You get the parameters for the
			* functions as a result.
			*
			* \param[in] cpu_raw_data The image which is going to be used
			*		as basis for the fitting. dimension : width*height
			* \param[in] *dynamic_recentering Boolean to choose whether to automatically
			*		recenter the x-axis of the curve or not. If *true*, the fitted center of the rayleigh peak is used to shift the stokes peak
			*		and the antistokes peak (x -= rayleigh_x0). This is done before the data extraction, so it may affect the frequency
			*		range choosen. If *false*, the x-axis of the peaks aren't	changed. *For the moment, please use False*
			* \param[out] fit_stokes Array containt the result of the fit ([amplitude, shift, width, offset, amplitude, shift, width, offset, ...]).
			*		dimension : 4*n_fits
			* \param[out] fit_rayleigh Array containt the result of the fit ([amplitude, shift, width, offset, amplitude, shift, width, offset, ...]).
			*		dimension : 4*n_fits
			* \param[out] fit_antistokes Array containt the result of the fit ([amplitude, shift, width, offset, amplitude, shift, width, offset, ...]).
			*		dimension : 4*n_fits
			*
			* \returns
			*	*fit_stokes*, *fit_rayleigh* and *fit_antistokes* are the return variables. They need to be allocated before being passed to this function.
			* \ingroup pipeline
			*/
		void pipeline_sum_and_fit_to_array(uint16_t* cpu_raw_data,
			bool dynamic_recentering,
			float* fitted_stokes,
			float* fitted_rayleigh,
			float* fitted_antistokes) {

			// Process the image
			pipeline_sum_and_fit_internal(cpu_raw_data, dynamic_recentering);

			// Return the results - into pre-allocated arrays
			rayleigh->export_fitted_parameters(fitted_rayleigh);
			stokes->export_fitted_parameters(fitted_stokes);
			antistokes->export_fitted_parameters(fitted_antistokes);
		}

		/**
		 * Same function as pipeline_sum_and_fit except this version take measurements of each step of the algorithm is taking.
		 * Therefore, it should be slower than the other version : only use it for debugging or/and performance measurements.
		 * The timings are :
		 *  - 1. Copying the data to the GPU
		 *  - 2. Fusing the orders
		 *  - 3. Pre-processing
		 *  - 4. Fitting the rayleigh peak
		 *  - 5. post-processing
		 *  - 6. Fitting antistokes peak
		 *  - 7. Fitting the stokes peak
		 *  - 8. Sending the data to the CPU
		 *
		 * \param[in] cpu_raw_data The image which is going to be used as basis for the fitting. dimension : width*height
		 * \param[in] dynamic_recentering  boolean to choose whether to automatically recenter the x-axis of the curve or not.
		 *		If *true*, the fitted center of the rayleigh peak is used to shift the stokes peak and the antistokes peak
		 *		(x -= rayleigh_x0). This is done before the data extraction, so it may affect the frequency range choosen.
		 *		If *false*, the x-axis of the peaks aren't changed.
		 * \param[out] fitted_stokes A structure containing the fitted parameters for each fit of the Stokes peak.
		 * \param[out] fitted_rayleigh A structure containing the fitted parameters for each fit of the Rayleigh peak.
		 * \param[out] fitted_antistokes A structure containing the fitted parameters for each fit of the antiStokes peak.
		 * \param[out] timings A float array of size at least 8.
		 *
		 * \returns
		 *	*fitted_stokes*, *fitted_rayleigh*, *fitted_antistokes* and *timings* are the return variables. They need to be allocated
		 *	before being passed to this function.
		 *
		 * \ingroup pipeline
		 */
		void pipeline_sum_and_fit_timed(uint16_t* cpu_raw_data, bool dynamic_recentering,
			Fitted_Function* fitted_stokes, Fitted_Function* fitted_rayleigh, Fitted_Function* fitted_antistokes, float* timings) {


			//Create the Cuda event to measure the elapsed time
			cudaEvent_t start, host_to_Device, fusion, pre_processing, post_processing,
				timer_rayleigh, timer_antistokes, timer_stokes, device_to_host;
			cudaEventCreate(&start);
			cudaEventCreate(&host_to_Device);
			cudaEventCreate(&fusion);
			cudaEventCreate(&pre_processing);
			cudaEventCreate(&timer_rayleigh);
			cudaEventCreate(&post_processing);
			cudaEventCreate(&timer_antistokes);
			cudaEventCreate(&timer_stokes);
			cudaEventCreate(&device_to_host);

			//Sending new image to the GPU
			cudaEventRecord(start);
			update_GPUImage(cpu_raw_data, image);
			cudaEventRecord(host_to_Device);

			//Creating summed up curves
			switch (curve_extraction_context->interpolation)
			{
			case LINEAR:
				LaunchKernel::combine_peaks(curve_extraction_context, image->texture, image->size,
					gpu_peak_numbers, gpu_frq_lut, gpu_start_ROI, gpu_end_ROI, gpu_summed_curv_buf);
				break;
			case SPLINE:
				LaunchKernel::combine_peaks_spline(curve_extraction_context, image->texture, image->size,
					gpu_peak_numbers, gpu_frq_lut, gpu_start_ROI, gpu_end_ROI, gpu_summed_curv_buf, gpu_spline_buffers);
				break;
			default: //nothing
				break;
			}

			LaunchKernel::estimate_noise_signal(curve_extraction_context, image->size.width, gpu_summed_curv_buf,
				gpu_translation_lut, stokes->get_start_y(), antistokes->get_end_y(), gpu_error_deviation);
			cudaEventRecord(fusion);
			cudaDeviceSynchronize();
			cudaEventRecord(pre_processing);
			//Fitting the rayleigh peak
			rayleigh->extract_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE);
			rayleigh->get_initial_parameters();
			if (rayleigh->use_fit_constraints())
				rayleigh->determine_fitting_constraints();
			rayleigh->fit();
			rayleigh->sanity_check(gpu_error_deviation);
			cudaEventRecord(timer_rayleigh);
			cudaDeviceSynchronize();

			//Extracting data of Stokes and antistokes peak (taking into account recentering)
			if (dynamic_recentering) {
				stokes->extract_recentered_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE, rayleigh);
				antistokes->extract_recentered_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE, rayleigh);
			}
			else {
				stokes->extract_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE);
				antistokes->extract_data(gpu_summed_curv_buf, image->size.width, gpu_translation_lut, DEVICE);
			}
			cudaEventRecord(post_processing);

			//Fitting the antistokes peak
			stokes->get_initial_parameters();
			if (stokes->use_fit_constraints())
				stokes->determine_fitting_constraints();
			stokes->fit();
			stokes->sanity_check(gpu_error_deviation);
			cudaEventRecord(timer_antistokes);
			cudaDeviceSynchronize();

			//Fitting the stokes peak
			antistokes->get_initial_parameters();
			if (antistokes->use_fit_constraints())
				antistokes->determine_fitting_constraints();
			antistokes->fit();
			antistokes->sanity_check(gpu_error_deviation);
			cudaEventRecord(timer_stokes);
			cudaDeviceSynchronize();


			//Return the results
			rayleigh->export_fitted_parameters(fitted_rayleigh->amplitude, fitted_rayleigh->shift, fitted_rayleigh->width, fitted_rayleigh->offset);
			rayleigh->export_sanity(fitted_rayleigh->sanity);
			stokes->export_fitted_parameters(fitted_stokes->amplitude, fitted_stokes->shift, fitted_stokes->width, fitted_stokes->offset);
			stokes->export_sanity(fitted_stokes->sanity);
			antistokes->export_fitted_parameters(fitted_antistokes->amplitude, fitted_antistokes->shift, fitted_antistokes->width, fitted_antistokes->offset);
			antistokes->export_sanity(fitted_antistokes->sanity);
			cudaEventRecord(device_to_host);

			//Wait for last Cuda event to happend, and then retrieve the elapsed times
			cudaEventSynchronize(device_to_host);
			cudaEventElapsedTime(&timings[0], start, host_to_Device);
			cudaEventElapsedTime(&timings[1], host_to_Device, fusion);
			cudaEventElapsedTime(&timings[2], fusion, pre_processing);
			cudaEventElapsedTime(&timings[3], pre_processing, timer_rayleigh);
			cudaEventElapsedTime(&timings[4], timer_rayleigh, post_processing);
			cudaEventElapsedTime(&timings[5], post_processing, timer_antistokes);
			cudaEventElapsedTime(&timings[6], timer_antistokes, timer_stokes);
			cudaEventElapsedTime(&timings[7], timer_stokes, device_to_host);

		}

		/**	\brief Optional function to call to get more informations about the last fit.
		*
		*	This function copies the output states variables of the GPUfit to CPU array to do some goodness of fit analysis.
		*	If this function is called before a @ref pipeline_sum_and_fit then the state of the GPU memory is random, and the results are
		*	gibberish : only call this function after pipeline_sum_and_fit. Sameways, a new call of pipeline_sum_and_fit will override the
		*	memory in the GPU, and the previous gof metric will be overwritten : This function can only retrieve the gof of the last fitting.
		*
		*
		* \param[out] stokes_gof The goodness of fit structure containing the information about the Stoke fit.
		* \param[out] rayleigh_gof The goodness of fit structure containing the information about the Rayleigh fit.
		* \param[out] antistokes_gof The goodness of fit structure containing the information about the antiStoke fit.
		*
		* \returns
		*	*stokes_gof*, *rayleigh_gof* and *antistokes_gof* have to be preallocated prior to being used in this function.
		*
		* \ingroup pipeline
		*/
		void pipeline_get_gof(Goodness_Of_Fit* stokes_gof, Goodness_Of_Fit* rayleigh_gof, Goodness_Of_Fit* antistokes_gof) {

			gpuErrchk(cudaMemcpy(stokes_gof->states, stokes->output_states, sizeof(int) * stokes->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(stokes_gof->chi_squares, stokes->output_chi_squares, sizeof(float) * stokes->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(stokes_gof->n_iterations, stokes->output_n_iterations, sizeof(int) * stokes->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(stokes_gof->SNR, stokes->get_SNR(), sizeof(float) * stokes->n_fits, cudaMemcpyDeviceToHost));


			gpuErrchk(cudaMemcpy(rayleigh_gof->states, rayleigh->output_states, sizeof(int) * rayleigh->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(rayleigh_gof->chi_squares, rayleigh->output_chi_squares, sizeof(float) * rayleigh->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(rayleigh_gof->n_iterations, rayleigh->output_n_iterations, sizeof(int) * rayleigh->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(rayleigh_gof->SNR, rayleigh->get_SNR(), sizeof(float) * rayleigh->n_fits, cudaMemcpyDeviceToHost));

			gpuErrchk(cudaMemcpy(antistokes_gof->states, antistokes->output_states, sizeof(int) * antistokes->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(antistokes_gof->chi_squares, antistokes->output_chi_squares, sizeof(float) * antistokes->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(antistokes_gof->n_iterations, antistokes->output_n_iterations, sizeof(int) * antistokes->n_fits, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(antistokes_gof->SNR, antistokes->get_SNR(), sizeof(float) * antistokes->n_fits, cudaMemcpyDeviceToHost));
		}


		/** Call this function if you want to constraint the parameters of the function.
		 *  All 3 fits use theses parameters to compute their constraints. They are based on the maximum found inside
		 *  the frequency range of the function.
		 *  Keep in mind that the Gaussian/Lorentzian function and the Stokes/Antistokes don't have the same amplitude scale, as
		 *  the functions aren't normalized. Therefore, a too big min_amplitude_of_maximum blocks the fit of the
		 *  Stokes/Antisokes functions, and a too small max_amplitude_of_maximum blocks the fit of the Gaussian/Lorentzian function.
		 *  Recommended is min_amplitude_of_maximum = 0 and max_amplitude_of_maximum = 0.90
		 *
		 * .
		 *
		 * \param[in] use_constraints 'true' if you want to use the constraints, 'false' otherwise.
		 *		The other values don't matter if you select 'false'.
		 * \param[in] min_width The lower bound of the width parameter of the functions.
		 * \param[in] max_width The upper bound of the width parameter of the functions.
		 * \param[in] max_distance_to_maximum The shift's lower bound is position_of_maximum - max_distance_to_maximum.
		 *		The shift's upper bound is position_of_maximum + max_distance_to_maximum.
		 * \param[in] min_amplitude_of_maximum The lower bound of the amplitude is amplitude_of_maximum * min_amplitude_of_maximum.
		 * \param[in] max_amplitude_of_maximum The upper bound of the amplitude is amplitude_of_maximum * max_amplitude_of_maximum.
		 *
		 * \ingroup pipeline
		 */
		void pipeline_set_constraints(bool use_constraints, float min_width, float max_width, float max_distance_to_maximum,
			float min_amplitude_of_maximum, float max_amplitude_of_maximum)
		{
			rayleigh->constraint_settings(use_constraints, min_width, max_width, max_distance_to_maximum,
				min_amplitude_of_maximum, max_amplitude_of_maximum);
			stokes->constraint_settings(use_constraints, min_width, max_width, max_distance_to_maximum,
				min_amplitude_of_maximum, max_amplitude_of_maximum);
			antistokes->constraint_settings(use_constraints, min_width, max_width, max_distance_to_maximum,
				min_amplitude_of_maximum, max_amplitude_of_maximum);

		}


		/**
		 * 	Call this function to initialize all the global buffer used by this library. This has to be
		 *	called before <pipeline_sum_and_fit> and <pipeline_send_experiment_settings>. The parameters used here are
		 *	final : calling this function again will likely cause a memory leak, if it works. Call <pipeline_close> if
		 *  you wish to reset the internal global variables.
		 *	The data range used for the antistokes peak is [frq_l0 ; frq_l1 [.
		 *	The data range used for the rayleigh peak is [frq_l1 ; frq_l2 [.
		 *
		 * \param[in] width Width of the image.
		 * \param[in] height Height of the image.
		 * \param[in] gpufit_context Information relating to how the fitting algorithm should behave and how to know if a fit is sane.
		 * \param[in] cec Information relating to how to convert from the image to the summed curves.
		 * \param[in] stokes_fitting_function Choose which function to fit over the Stokes peak.
		 * \param[in] rayleigh_fitting_function Choose which function to fit over the Rayleigh peak.
		 * \param[in] antistokes_fitting_function Choose which function to fit over the antiStokes peak.
		 * \param[in] stokes_range The range of the frequency where the stokes peak is found.
		 *		Array containing 2 values : lowerbound (included) and upperbound (excluded). dimension : 2
		 * \param[in] rayleigh_range The range of the frequency where the rayleigh peak is found.
		 *		Array containing 2 values : lowerbound (included) and upperbound (excluded). dimension : 2
		 * \param[in] antistokes_range The range of the frequency where the antistokes peak is found.
		 *		Array containing 2 values : lowerbound (included) and upperbound (excluded). dimension : 2
		 * \param[in] angle_context Information relating to how to compute Stokes/antiStokes function.
		 *		It will have no impact of the result if you use another function.
		 *
		 * \ingroup pipeline
		 */
		void pipeline_initialisation(int width, int height, Fitting_Context* gpufit_context, Curve_Extraction_Context* cec,
			Fitting_Function_t stokes_fitting_function, Fitting_Function_t rayleigh_fitting_function, Fitting_Function_t antistokes_fitting_function,
			float* stokes_range, float* rayleigh_range, float* antistokes_range, Stokes_antiStokes_Context* angle_context) {

			/* Getting the curve extraction context */
			curve_extraction_context = new Curve_Extraction_Context;
			curve_extraction_context->max_n_peaks = cec->max_n_peaks;
			curve_extraction_context->n_points = cec->n_points;
			curve_extraction_context->starting_freq = cec->starting_freq;
			curve_extraction_context->step = cec->step;
			curve_extraction_context->interpolation = cec->interpolation;

			/* Pre-processing of the data*/
			cudaExtent dim{ width, height, 1 };
			uint16_t* fake_data = new uint16_t[width * height];
			image = new GPU_Image;
			init_GPUImage(fake_data, dim, image);
			delete[] fake_data;

			gpuErrchk(cudaMalloc(&gpu_frq_lut, sizeof(float) * dim.width * dim.height));
			gpuErrchk(cudaMalloc(&gpu_summed_curv_buf, sizeof(float) * dim.width * curve_extraction_context->n_points));
			gpuErrchk(cudaMalloc(&gpu_peak_numbers, sizeof(int) * dim.width * 1));
			gpuErrchk(cudaMalloc(&gpu_original_peak_positions, sizeof(int) * dim.width * curve_extraction_context->max_n_peaks));
			gpuErrchk(cudaMalloc(&gpu_translation_lut, sizeof(int) * (dim.width + 1) * 1));
			gpuErrchk(cudaMalloc(&gpu_start_ROI, sizeof(int) * dim.width * 1));
			gpuErrchk(cudaMalloc(&gpu_end_ROI, sizeof(int) * dim.width * 1));

			gpuErrchk(cudaMalloc(&gpu_error_deviation, sizeof(float) * gpufit_context->n_fits * 1));


			/* Allocate memory for the spline fitting*/

			size_t buffer_size;
			switch (cec->interpolation)
			{
			case SPLINE: //Allocate gpu buffer for spline interpolation

				buffer_size = dim.width * dim.height * sizeof(float); //Oversized, in theory could be smaller if required
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.data_x, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.data_y, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.a, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.b, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.c, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.d, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.A, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.h, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.l, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.u, buffer_size));
				gpuErrchk(cudaMalloc(&gpu_spline_buffers.z, buffer_size));

				break;
			default: //Not spline interpolation, so no need for special buffers
				break;
			}

			/* create the Fitting modules */
			rayleigh = create_PeakFitting(rayleigh_fitting_function, rayleigh_range, gpufit_context, angle_context, curve_extraction_context);
			antistokes = create_PeakFitting(antistokes_fitting_function, antistokes_range, gpufit_context, angle_context, curve_extraction_context);
			stokes = create_PeakFitting(stokes_fitting_function, stokes_range, gpufit_context, angle_context, curve_extraction_context);
		}

		/**
		 *	Call this function once at the end of the program. It will clean up all memory used on the GPU,
		 *	as well as deleting the global variable used.
		 *
		 *
		 * \param
		 *
		 * \returns
		 *
		 * \relates
		 * \ingroup pipeline
		 */
		void pipeline_close() {
			/* Pre-processing of the data */
			deinit_GPUImage(image);
			gpuErrchk(cudaFree(gpu_frq_lut));
			gpuErrchk(cudaFree(gpu_peak_numbers));
			gpuErrchk(cudaFree(gpu_summed_curv_buf));
			gpuErrchk(cudaFree(gpu_original_peak_positions));
			gpuErrchk(cudaFree(gpu_translation_lut));
			gpuErrchk(cudaFree(gpu_start_ROI));
			gpuErrchk(cudaFree(gpu_end_ROI));
			gpuErrchk(cudaFree(gpu_error_deviation));

			switch (curve_extraction_context->interpolation)
			{
			case SPLINE: // free allocated gpu buffers
				gpuErrchk(cudaFree(gpu_spline_buffers.data_x));
				gpuErrchk(cudaFree(gpu_spline_buffers.data_y));
				gpuErrchk(cudaFree(gpu_spline_buffers.a));
				gpuErrchk(cudaFree(gpu_spline_buffers.b));
				gpuErrchk(cudaFree(gpu_spline_buffers.c));
				gpuErrchk(cudaFree(gpu_spline_buffers.d));
				gpuErrchk(cudaFree(gpu_spline_buffers.A));
				gpuErrchk(cudaFree(gpu_spline_buffers.h));
				gpuErrchk(cudaFree(gpu_spline_buffers.l));
				gpuErrchk(cudaFree(gpu_spline_buffers.u));
				gpuErrchk(cudaFree(gpu_spline_buffers.z));
				break;
			default:
				break;
			}

			/* Fitting*/
			delete rayleigh;
			delete antistokes;
			delete stokes;
		}

		/**
		 *	This function transfer data from CPU memory to GPU memory, into the allocated memory. Do this after initialisation
		 *	and before the first call of <pipeline_sum_and_fit>.
		 *
		 *
		 *   \param[in] peak_numbers A 1D array containing the number of rayleigh peaks detected in this column. dimension : image.width
		 *   \param[in] peak_original_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The position is given in the pixel of the image, without linearisation.
		 *		dimension : image.width * cec.max_n_peaks
		 *  \param[in] translation_lut A 1D array to do the translation between column number and fitting number. Used to pack the data,
		 *		so that fits are only done on columns with real data.
		 *		-1 indicates we don't use the column, a positive value is the fitting number (fit #0, #1, ... , #(n_fits-1)).
		 *		 n_fits is appended at the end.
		 *		dimension : image.width + 1
		 *   \param[in] frq_lut A 2D array where each value represents the frequency in linearized space of the corresponding pixel in the image.
		 *		dimension : image.width * image.height
		 *   \param[in] start_ROI A 1D array to indicate for each column at which pixel to start the sumation of the curves.
		 *		dimension : image.width
		 *   \param[in] end_ROI A 1D array to indicate for each column at which pixel to end the sumation of the curves.
		 *		dimension : image.width
		 *
		 * \returns
		 *
		 *
		 * \relates
		 * \ingroup pipeline
		 *
		 */
		void pipeline_send_experiment_settings(int* peak_numbers, float* peak_original_positions, int* translation_lut,
			float* frq_lut, int* start_ROI, int* end_ROI) {

			size_t peak_number_size = sizeof(int) * image->size.width;
			gpuErrchk(cudaMemcpy(gpu_peak_numbers, peak_numbers, peak_number_size, cudaMemcpyHostToDevice));

			size_t peak_original_positions_size = sizeof(float) * image->size.width * curve_extraction_context->max_n_peaks;
			gpuErrchk(cudaMemcpy(gpu_original_peak_positions, peak_original_positions, peak_original_positions_size, cudaMemcpyHostToDevice));

			size_t translation_lut_size = sizeof(int) * (image->size.width + 1);
			gpuErrchk(cudaMemcpy(gpu_translation_lut, translation_lut, translation_lut_size, cudaMemcpyHostToDevice));

			size_t frq_lut_size = sizeof(float) * image->size.width * image->size.height;
			gpuErrchk(cudaMemcpy(gpu_frq_lut, frq_lut, frq_lut_size, cudaMemcpyHostToDevice));

			size_t ROI_size = sizeof(int) * image->size.width;
			gpuErrchk(cudaMemcpy(gpu_start_ROI, start_ROI, ROI_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_end_ROI, end_ROI, ROI_size, cudaMemcpyHostToDevice));
		}



		/**
		 *	The function create the preprocessing data needed for the whole sum_and_fit process. Normaly, this should only be called
		 *	once per experiment, maybe even only once per setup : in that case, save the result somewhere and reload the arrays
		 *	instead of calculating everything again.
		 *	The arrays have the correct format to be given to <pipeline_send_experiment_settings> without any other processing.
		 *
		 *	 \param[in] cec Information relating to how to convert from the image to the summed curves.
		 *   \param[in] width Width of the image
		 *   \param[in] height Height of the image
		 *   \param[in] peak_numbers A 1D array containing the number of rayleigh peaks detected in this column. dimension : image.width
		 *   \param[in] peak_original_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The position is given in the pixel of the image, without linearisation.
		 *		dimension : image.width * cec.max_n_peaks
		 *   \param[in] peak_remapped_positions  A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The position is given in linearized space, at the wanted position.
		 *		dimension : image.width * cec.max_n_peaks
		 *  \param[out] translation_lut A 1D array to do the translation between column number and fitting number. Used to pack the data,
		 *		so that fits are only done on columns with real data.
		 *		-1 indicates we don't use the column, a positive value is the fitting number (fit #0, #1, ... , #(n_fits-1)).
		 *		 n_fits is appended at the end.
		 *		dimension : image.width + 1
		 *   \param[out] frq_lut  A 2D array where each value represents the frequency in linearized space of the corresponding pixel in the image.
		 *		dimension : image.width * image.height
		 *   \param[out] start_ROI A 1D array to indicate for each column at which pixel to start the sumation of the curves.
		 *		dimension : image.width
		 *   \param[out] end_ROI A 1D array to indicate for each column at which pixel to end the sumation of the curves.
		 *		dimension : image.width
		 *
		 * \returns
		 *   translation_lut, frq_lut, start_ROI, end_ROI : they have to be initialised to the correct sizes beforehand.
		 *
		 * \relates
		 * \ingroup preprocessing
		 *
		 */
		void create_preprocessing_data(Curve_Extraction_Context* cec, int width, int height, int* peak_numbers,
			float* peak_original_positions, float* peak_remapped_positions, int* translation_lut,
			float* frq_lut, int* start_ROI, int* end_ROI) {

			cudaExtent dim{ width, height, 1 };
			int* gpu_peak_numbers;
			float* gpu_original_peak_positions;
			gpuErrchk(cudaMalloc(&gpu_peak_numbers, sizeof(int) * dim.width));
			gpuErrchk(cudaMalloc(&gpu_original_peak_positions, sizeof(int) * dim.width * cec->max_n_peaks));
			gpuErrchk(cudaMemcpy(gpu_peak_numbers, peak_numbers, sizeof(int) * dim.width, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_original_peak_positions, peak_original_positions, sizeof(float) * dim.width * cec->max_n_peaks, cudaMemcpyHostToDevice));

			/* Remapping */
			float* a = new float[dim.width];
			float* b = new float[dim.width];
			float* c = new float[dim.width];
			float* gpu_a, * gpu_b, * gpu_c;
			gpuErrchk(cudaMalloc(&gpu_a, sizeof(float) * dim.width));
			gpuErrchk(cudaMalloc(&gpu_b, sizeof(float) * dim.width));
			gpuErrchk(cudaMalloc(&gpu_c, sizeof(float) * dim.width));

			Poly2Fitting poly2(dim.width, cec->max_n_peaks, 50, 1e-6);
			poly2.fit(peak_numbers, peak_original_positions, peak_remapped_positions);
			poly2.get_fitted_parameters(a, b, c);

			gpuErrchk(cudaMemcpy(gpu_a, a, sizeof(float) * dim.width, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_b, b, sizeof(float) * dim.width, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpu_c, c, sizeof(float) * dim.width, cudaMemcpyHostToDevice));


			/* Create frq_lut */
			float* gpu_frq_lut;
			gpuErrchk(cudaMalloc(&gpu_frq_lut, sizeof(float) * dim.width * dim.height));

			LaunchKernel::create_frq_lut_extrapolation(cec, dim, gpu_peak_numbers, gpu_original_peak_positions, gpu_a, gpu_b, gpu_c, gpu_frq_lut);

			gpuErrchk(cudaMemcpy(frq_lut, gpu_frq_lut, sizeof(float) * dim.width * dim.height, cudaMemcpyDeviceToHost));


			/* Create ROI */
			int* gpu_start_ROI, * gpu_end_ROI;
			gpuErrchk(cudaMalloc(&gpu_start_ROI, sizeof(int) * dim.width));
			gpuErrchk(cudaMalloc(&gpu_end_ROI, sizeof(int) * dim.width));
			LaunchKernel::create_ROI(cec, gpu_peak_numbers, gpu_original_peak_positions, dim.width, dim.height, gpu_a, gpu_b, gpu_c, cec->starting_order, cec->ending_order, gpu_start_ROI, gpu_end_ROI);
			gpuErrchk(cudaMemcpy(start_ROI, gpu_start_ROI, sizeof(int) * dim.width, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(end_ROI, gpu_end_ROI, sizeof(int) * dim.width, cudaMemcpyDeviceToHost));

			/* Create translation_lut */
			int n_fits = 0;
			for (int y = 0; y < dim.width; y++) {
				if (peak_numbers[y] >= 3) {
					translation_lut[y] = n_fits;
					n_fits++;
				}
				else {
					translation_lut[y] = -1;
				}
			}
			translation_lut[dim.width] = n_fits;

			/* Clean-up*/
			delete[] a;
			delete[] b;
			delete[] c;
			gpuErrchk(cudaFree(gpu_a));
			gpuErrchk(cudaFree(gpu_b));
			gpuErrchk(cudaFree(gpu_c));
			gpuErrchk(cudaFree(gpu_frq_lut));
			gpuErrchk(cudaFree(gpu_start_ROI));
			gpuErrchk(cudaFree(gpu_end_ROI));
			gpuErrchk(cudaFree(gpu_peak_numbers));
			gpuErrchk(cudaFree(gpu_original_peak_positions));

		}

		/**
		 *	This is only a *debug* function, used because it's a bother to do lowlevel analysis of array in LabView.
		 *	Given a binary image, this image finds the rayleigh peaks in each column, marks their position and also
		 *	calculate their position in the linearized space.
		 *	The arrays can be fed directly into the next function in the pipeline : <create_preprocessing_data>
		 *
		 *	This function hasn't been used in a while, might not work as expected. Use the matlab scripts or the
		 *  synthetic signal to test the pipeline.
		 *
		 *   \param image binary thresholded image of all the peaks. dimension : width * height
		 *   \param width Width of the image
		 *   \param height Height of the image
		 *   \param peak_numbers A 1D array containing the number of rayleigh peaks detected in this column. dimension : image.width
		 *   \param peak_original_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The position is given in the pixel of the image, without linearisation.
		 *		dimension : image.width * cec.max_n_peaks
		 *   \param peak_remapped_positions A 2D array containing the position in the image of the rayleigh peaks
		 *		detected in the columns. The position is given in linearized space, at the wanted position.
		 *		dimension : image.width * cec.max_n_peaks
		 *
		 * \returns
		 *
		 *
		 * \relates
		 * \ingroup preprocessing
		 *
		 */
		void find_peaks_debug(uint16_t* image, int width, int height, int* peak_numbers,
			float* peak_original_positions, float* peak_remapped_poisitions) {
			cudaExtent dim{ width, height, 1 };
			const int MAX_N_PEAKS = 20;
			GPU_Image gpu_thresholded;
			init_GPUImage(image, dim, &gpu_thresholded);
			int* gpu_peak_numbers;
			float* gpu_original_peak_positions;
			float* gpu_remapped_peak_positions;
			gpuErrchk(cudaMalloc(&gpu_peak_numbers, sizeof(int) * dim.width));
			gpuErrchk(cudaMalloc(&gpu_original_peak_positions, sizeof(float) * dim.width * MAX_N_PEAKS));
			gpuErrchk(cudaMalloc(&gpu_remapped_peak_positions, sizeof(float) * dim.width * MAX_N_PEAKS));

			/* Work on the GPU */
			dim3 block_dim(std::min((int)dim.width, 1024));
			dim3 grid_dim((dim.width + block_dim.x - 1) / block_dim.x);
			LaunchKernel::get_Rayleigh_peaks(curve_extraction_context, grid_dim, block_dim, gpu_thresholded.texture, dim, gpu_peak_numbers, gpu_original_peak_positions, gpu_remapped_peak_positions);

			gpuErrchk(cudaMemcpy(peak_numbers, gpu_peak_numbers, sizeof(int) * dim.width, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(peak_original_positions, gpu_original_peak_positions, sizeof(float) * dim.width * MAX_N_PEAKS, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(peak_remapped_poisitions, gpu_remapped_peak_positions, sizeof(float) * dim.width * MAX_N_PEAKS, cudaMemcpyDeviceToHost));


			/* Cleaning everything before starting the DLL_wrapper test*/
			gpuErrchk(cudaFree(gpu_peak_numbers));
			gpuErrchk(cudaFree(gpu_original_peak_positions));
			gpuErrchk(cudaFree(gpu_remapped_peak_positions));

			deinit_GPUImage(&gpu_thresholded);

		}
	}

	/**
	* Internal function of the library : used to create a PeakFitting object, which is the wrapper needed to call GpuFit.
	* This specific wrapper takes the range in frq, and expects the data to be extracted from the summed curve.
	*
	*/
	PeakFitting* create_PeakFitting(Fitting_Function_t function, float* range, Fitting_Context* gpufit_context,
		Stokes_antiStokes_Context* angle_context, Curve_Extraction_Context* cec) {
		PeakFitting* fitting_function;
		switch (function)
		{
		case FIT_LORENTZIAN:
			fitting_function = new LorentzianFitting(DEVICE, gpufit_context->n_fits, gpufit_context->max_iteration, gpufit_context->tolerance,
				range[0], range[1], gpufit_context->SNR_threshold, cec);
			break;
		case FIT_GAUSSIAN:
			fitting_function = new GaussianFitting(DEVICE, gpufit_context->n_fits, gpufit_context->max_iteration, gpufit_context->tolerance,
				range[0], range[1], gpufit_context->SNR_threshold, cec);

			break;
		case FIT_STOKES:
			fitting_function = new StokesFitting(DEVICE, gpufit_context->n_fits, gpufit_context->max_iteration, gpufit_context->tolerance,
				range[0], range[1], gpufit_context->SNR_threshold, cec, angle_context->NA_illum, angle_context->NA_coll, angle_context->angle,
				angle_context->angle_distribution_n, angle_context->geometrical_correction);
			break;
		case FIT_ANTISTOKES:
			fitting_function = new AntiStokesFitting(DEVICE, gpufit_context->n_fits, gpufit_context->max_iteration, gpufit_context->tolerance,
				range[0], range[1], gpufit_context->SNR_threshold, cec, angle_context->NA_illum, angle_context->NA_coll, angle_context->angle,
				angle_context->angle_distribution_n, angle_context->geometrical_correction);
			break;
		case FIT_PHASOR:
			//TODO
			break;
		default:
			break;
		}

		fitting_function->parameterContext->parameter_amplitude(1, NONE, 6000, 7000);
		fitting_function->parameterContext->parameter_shift(1, NONE, 0, 0);
		fitting_function->parameterContext->parameter_width(1, NONE, 0, 0);
		fitting_function->parameterContext->parameter_offset(1, NONE, 0, 0);
		fitting_function->parameterContext->set_use_constraints(false);

		return fitting_function;
	}


	/**
	* Internal function of the library : used to create a PeakFitting object, which is the wrapper needed to call GpuFit.
	* This specific wrapper takes the number of points, and does not need to do an extraction from the summed curve to the
	* (X,Y) data for the fit.
	*
	*/
	PeakFitting* create_PeakFitting_no_extraction(Fitting_Function_t function, int n_points, Fitting_Context* fit_co,
		Stokes_antiStokes_Context* angle_context) {
		PeakFitting* fitting;
		switch (function)
		{
		case FIT_LORENTZIAN:
			fitting = new LorentzianFitting(HOST, fit_co->n_fits, fit_co->max_iteration, fit_co->tolerance,
				n_points, fit_co->SNR_threshold);
			break;
		case FIT_GAUSSIAN:
			fitting = new GaussianFitting(HOST, fit_co->n_fits, fit_co->max_iteration, fit_co->tolerance,
				n_points, fit_co->SNR_threshold);

			break;
		case FIT_STOKES:
			fitting = new StokesFitting(HOST, fit_co->n_fits, fit_co->max_iteration, fit_co->tolerance,
				n_points, fit_co->SNR_threshold, angle_context->NA_illum, angle_context->NA_coll, angle_context->angle,
				angle_context->angle_distribution_n, angle_context->geometrical_correction);

			break;
		case FIT_ANTISTOKES:
			fitting = new AntiStokesFitting(HOST, fit_co->n_fits, fit_co->max_iteration, fit_co->tolerance,
				n_points, fit_co->SNR_threshold, angle_context->NA_illum, angle_context->NA_coll, angle_context->angle,
				angle_context->angle_distribution_n, angle_context->geometrical_correction);
			break;
		case FIT_PHASOR:
			//TODO
			break;
		default:
			break;
		}

		fitting->parameterContext->parameter_amplitude(1, NONE, 0, 0);
		fitting->parameterContext->parameter_shift(1, NONE, 0, 0);
		fitting->parameterContext->parameter_width(1, NONE, 0, 0);
		fitting->parameterContext->parameter_offset(1, NONE, 0, 0);
		fitting->parameterContext->set_use_constraints(false);

		return fitting;
	}

}


