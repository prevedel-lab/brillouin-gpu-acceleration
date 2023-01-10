/*****************************************************************//**
 * \file   DLL_wrapper.h
 * \brief The wrapper around the code which acts as the library to import.
 * 
 * 
 * \author Sebastian
 * \date:  June 2020
 *********************************************************************/
#pragma once

//Only include this if the header is part of the .cpp file and to be compiled
//else we ignore this
#ifdef DLL_COMPILATION
	#include "cuda/kernel.cuh"
	#include "gpufit_wrapper/GPUFit_calls.h"
	#include "gpufit_wrapper/FunctionFitting.h"
	#include "other/CImg.h"
	#include "other/SyntheticSignal.h"
#endif // DLL_COMPILATION

#ifdef _WINDLL
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT 
#endif // DLL

#include "DLL_struct.h"
typedef unsigned short     uint16_t;

/** \defgroup pipeline The standard pipeline
*	\brief Minimal GPU-CPU memory transfer.
*	
*	First, call @ref DLL_wrapper::pipeline_initialisation "pipeline_initialisation" : This function allocates 
*	the correct amouns of memory on the GPU. \n
*	Next, you need to call @ref DLL_wrapper::pipeline_send_experiment_settings "pipeline_send_experiment_settings": 
*	this will send all the information relating to the Rayleigh peaks to the GPU.\n
*	You can then call @ref DLL_wrapper::pipeline_sum_and_fit "pipeline_sum_and_fit" as often as you want, 
*	and fit as many experiments as you wish. You may also call @ref DLL_wrapper::pipeline_get_gof "pipeline_get_gof" 
*	to get some goodness of fit metric on the previous fit done.\n
*	At the end of the experiment, call @ref DLL_wrapper::pipeline_close "pipeline_close" to clean up all the allocated memory, 
*	both on the GPU and CPU.\n
*
*
*	\defgroup standalone Standalone functions
*	\brief Some functions to do the same algorithm, but step-by-step : usefull for debugging.
*
*	\defgroup preprocessing Preprocessing pipeline
*/

#ifdef __cplusplus
namespace DLL_wrapper{
	extern "C" {
#endif



		/* Processing the data */

		DLL_EXPORT void linearize_pixelspace(Curve_Extraction_Context* cec, int width, int* peak_numbers, float* original_peak_positions,
			float* remapped_peak_positions, float* a, float* b, float* c);
		DLL_EXPORT void create_frq_lut(Curve_Extraction_Context* cec, float* frq_lut, int height, int width,
			int* peak_numbers, float* original_peak_positions, float* a, float* b, float* c);
		DLL_EXPORT void create_summed_curves(Curve_Extraction_Context* cec, uint16_t* raw_data, int height, int width, int* start_ROI, int* end_ROI,
			int* peak_numbers, float* lut_frq, float* dest);
		DLL_EXPORT void create_ROI(Curve_Extraction_Context* cec, int height, int width, int* peak_numbers, float* original_peak_positions, float* a,
			float* b, float* c, int* start_ROI, int* end_ROI);
		DLL_EXPORT void extract_data_for_fitting(Curve_Extraction_Context* cec, int width, float* summed_curves, int* translation_lut, float start_frq, float end_frq,
			int* n_points, float* data_X, float* data_Y);
		DLL_EXPORT void extract_data_for_fitting_recentering(Curve_Extraction_Context* cec, int width, float* summed_curves, int* translation_lut, float start_frq, float end_frq,
			int* n_points, float* data_X, float* data_Y, Fitted_Function fitted_rayleigh);
		DLL_EXPORT void estimate_noise_deviation(Curve_Extraction_Context* cec, int width, float* summed_curves,
			int* translation_lut,
			float stokes_start_frq, float antistokes_end_frq, float* noise_deviation);

		/* Fitting functions */

		DLL_EXPORT void fitting(float* data_X, float* data_Y, float* error_deviation, int n_points,
			Fitting_Function_t fitting_function,
			Fitting_Context* fit_co,
			Stokes_antiStokes_Context* angle_context,
			Fitted_Function* fitted_function);

		DLL_EXPORT void calculate_fitted_curve(float* data_X, int n_points, int n_fits,
			Fitting_Function_t function, Fitted_Function* fitted_function, Stokes_antiStokes_Context* angle_context, float* function_Y);


		/* The pipeline	*/

		DLL_EXPORT void pipeline_initialisation(int width, int height, Fitting_Context* gpufit_context, Curve_Extraction_Context* cec,
			Fitting_Function_t stokes_fitting_function, Fitting_Function_t rayleigh_fitting_function, Fitting_Function_t antistokes_fitting_function,
			float* stokes_range, float* rayleigh_range, float* antistokes_range, Stokes_antiStokes_Context* angle_context);
		DLL_EXPORT void pipeline_close();
		DLL_EXPORT void pipeline_sum_and_fit(uint16_t* cpu_raw_data, bool dynamic_recentering,
			Fitted_Function* stokes, Fitted_Function* rayleigh, Fitted_Function* antistokes);
        DLL_EXPORT void pipeline_sum_and_fit_to_array(
                    uint16_t* cpu_raw_data,
                    bool dynamic_recentering,
                    float* stokes,
                    float* rayleigh,
                    float* antistokes);
		DLL_EXPORT void pipeline_sum_and_fit_timed(uint16_t* cpu_raw_data, bool dynamic_recentering,
			Fitted_Function* stokes, Fitted_Function* rayleigh, Fitted_Function* antistokes, float* timings);
		DLL_EXPORT void pipeline_get_gof(Goodness_Of_Fit* stokes_gof, Goodness_Of_Fit* rayleigh_gof, Goodness_Of_Fit* antistokes_gof);
		DLL_EXPORT void pipeline_send_experiment_settings(int* peak_numbers, float* peak_original_positions, int* translation_lut,
			float* frq_lut, int* start_ROI, int* end_ROI);
		DLL_EXPORT void pipeline_set_constraints(bool use_constraints, float min_width, float max_width, float max_distance_to_maximum,
			float min_amplitude_of_maximum, float max_amplitude_of_maximum);


		/* Function to create data into the correct shape */

		DLL_EXPORT void create_preprocessing_data(Curve_Extraction_Context* cec, int width, int height, int* peak_numbers,
			float* peak_original_positions, float* peak_remapped_positions, int* translation_lut, float* frq_lut, int* start_ROI, int* end_ROI);
		DLL_EXPORT void find_peaks_debug(uint16_t* image, int width, int height, int* peak_numbers, float* peak_original_positions,
			float* peak_remapped_poisitions);


#ifdef __cplusplus
	}

	/* Private functions*/
	PeakFitting* create_PeakFitting(Fitting_Function_t function, float* range, Fitting_Context* gpufit_context,
		Stokes_antiStokes_Context* angle_context, Curve_Extraction_Context* cec);
	PeakFitting* create_PeakFitting_no_extraction(Fitting_Function_t function, int n_points, Fitting_Context* gpufit_context,
		Stokes_antiStokes_Context* angle_context);
};
#endif

