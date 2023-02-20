/*****************************************************************//**
 * \file   GPUFit_calls.h
 * \brief  Class for peak fitting functions
 *
 * From the total 5 functions, only 4 are used to fit a peak
 *  - gaussian
 *  - lorentzian
 *  - broadened Brillouin lineshape (negative shift)
 *  - broadened Brillouin lineshape (positive shift)
 *
 * \author Sebastian
 * \date   August 2020
 *********************************************************************/

#pragma once
#include "GPUFit_calls.h"
#include "../cuda/kernel.cuh"

class PeakFitting : public GPUFit
{

protected:

	//Data range
	float start_freq, end_freq;
	int start_y, end_y;

	//Sanity check
	int* sanity;
	float SNR_threshold;
	float* SNR;

	//Fitting constraints
	float min_width = 0.2, max_width = 3; //constraints of the width
	float max_distance_to_maximum = 1; //constraint of the shift
	float min_amplitude_of_maximum = 0.0, max_amplitude_of_maximum = 0.90; //constraints of the amplitude (% of max amplitude)
	
	//Determining the fitting constraints
	float* maximum_position;
	float* maximum_amplitude;

	Curve_Extraction_Context * cec;

private:
	void initialisation();

public:
	~PeakFitting();	
	PeakFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, ModelID model_id, EstimatorID estimator_id, float start_freq, float end_freq, float SNR_threshold, Curve_Extraction_Context* cec) :
		cec(cec),
		start_freq(start_freq), end_freq(end_freq),
		start_y(round((start_freq - cec->starting_freq) / cec->step)), end_y(round((end_freq - cec->starting_freq) / cec->step)),
		SNR_threshold(SNR_threshold),
		GPUFit(location, n_fits, round((end_freq - cec->starting_freq) / cec->step) - round((start_freq - cec->starting_freq) / cec->step),
			max_iterations, tolerance, model_id, estimator_id) {
		initialisation();
	};

	PeakFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, ModelID model_id, EstimatorID estimator_id, int n_points, float SNR_threshold) :
		cec(cec),
		start_freq(0), end_freq(0),
		start_y(0), end_y(n_points),
		SNR_threshold(SNR_threshold),
		GPUFit(location, n_fits, n_points, max_iterations, tolerance, model_id, estimator_id) {
		initialisation();
	};

	// Function needed to compute a fitted curve
	virtual float evaluate(float x, float amplitude, float x0, float width, float offset) = 0;
	virtual void evaluate_batch_gpu(float* x, float* output_y, float* amplitude, float* shift, float* width, float* offset) = 0;
	void compute_fitted_curve(float* x_coord, Fitted_Function* fitted, float* fitted_y);


	// Debug function : usefull to save a fit to a file, for late analysis
	void save_to_file(float* data_X, float* data_Y, float* parameters, int* states, int* n_iterations, float* SNR, int* sanity, const char* file);
	void save_to_file(const char* file);

	//fit
	void fit() { no_copy_fit(x_coord, data, initial_parameters, NULL); };	
	
	//retrieve some parameters
	int* get_sanity() { return sanity; };
	float* get_SNR() { return SNR; };
	int get_start_y() { return start_y; };
	int get_end_y() { return end_y; };
	float get_start_freq() { return start_freq; };
	float get_end_freq() { return end_freq; };

	//Static methods | maybe useless ?
	static void extract_data_gpu(Curve_Extraction_Context* cec, float* summed_curves, size_t width, int* translation_lut,
		float start_frq, float end_frq, float* data_X, float* data_Y);
	static void get_initial_parameters_gpu(int n_fits, float* data_X, float* data_Y, int n_points, float* parameters, bool use_abs_value);

	//Instance method
	void get_initial_parameters(float* data_X, float* data_Y, float* parameters, DataLocation data_location, bool use_abs_value);
	void get_initial_parameters(bool use_abs_value) { get_initial_parameters(x_coord, data, initial_parameters, storage, use_abs_value); };
	virtual void get_initial_parameters() = 0;
	virtual void get_initial_parameters(float* data_X, float* data_Y, float* parameters, DataLocation data_location) = 0;
	
	// Extract the data from the summed curves, to prepare for fit
	void extract_data(float* summed_curves, size_t width, int* translation_lut, DataLocation source_storage);
	void extract_recentered_data( float* summed_curves,
		size_t width, int* translation_lut, DataLocation source_storage, PeakFitting* central_peak);
	
	// Perform a sanity check on the previous fit
	virtual void sanity_check(float* noise_level, float threshold, float* param) = 0;
	virtual void sanity_check(float* noise_level) = 0;
	
	// Export parameters
	void export_fitted_parameters(float* amplitude, float* center, float * width, float* offset);
	void export_sanity(int* sanity);

	// Fitting constraints functions
	bool use_fit_constraints() { return parameterContext->use_constraints; };
	void constraint_settings(bool use_constraints, float min_width, float max_width, float max_distance_to_maximum, 
		float min_amplitude_of_maximum, float max_amplitude_of_maximum);
	void determine_fitting_constraints(float* data_X, float* data_Y);
	void determine_fitting_constraints() { determine_fitting_constraints(x_coord, data); };	
	virtual void apply_fitting_constraints(float min_amp, float max_amp, float min_shift, float max_shift) = 0;



	
};

///////////////////////////////
/// Rayleigh (Gaussian or Lorentzian) Fitting
/// /////////////////////////// 
class RayleighFitting : public PeakFitting{
private:
	void create_user_info(float* data_X, float* data_Y, float* initial_parameters, float* weights);
public:
	RayleighFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, ModelID model_id, EstimatorID estimator_id, float start_freq, float end_freq, float SNR_threshold, Curve_Extraction_Context* cec) :
		PeakFitting(location, n_fits, max_iterations, tolerance, model_id, estimator_id, start_freq, end_freq, SNR_threshold, cec) {};
	RayleighFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, ModelID model_id, EstimatorID estimator_id, int n_points, float SNR_threshold) :
		PeakFitting(location, n_fits, max_iterations, tolerance, model_id, estimator_id, n_points, SNR_threshold) {};
	
	// Perform a sanity check for this specific function
	void sanity_check(float* noise_level, float threshold, float* param);
	void sanity_check(float* noise_level, float threshold) { sanity_check(noise_level, threshold, fit_parameters); };
	void sanity_check(float* noise_level) { sanity_check(noise_level, SNR_threshold, fit_parameters); };

	void apply_fitting_constraints(float min_amp, float max_amp, float min_shift, float max_shift);

	void evaluate_batch_gpu(float* x, float* output_y, float* amplitude, float* shift, float* width, float* offset) {
		LaunchKernel::batch_evaluation(model_id, x, output_y, n_fits, n_points, amplitude, shift, width, offset, NULL, 0);
	}

	//For Gaussian and Lorentzian, the shift can be negativ
	void get_initial_parameters() { PeakFitting::get_initial_parameters(false); };
	void get_initial_parameters(float* data_X, float* data_Y, float* parameters, DataLocation data_location)
	{
		PeakFitting::get_initial_parameters(data_X, data_Y, parameters, data_location, false);
	}
};


///////////////////////////////
/// Stokes or AntiStokes Fitting
/// /////////////////////////// 
class StokesOrAntiStokesFitting : public PeakFitting {
private : 
	void create_user_info(float* data_X, float* data_Y, float* initial_parameters, float* weights);
	void initialisation();

protected:
	int angle_distribution_n;
	float* angle_distribution;
	float* angle_distribution_cpu;
	float NA_illum, NA_coll, angle;
	float geometrical_correction;

public:
	StokesOrAntiStokesFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, ModelID model_id, EstimatorID estimator_id, float start_freq, float end_freq, float SNR_threshold, 
		Curve_Extraction_Context* cec, 
		float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction);
	StokesOrAntiStokesFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, ModelID model_id, EstimatorID estimator_id, int n_points, float SNR_threshold, 
		float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction);
	~StokesOrAntiStokesFitting();

	// Perform a sanity check for this group of functions
	void sanity_check(float* noise_level, float threshold, float* param);
	void sanity_check(float* noise_level, float threshold) { sanity_check(noise_level, threshold, fit_parameters); };
	void sanity_check(float* noise_level) { sanity_check(noise_level, SNR_threshold, fit_parameters); };

	//post-extraction recentering
	void dynamic_recenter(float* rayleigh_parameters, int* rayleigh_sanity); 

	// Create the angle distribution needed to compute a broadened Brillouin lineshape function
	static void init_angle_distribution(float NA_illum, float NA_coll, float angle, int N, float* angle_distrib);
	static double normsInv(double p, double mu, double sigma);
	static double normsInv_2(double p, double mu, double sigma);

	void apply_fitting_constraints(float min_amp, float max_amp, float min_shift, float max_shift);

	void evaluate_batch_gpu(float* x, float* output_y, float* amplitude, float* shift, float* width, float* offset);
	
	//Use absolute value because shift is measured to be positiv in stokes and antistokes
	void get_initial_parameters() { PeakFitting::get_initial_parameters(true); };
	void get_initial_parameters(float* data_X, float* data_Y, float* parameters, DataLocation data_location)
	{
		PeakFitting::get_initial_parameters(data_X, data_Y, parameters, data_location, true);
	}
};