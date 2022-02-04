/*****************************************************************//**
 * \file   FunctionFitting.h
 * \brief  Specification of the different fitting functions
 *
 * The last level of differentiation of the fitting funtions
 *
 * \author Sebastian
 * \date   August 2020
 *********************************************************************/

#pragma once
#include "PeakFitting.h"

///////////////////////////////
/// Gaussian Fitting
///////////////////////////////
class GaussianFitting : public RayleighFitting {
public:
	GaussianFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, float start_freq, float end_freq, float SNR_threshold, Curve_Extraction_Context* cec) :
		RayleighFitting(location, n_fits, max_iterations, tolerance, GAUSS_1D, LSE, start_freq, end_freq, SNR_threshold, cec) {};
	GaussianFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, int n_points, float SNR_threshold) :
		RayleighFitting(location, n_fits, max_iterations, tolerance, GAUSS_1D, LSE, n_points, SNR_threshold) {};

	float evaluate(float x, float amplitude, float x0, float width, float offset) {
		return Functions::gaussian(x, amplitude, x0, width) + offset;
	};
};

///////////////////////////////
/// Lorentzian Fitting
/// ///////////////////////////
class LorentzianFitting : public RayleighFitting {

public:
	LorentzianFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, float start_freq, float end_freq, float SNR_threshold, Curve_Extraction_Context* cec) :
		RayleighFitting(location, n_fits, max_iterations, tolerance, CAUCHY_LORENTZ_1D, LSE, start_freq, end_freq, SNR_threshold, cec) {};
	LorentzianFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, int n_points, float SNR_threshold) :
		RayleighFitting(location, n_fits, max_iterations, tolerance, CAUCHY_LORENTZ_1D, LSE, n_points, SNR_threshold) {};
	float evaluate(float x, float amplitude, float x0, float width, float offset) {
		return Functions::lorentzian(x, amplitude, x0, width) + offset;
	};

};

///////////////////////////////
/// Stokes Fitting
///////////////////////////////
class StokesFitting : public StokesOrAntiStokesFitting {


public:
	StokesFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, float start_freq, float end_freq, float SNR_threshold, Curve_Extraction_Context* cec,
		float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction) :
		StokesOrAntiStokesFitting(location, n_fits, max_iterations, tolerance, STOKES, LSE, start_freq, end_freq, SNR_threshold, cec, 
			NA_illum, NA_coll, angle, angle_distribution_n, geometrical_correction) {};
	StokesFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, int n_points, float SNR_threshold,
		float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction) :
		StokesOrAntiStokesFitting(location, n_fits, max_iterations, tolerance, STOKES, LSE, n_points, SNR_threshold,
			NA_illum, NA_coll, angle, angle_distribution_n, geometrical_correction) {};

	float evaluate(float x, float amplitude, float x0, float width, float offset) {
		return Functions::stokes(x, amplitude, x0, width, angle_distribution_cpu, angle_distribution_n) + offset;
	};

};

///////////////////////////////
/// AntiStokes Fitting
///////////////////////////////
class AntiStokesFitting : public StokesOrAntiStokesFitting {

public:
	AntiStokesFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, float start_freq, float end_freq, float SNR_threshold, Curve_Extraction_Context* cec,
		float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction) :
		StokesOrAntiStokesFitting(location, n_fits, max_iterations, tolerance, ANTI_STOKES, LSE, start_freq, end_freq, SNR_threshold, cec,
			NA_illum, NA_coll, angle, angle_distribution_n, geometrical_correction) {};
	AntiStokesFitting(DataLocation location, size_t n_fits, int max_iterations,
		float tolerance, int n_points, float SNR_threshold,
		float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction) :
		StokesOrAntiStokesFitting(location, n_fits, max_iterations, tolerance, ANTI_STOKES, LSE, n_points, SNR_threshold,
			NA_illum, NA_coll, angle, angle_distribution_n, geometrical_correction) {};

	float evaluate(float x, float amplitude, float x0, float width, float offset) {
		return Functions::anti_stokes(x, amplitude, x0, width, angle_distribution_cpu, angle_distribution_n) + offset;
	};
};

///////////////////////////////
/// Poly2 Fitting
///////////////////////////////
class Poly2Fitting : public GPUFit {

	void create_weights(int* peak_numbers);
	void create_initial_parameters(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions);
	void set_parameter_context();
	void create_user_info(float* data_X, float* data_Y, float* initial_parameters, float* weights);

public:
	Poly2Fitting(int n_fits, int n_points, int max_iterations, float tolerance) :
		GPUFit(HOST, n_fits, n_points, max_iterations, tolerance, POLY2, LSE) {};

	void fit(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions);
	void explicit_fit_3orders(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions);
	void get_fitted_parameters(float* a, float* b, float* c);
	float evaluate(float x, float a, float b, float c);
	void save_to_file(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions, const char* file);

};