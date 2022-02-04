/*****************************************************************//**
 * \file   GPUFit_calls.h
 * \brief  Gpufit wrapper
 *
 * The wrapper to call the different types of fitting function needed by the library.
 * There are 5 types of functions :
 *  - 2nd degree polynomial
 *  - gaussian
 *  - lorentzian
 *  - broadened Brillouin lineshape (negative shift)
 *  - broadened Brillouin lineshape (positive shift)
 * 
 * \author Sebastian
 * \date   August 2020
 *********************************************************************/

#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gpufit.h"
#include <random>


//#ifdef _DEBUG
	#include <iostream> //output in file
	#include <fstream>
	#include <sstream>
	#include <stdio.h>
//#endif	//_DEBUG
#define OUTPUT_FILE

#include "../cuda/errorHandling.cuh"
//#include "GPUFitting_constants.h"

// Class containing some information about the parameters and how they should be fitted
class ParametersContext {
private:
	bool cleanup;

public:
	int n_param = 0 ;
	ConstraintType* constraint_types = NULL;
	float* contraints = NULL;
	int* parameters_to_fit = NULL;
	bool use_constraints = false;


	ParametersContext() { cleanup = false; };
	ParametersContext(int n_param);
	~ParametersContext();

	void set_use_constraints(bool use) { use_constraints = use; };
	void set_constraint(int fit, ConstraintType type, float lower_bound, float upper_bound, int i);
	float get_constraint_lower(int index) {
		return contraints[2 * index + 0];
	}
	float get_constraint_upper(int index) {
		return contraints[2 * index + 1];
	}



	/* Renamed functions for better lisibility */
	void parameter_amplitude(int fit, ConstraintType type, float lower_bound, float upper_bound) 
	{ 
		set_constraint(fit, type, lower_bound, upper_bound, 0); 
	};
	void parameter_shift(int fit, ConstraintType type, float lower_bound, float upper_bound)
	{
		set_constraint(fit, type, lower_bound, upper_bound, 1);
	};
	void parameter_width(int fit, ConstraintType type, float lower_bound, float upper_bound)
	{
		set_constraint(fit, type, lower_bound, upper_bound, 2);
	};
	void parameter_offset(int fit, ConstraintType type, float lower_bound, float upper_bound)
	{
		set_constraint(fit, type, lower_bound, upper_bound, 3);
	};

};


// Wrapper class for GPUFit
class GPUFit {
public:
	//to know if the data is on Host (CPU) or Device (GPU) memory
	DataLocation storage; //from GPUfit.h

	//Necessary parameters to call gpufit
	size_t n_fits;
	size_t n_points;
	float* data;
	float* weights;
	ModelID model_id;
	float* initial_parameters;
	float tolerance;
	int max_iterations;
	EstimatorID estimator_id;
	size_t user_info_size;
	char* user_info;
	float* fit_parameters;
	int* output_states;
	float* output_chi_squares;
	int* output_n_iterations;

	//constrained fit
	ParametersContext* parameterContext;

	//Other usefull information
	int n_parameters;
	bool function_set;
	float* x_coord;
	bool no_weights;
	int result;
	
	
public:
	GPUFit();
	GPUFit(DataLocation location, size_t n_fits, size_t n_points, int max_iterations, 
		float tolerance, ModelID model_id, EstimatorID estimator_id) ;
	~GPUFit();

	/* to update the different arrays*/
	void update_stored_data(float* data_Y, float* initial_parameters, DataLocation arraySource);
	void update_stored_data(float* data_X, float* data_Y, float* initial_parameters, DataLocation arraySource);
	void update_weighs(float* weigths, DataLocation arraySource);
	
	/* Call the correct gpufit function */
	void fit() { no_copy_fit(x_coord, data, initial_parameters, (no_weights ? NULL : weights)); };
	void no_copy_fit(float* data_X, float* data_Y, float* initial_parameters, float* weights);
	
	virtual void create_user_info(float* data_X, float* data_Y, float* initial_parameters, float* weights) = 0;


public:
	void static copy_array(void* source_array, DataLocation source_location, void* dest_array, DataLocation dest_location, size_t size);
};