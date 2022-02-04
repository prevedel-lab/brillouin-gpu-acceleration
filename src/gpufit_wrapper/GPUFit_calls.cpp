#include "GPUFit_calls.h"


ParametersContext::ParametersContext(int n_param) {
	cleanup = true;
	this->n_param = n_param;
	constraint_types = new ConstraintType[n_param];
	contraints = new float[2 * n_param];
	parameters_to_fit = new int[n_param];
}

ParametersContext::~ParametersContext() {
	if (cleanup) {
		delete[] constraint_types;
		delete[] contraints;
		delete[] parameters_to_fit;
	}

}

void ParametersContext::set_constraint(int fit, ConstraintType type, float lower_bound, float upper_bound, int i) {
	parameters_to_fit[i] = fit;
	constraint_types[i] = type;
	contraints[2 * i + 0] = lower_bound;
	contraints[2 * i + 1] = upper_bound;
}

GPUFit::GPUFit() {

}

GPUFit::GPUFit(DataLocation location, size_t n_fits, size_t n_points, int max_iterations,
	float tolerance, ModelID model_id, EstimatorID estimator_id) {
	
	storage = location;

	//Get the parameters
	this->n_fits = n_fits;
	this->n_points = n_points; 
	this->max_iterations = max_iterations;
	this->tolerance = tolerance;
	this->model_id = model_id;
	this->estimator_id = estimator_id; 
	
	switch (model_id)
	{
	case GAUSS_1D:
		n_parameters = 4;
		break;
	case ANTI_STOKES:
		n_parameters = 4;
		break;
	case STOKES:
		n_parameters = 4;
		break;
	case POLY2:
		n_parameters = 3;
		break;
	case CAUCHY_LORENTZ_1D:
		n_parameters = 4;
		break;
	default:
		break;
	}

	//Allocate the memory

	//Always on CPU memory
	//parameters_to_fit = new int[n_parameters];
	parameterContext = new ParametersContext(n_parameters);

	switch (location) {
	case HOST:

		data = new float[n_points * n_fits];
		weights = new float[n_points * n_fits];
		initial_parameters = new float[n_parameters * n_fits];
		output_states = new int[n_fits];
		output_chi_squares = new float[n_fits];
		output_n_iterations = new int[n_fits];
		x_coord = new float[n_points * n_fits];
		fit_parameters = new float[n_parameters * n_fits];


		break;
	case DEVICE:

		gpuErrchk(cudaMalloc(&data, sizeof(float) * n_points * n_fits));
		gpuErrchk(cudaMalloc(&weights, sizeof(float) * n_points * n_fits));
		gpuErrchk(cudaMalloc(&initial_parameters, sizeof(float) * n_parameters * n_fits));
		gpuErrchk(cudaMalloc(&output_states, sizeof(int) * n_fits));
		gpuErrchk(cudaMalloc(&output_chi_squares, sizeof(float) * n_fits));
		gpuErrchk(cudaMalloc(&output_n_iterations, sizeof(int) * n_fits));
		gpuErrchk(cudaMalloc(&x_coord, sizeof(int) * n_points * n_fits));
		//gpuErrchk(cudaMalloc(&fit_parameters, sizeof(float) * n_parameters * n_fits));
		fit_parameters = initial_parameters;

		break;
	default:
		break;
	}

}

GPUFit::~GPUFit() {
	delete parameterContext;
	switch(storage) {
	case HOST:

		delete[] data;
		delete[] weights;
		delete[] initial_parameters;
		delete[] output_states;
		delete[] output_chi_squares;
		delete[] output_n_iterations;
		delete[] x_coord;
		delete[] fit_parameters;


		break;
	case DEVICE:

		gpuErrchk(cudaFree(data));
		gpuErrchk(cudaFree(weights));
		gpuErrchk(cudaFree(initial_parameters));
		//No separate initial_parameters array in the cuda_interface;
		gpuErrchk(cudaFree(output_states));
		gpuErrchk(cudaFree(output_chi_squares));
		gpuErrchk(cudaFree(output_n_iterations));
		gpuErrchk(cudaFree(x_coord));
		//gpuErrchk(cudaFree(fit_parameters));
		//gpuErrchk(cudaFree(constraints));
		
		break;
	default:
		break;
	}
}

void GPUFit::copy_array(void* source_array, DataLocation source_location, void* dest_array, DataLocation dest_location, size_t size) {
	switch (source_location)
	{
	case HOST:
		switch (dest_location)
		{
		case HOST: //Host -> Host
			memcpy(dest_array, source_array, size);
			break;
		case DEVICE: //Host -> Device
			gpuErrchk(cudaMemcpy(dest_array, source_array, size, cudaMemcpyHostToDevice));
			break;
		default:
			break;
		}
		break;
	case DEVICE:		
		switch (dest_location)
		{
		case HOST: //Device -> Host
			gpuErrchk(cudaMemcpy(dest_array, source_array, size, cudaMemcpyDeviceToHost));
			break;
		case DEVICE: //Device -> Device
			gpuErrchk(cudaMemcpy(dest_array, source_array, size, cudaMemcpyDeviceToDevice));
			break;
		default:
			break;
		}
			break;
	default:
		break;
	}
}

void GPUFit::update_stored_data(float* data_Y, float* initial_parameters, DataLocation arraySource) {
	copy_array(data_Y, arraySource, data, storage, sizeof(float) * n_points * n_fits);
	copy_array(initial_parameters, arraySource, this->initial_parameters, storage, sizeof(float) * n_parameters * n_fits);
	
}

void GPUFit::update_stored_data(float* data_X, float* data_Y, float* initial_parameters, DataLocation arraySource) {
	copy_array(data_Y, arraySource, data, storage, sizeof(float) * n_points * n_fits);
	copy_array(data_X, arraySource, x_coord, storage, sizeof(float) * n_points);
	copy_array(initial_parameters, arraySource, this->initial_parameters, storage, sizeof(float) * n_parameters * n_fits);
}

void GPUFit::update_weighs(float* weigths, DataLocation arraySource) {
	if (weigths == NULL) {
		no_weights = true;
	}
	else {
		copy_array(weigths, arraySource, this->weights, storage, sizeof(float) * n_points * n_fits);
	}
}


//data_X is expected to have the same dimension as data_Y 
// ==> n_points * n_fits : 1 x-scale per fit
void GPUFit::no_copy_fit(float* data_X, float* data_Y, float* initial_parameters, float* weights) {
	create_user_info(data_X, data_Y, initial_parameters, weights);
	switch (storage)
	{
	case HOST:
		result = gpufit_constrained(n_fits, n_points, data_Y, weights, model_id, initial_parameters,
			(!parameterContext->use_constraints) ? NULL : parameterContext->contraints,
			(!parameterContext->use_constraints) ? NULL : (int*)parameterContext->constraint_types,
			tolerance, max_iterations, parameterContext->parameters_to_fit,
			estimator_id, user_info_size, user_info, fit_parameters, output_states, output_chi_squares, output_n_iterations);

		free(user_info);
		break;


	case DEVICE:
		
		result = gpufit_constrained_cuda_interface(n_fits, n_points, data_Y, weights, model_id, tolerance, max_iterations,
			parameterContext->parameters_to_fit,
			(!parameterContext->use_constraints) ? NULL : parameterContext->contraints,
			(!parameterContext->use_constraints) ? NULL : (int*)parameterContext->constraint_types,
			estimator_id, user_info_size, user_info, initial_parameters, output_states, output_chi_squares, output_n_iterations);

		gpuErrchk(cudaFree(user_info));
		break;
	default:
		break;
	}
}