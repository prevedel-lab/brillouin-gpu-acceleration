#include "FunctionFitting.h"

///////////////////////////////
/// Poly2 Fitting
/// ///////////////////////////

void Poly2Fitting::create_weights(int* peak_numbers){
	for (int x = 0; x < n_fits; x++) {
		for (int i = 0; i < n_points; i++) {
			weights[x * n_points + i] = (i < peak_numbers[x]) ? 1 : 0; //Only take into account the peaks we found
		}
	}
}

void Poly2Fitting::create_initial_parameters(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions){
	for (int x = 0; x < n_fits; x++) {

		float x_0 = original_peak_positions[x * n_points + 0];
		float x_1 = original_peak_positions[x * n_points + 1];
		float x_end = original_peak_positions[x * n_points + peak_numbers[x] - 1];
		float y_0 = remapped_peak_positions[x * n_points + 0];
		float y_1 = remapped_peak_positions[x * n_points + 1];
		float y_end = remapped_peak_positions[x * n_points + peak_numbers[x] - 1];

		initial_parameters[x * n_parameters + 0] = ((y_0 - y_1) / (x_0 - x_1) - (y_1 - y_end) / (x_1 - x_end)) / ((x_0 + x_1) /2 - (x_1 + x_end) / 2); //a
		initial_parameters[x * n_parameters + 1] = (y_0 - y_end) / (x_0 - x_end); //b 
		initial_parameters[x * n_parameters + 2] = (y_0 + y_end) / 2; //c
	}
}

void Poly2Fitting::set_parameter_context(){
	parameterContext->set_constraint(1, NONE, 0, 0, 0); //a
	parameterContext->set_constraint(1, NONE, 0, 0, 1); //b 
	parameterContext->set_constraint(1, NONE, 0, 0, 2); //c
}

void Poly2Fitting::create_user_info(float* data_X, float* data_Y, float* initial_parameters, float* weights){
	user_info_size = sizeof(float) * n_points * n_fits;
	float * user_info_float = (float*)malloc(user_info_size);
	for (int x = 0; x < n_fits; x++) {
		for (int i = 0; i < n_points; i++) {
			user_info_float[x * n_points + i] = data_X[x * n_points + i];
		}
	}

	user_info = (char*)user_info_float;
}

void Poly2Fitting::fit(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions){
	
	create_weights(peak_numbers);
	create_initial_parameters(peak_numbers, original_peak_positions, remapped_peak_positions);
	set_parameter_context();

	no_copy_fit(original_peak_positions, remapped_peak_positions, initial_parameters, weights);
	//explicit_fit_3orders(peak_numbers, original_peak_positions, remapped_peak_positions);
}


/*
Based on Carlo's Code from old Arduino data processing

solve the linear system Ax = b
where
	 sum x4   sum x3   sum x2
A =  sum x3   sum x2   sum x
	 sum x2   sum x      n

	 a2
x =  a1
	 a0

	 sum x2y
b =  sum xy
	 sum y
*/
void Poly2Fitting::explicit_fit_3orders(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions) {
	int n = n_fits;
	
	for (int i = 0; i < n; i++)
	{
		double xs = 0, x2s = 0, x3s = 0, x4s = 0, ys = 0, xys = 0, x2ys = 0;
		if (peak_numbers[i] == 3) {
			for (int j = 0; j < 3; j++) {
				//X value
				double x = original_peak_positions[i*n_points + j];
				xs += x;
				double x2 = x * x;
				x2s += x2;
				double x3 = x2 * x;
				x3s += x3;
				double x4 = x2 * x2;
				x4s += x4;

				//Y value
				double y = remapped_peak_positions[i * n_points + j];
				ys += y;
				double xy = y * x;
				xys += xy;
				double x2y = x * xy;
				x2ys += x2y;

				double det = x2s * (3 * x4s + 2 * xs * x3s - x2s * x2s) - xs * xs * x4s - 3 * x3s * x3s;

				float a2 = (3 * x2s * x2ys + xs * ys * x3s + xs * xys * x2s - (x2s * x2s * ys + xs * xs * x2ys + 3 * xys * x3s)) / det;
				float a1 = (x4s * xys * 3 + x2ys * xs * x2s + x2s * x3s * ys - (x2s * xys * x2s + xs * ys * x4s + 3 * x3s * x2ys)) / det;
				float a0 = (x4s * x2s * ys + x3s * xys * x2s + x2ys * x3s * xs - (x2s * x2s * x2ys + xs * xys * x4s + ys * x3s * x3s)) / det;
				
				fit_parameters[i * n_parameters + 0] = a2;
				fit_parameters[i * n_parameters + 1] = a1;
				fit_parameters[i * n_parameters + 2] = a0;
				output_n_iterations[i] = 0;
				output_states[i] = -1;
			
			}
		}
		else {
			fit_parameters[i * n_parameters + 0] = 0;
			fit_parameters[i * n_parameters + 1] = 0;
			fit_parameters[i * n_parameters + 2] = 0;
			output_n_iterations[i] = -1;
			output_states[i] = -1;
		}



	}





}

void Poly2Fitting::get_fitted_parameters(float* a, float* b, float* c){
	for (int x = 0; x < n_fits; x++) {
		a[x] = fit_parameters[x * n_parameters + 0];
		b[x] = fit_parameters[x * n_parameters + 1];
		c[x] = fit_parameters[x * n_parameters + 2];
	}
}

float Poly2Fitting::evaluate(float x, float a, float b, float c){
	return Functions::poly2(x, a, b, c);
}

void Poly2Fitting::save_to_file(int* peak_numbers, float* original_peak_positions, float* remapped_peak_positions, const char* file) {
    std::ofstream myfile;
    myfile.open(file);
    for (int i = 0; i < n_fits; i++) {
        myfile << "custom_X ";
        for (int d = 0; d < n_points; d++)
            myfile << (original_peak_positions + i * n_points)[d] << " ";
        myfile << "\ndata_Y ";
        for (int d = 0; d < n_points; d++)
            myfile << (remapped_peak_positions + i * n_points)[d] << " ";
        myfile << "\noutput_parameters ";
        for (int p = 0; p < n_parameters; p++)
            myfile << (fit_parameters + i * n_parameters)[p] << " ";
        myfile << "\nFitted_Y ";

        for (int d = 0; d < n_points; d++)
            myfile << evaluate((original_peak_positions + i * n_points)[d], (fit_parameters + i * 4)[0],
                (fit_parameters + i * 4)[1], (fit_parameters + i * 4)[2]) << " ";

        myfile << "\nOutput_states " << output_states[i] << " " << output_n_iterations[i] ;
        myfile << "\n\n";
    }
    myfile.close();
}

