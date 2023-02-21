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

#include "PeakFitting.h"

///////////////////////////////
/// Peak Fitting
///////////////////////////////

void PeakFitting::initialisation(){
    switch (storage)
    {
    case HOST:
        sanity = new int[n_fits];
        SNR = new float[n_fits];
        maximum_position = new float[n_fits];
        maximum_amplitude = new float[n_fits];
        break;
    case DEVICE:
        gpuErrchk(cudaMalloc(&sanity, sizeof(int) * n_fits));
        gpuErrchk(cudaMalloc(&SNR, sizeof(float) * n_fits));
        gpuErrchk(cudaMalloc(&maximum_position, sizeof(float) * n_fits));
        gpuErrchk(cudaMalloc(&maximum_amplitude, sizeof(float) * n_fits));

        break;
    default:
        break;
    }
    
}

PeakFitting::~PeakFitting(){
    switch (storage)
    {
    case HOST:
        delete[] sanity;
        delete[] SNR;
        delete[] maximum_position;
        delete[] maximum_amplitude;
        break;
    case DEVICE:
        gpuErrchk(cudaFree(sanity));
        gpuErrchk(cudaFree(SNR));
        gpuErrchk(cudaFree(maximum_position));
        gpuErrchk(cudaFree(maximum_amplitude));
        break;
    default:
        break;
    }
}

/* Extracts the data from a certain frq range from the summed curve 
*  GPU version
*  Depraciated - TODO remove ?
*/
void PeakFitting::extract_data_gpu(Curve_Extraction_Context* cec, float* summed_curves, size_t width, int* translation_lut,
 float start_frq, float end_frq, float* data_X, float* data_Y) {
    int start_y = round((start_frq - cec->starting_freq) / cec->step);
    int end_y = round((end_frq - cec->starting_freq) / cec->step);
    LaunchKernel::extract_fitting_data(cec, summed_curves, width, translation_lut, start_y, end_y, data_X, data_Y);
};

/* Extracts the data from a certain frq range from the summed curve 
*  Data source can be Host or Device
*/
void PeakFitting::extract_data(float* summed_curves, size_t width, int* translation_lut, DataLocation source_storage){
    float* gpu_summed_curves;
    int* gpu_translation_lut;
    float* gpu_data_X, *gpu_data_Y;
    
    switch (source_storage)
    {
    case HOST:
        gpuErrchk(cudaMalloc(&gpu_summed_curves, sizeof(float) * width * cec->n_points));
        gpuErrchk(cudaMalloc(&gpu_translation_lut, sizeof(int) * (width + 1)));
        gpuErrchk(cudaMemcpy(gpu_summed_curves, summed_curves, sizeof(float) * width * cec->n_points, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_translation_lut, translation_lut, sizeof(int) * (width + 1), cudaMemcpyHostToDevice));
        break;
    case DEVICE:
        gpu_summed_curves = summed_curves;
        gpu_translation_lut = translation_lut;
        break;
    default:
        break;
    }

    switch (storage)
    {
    case HOST:
        gpuErrchk(cudaMalloc(&gpu_data_X, sizeof(float) * n_fits * n_points));
        gpuErrchk(cudaMalloc(&gpu_data_Y, sizeof(float) * n_fits * n_points));
        break;
    case DEVICE:
        gpu_data_X = x_coord;
        gpu_data_Y = data;
        break;
    default:
        break;
    }

    //extract_data_gpu(gpu_summed_curves, width, gpu_translation_lut, start_frq, end_frq, gpu_data_X, gpu_data_Y);
    LaunchKernel::extract_fitting_data(cec, gpu_summed_curves, width, gpu_translation_lut, start_y, end_y, gpu_data_X, gpu_data_Y);

    if (source_storage == HOST) {
        gpuErrchk(cudaFree(gpu_summed_curves));
        gpuErrchk(cudaFree(gpu_translation_lut));
    }

    if(storage == HOST){
        gpuErrchk(cudaMemcpy(x_coord, gpu_data_X, sizeof(float) * n_fits * n_points, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(data, gpu_data_Y, sizeof(float) * n_fits * n_points, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(gpu_data_X));
        gpuErrchk(cudaFree(gpu_data_Y));
    }


}

/* Extracts the data from a certain frq range from the summed curve, and does recentering.
*  Data source can be Host or Device.
*  Untested if central_peak is on host memory
*/
void PeakFitting::extract_recentered_data(float* summed_curves, size_t width, int* translation_lut, DataLocation source_storage, PeakFitting* central_peak)
{
    float* gpu_summed_curves;
    int* gpu_translation_lut;
    float* gpu_data_X, * gpu_data_Y;
    float* gpu_rayleigh_fit;
    int* gpu_rayleigh_sanity;

    switch (source_storage)
    {
    case HOST:
        gpuErrchk(cudaMalloc(&gpu_summed_curves, sizeof(float) * width * cec->n_points));
        gpuErrchk(cudaMalloc(&gpu_translation_lut, sizeof(int) * (width + 1)));
        gpuErrchk(cudaMemcpy(gpu_summed_curves, summed_curves, sizeof(float) * width * cec->n_points, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_translation_lut, translation_lut, sizeof(int) * (width + 1), cudaMemcpyHostToDevice));
        break;
    case DEVICE:
        gpu_summed_curves = summed_curves;
        gpu_translation_lut = translation_lut;
        break;
    default:
        break;
    }

    switch (storage)
    {
    case HOST:
        gpuErrchk(cudaMalloc(&gpu_data_X, sizeof(float) * n_fits * n_points));
        gpuErrchk(cudaMalloc(&gpu_data_Y, sizeof(float) * n_fits * n_points));
        break;
    case DEVICE:
        gpu_data_X = x_coord;
        gpu_data_Y = data;
        break;
    default:
        break;
    }

    switch (central_peak->storage)
    {
    case HOST:
        //TODO : untested for now 
        gpuErrchk(cudaMalloc(&gpu_rayleigh_fit, sizeof(float) * central_peak->n_fits * central_peak->n_parameters));
        gpuErrchk(cudaMemcpy(gpu_rayleigh_fit, central_peak->fit_parameters, sizeof(float) * central_peak->n_fits * central_peak->n_parameters, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc(&gpu_rayleigh_sanity, sizeof(int) * central_peak->n_fits * 1));
        gpuErrchk(cudaMemcpy(gpu_rayleigh_sanity, central_peak->get_sanity(), sizeof(int) * central_peak->n_fits * 1, cudaMemcpyHostToDevice));

        break;
    case DEVICE:
        gpu_rayleigh_fit = central_peak->fit_parameters;
        gpu_rayleigh_sanity = central_peak->get_sanity();
        break;
    default:
        break;
    }

    //extract_data_gpu(gpu_summed_curves, width, gpu_translation_lut, start_frq, end_frq, gpu_data_X, gpu_data_Y);
    LaunchKernel::extract_fitting_data_dynamic_recentering(cec, gpu_summed_curves, width, 
        gpu_translation_lut, start_y, end_y, gpu_data_X, gpu_data_Y, gpu_rayleigh_fit, gpu_rayleigh_sanity);

    if (source_storage == HOST) {
        gpuErrchk(cudaFree(gpu_summed_curves));
        gpuErrchk(cudaFree(gpu_translation_lut));
    }

    if (storage == HOST) {
        gpuErrchk(cudaMemcpy(x_coord, gpu_data_X, sizeof(float) * n_fits * n_points, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(data, gpu_data_Y, sizeof(float) * n_fits * n_points, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(gpu_data_X));
        gpuErrchk(cudaFree(gpu_data_Y));
    }

    if (central_peak->storage == HOST) {
        gpuErrchk(cudaFree(gpu_rayleigh_fit));
        gpuErrchk(cudaFree(gpu_rayleigh_sanity));
    }




}

/* Determines initial parameter given the (X,Y) data. input can be either Host or Device memory.
*/
void PeakFitting::get_initial_parameters(float* data_X, float* data_Y, float* parameters, DataLocation data_location, bool use_abs_value = true){
    float* gpu_data_x, * gpu_data_y, *gpu_parameters;
    switch (data_location)
    {
    case HOST:
        gpuErrchk(cudaMalloc(&gpu_data_x, sizeof(float) * n_fits * n_points));
        gpuErrchk(cudaMalloc(&gpu_data_y, sizeof(float) * n_fits * n_points));
        gpuErrchk(cudaMalloc(&gpu_parameters, sizeof(float) * n_fits * n_parameters));
        gpuErrchk(cudaMemcpy(gpu_data_x, data_X, sizeof(float) * n_fits * n_points, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_data_y, data_Y, sizeof(float) * n_fits * n_points, cudaMemcpyHostToDevice));
        break;
    case DEVICE:
        gpu_data_x = data_X;
        gpu_data_y = data_Y;
        gpu_parameters = parameters;
        break;
    default:
        break;
    }

    get_initial_parameters_gpu(n_fits, gpu_data_x, gpu_data_y, n_points, gpu_parameters, use_abs_value);

    if(data_location == HOST){
        gpuErrchk(cudaMemcpy(parameters, gpu_parameters, sizeof(float) * n_fits * n_parameters, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(gpu_data_x));
        gpuErrchk(cudaFree(gpu_data_y));
        gpuErrchk(cudaFree(gpu_parameters));
    }


}

/* Determines initial parameter given the (X,Y) data.
*  GPU version : everything is on the Device memory.
*/
void PeakFitting::get_initial_parameters_gpu(int n_fits, float* data_X, float* data_Y, int n_points, float* parameters, bool use_abs_value){
    LaunchKernel::get_initial_parameters(n_fits, data_X, data_Y, n_points, parameters, use_abs_value);
}

/* Saves the parameter into the specified file
*/
void PeakFitting::save_to_file(float* data_X, float* data_Y, float* parameters, int* states, int* n_iterations, float* SNR, int* sanity, const char* file) {
    std::ofstream myfile;
    myfile.open(file);
    for (int i = 0; i < n_fits; i++) {
        myfile << "# " << i;
        myfile << "\ncustom_X ";
        for (int d = 0; d < n_points; d++)
            myfile << (data_X + i * n_points)[d] << " ";
        myfile << "\ndata_Y ";
        for (int d = 0; d < n_points; d++)
            myfile << (data_Y + i * n_points)[d] << " ";
        myfile << "\nFitted_Y ";

        for (int d = 0; d < n_points; d++)
            myfile << evaluate((data_X + i * n_points)[d], (parameters + i * 4)[0],
                (parameters + i * 4)[1], (parameters + i * 4)[2], (parameters + i * 4)[3])  
            << " ";

        myfile << "\nAmplitude " << (parameters + i * n_parameters)[0] << " " 
            << parameterContext->get_constraint_lower(0) << " " << parameterContext->get_constraint_upper(0);
        myfile << "\nShift " << (parameters + i * n_parameters)[1] << " "
            << parameterContext->get_constraint_lower(1) << " " << parameterContext->get_constraint_upper(1);
        myfile << "\nWidth " << (parameters + i * n_parameters)[2] << " "
            << parameterContext->get_constraint_lower(2) << " " << parameterContext->get_constraint_upper(2);
        myfile << "\nOffset " << (parameters + i * n_parameters)[3] << " "
            << parameterContext->get_constraint_lower(3) << " " << parameterContext->get_constraint_upper(3);
        myfile << "\nOutput_states " << states[i] << " " << n_iterations[i];
        myfile << "\nSanity " << ((sanity == NULL) ? 0 : SNR[i] )<< " "<< ((sanity == NULL) ? " " : (sanity[i] ? "1" : "0"));
        myfile << "\n\n";
    }
    myfile.close();
}

/* Saves the parameter from the previous fit into a file. 
*  If it's stored on GPU memory, does the necessary copying
*/
void PeakFitting::save_to_file(const char* file) {
    float* data_x;
    float* data_y;
    float* parameters;
    int* states;
    int* n_iterations;
    int* sanity_cpu;
    float* SNR_cpu;

    switch (storage)
    {
    case HOST:
        save_to_file(x_coord, data, fit_parameters, output_states, output_n_iterations, SNR, sanity, file);
        break;
    case DEVICE:
        data_x = new float[n_fits * n_points];
        data_y = new float[n_fits * n_points];
        parameters = new float[n_parameters * n_fits];
        states = new int[n_fits];
        n_iterations = new int[n_fits];
        sanity_cpu = new int[n_fits];
        SNR_cpu = new float[n_fits];
        copy_array(x_coord, DEVICE, data_x, HOST, sizeof(float) * n_fits * n_points);
        copy_array(data, DEVICE, data_y, HOST, sizeof(float) * n_fits * n_points);
        copy_array(fit_parameters, DEVICE, parameters, HOST, sizeof(float) * n_fits * 4);
        copy_array(output_states, DEVICE, states, HOST, sizeof(int) * n_fits);
        copy_array(output_n_iterations, DEVICE, n_iterations, HOST, sizeof(int) * n_fits);
        copy_array(sanity, DEVICE, sanity_cpu, HOST, sizeof(int) * n_fits);
        copy_array(SNR, DEVICE, SNR_cpu, HOST, sizeof(float) * n_fits);

        save_to_file(data_x, data_y, parameters, states, n_iterations, SNR_cpu, sanity_cpu, file);

        delete[] data_x;
        delete[] data_y;
        delete[] parameters;
        delete[] states;
        delete[] n_iterations;
        delete[] sanity_cpu;
        delete[] SNR_cpu;
        break;
    default:
        break;
    }
}

/* Retrieves the fitted parameter, and split the parameters into 4 independent arrays.*
*/
void PeakFitting::export_fitted_parameters(float* amplitude, float* center, float* width, float* offset){
    //TODO : add a Device-side method to copy the parameters into the right shape
    if(storage == DEVICE){
        fit_parameters = new float[n_fits * n_parameters];
        gpuErrchk(cudaMemcpy(fit_parameters, initial_parameters, sizeof(float) * n_fits * n_parameters, cudaMemcpyDeviceToHost));
    }
    
    for (int y = 0; y < n_fits; y++) {
        amplitude[y] = fit_parameters[y * 4 + 0];
        center[y] = fit_parameters[y * 4 + 1];
        width[y] = fit_parameters[y * 4 + 2];
        offset[y] = fit_parameters[y * 4 + 3];
    }

    if(storage == DEVICE){
        delete[] fit_parameters;
        fit_parameters = initial_parameters;
    }

}

/* Retrieves parameter sanity from the fit
*/
void PeakFitting::export_sanity(int* sanity)
{
    if (storage == DEVICE) {
        gpuErrchk(cudaMemcpy(sanity, this->sanity, sizeof(int) * n_fits, cudaMemcpyDeviceToHost));
    } else {
        memcpy(sanity, this->sanity, sizeof(int) * n_fits);
    }

}

/* Apply the given constraints for the following fits
*/
void PeakFitting::constraint_settings(bool use_constraints, float min_width, float max_width, float max_distance_to_maximum, float min_amplitude_of_maximum, float max_amplitude_of_maximum)
{
    this->min_width = min_width;
    this->max_width = max_width;
    this->max_distance_to_maximum = max_distance_to_maximum;
    this->max_amplitude_of_maximum = max_amplitude_of_maximum;
    this->min_amplitude_of_maximum = min_amplitude_of_maximum;
    parameterContext->set_use_constraints(use_constraints);

}

/* Determine the fit constraints based on the (X,Y) input data and some general rules.
* Input data is expected to be on the same side as the storage of this PeakFitting
*/
void PeakFitting::determine_fitting_constraints(float* data_X, float* data_Y){
    //Get position of maximum for each fit
    float* gpu_data_X, * gpu_data_Y;
    float* gpu_maximum_pos, * gpu_maximum_amp;  
    float* cpu_maximum_pos, * cpu_maximum_amp;  
    
    switch (storage)
    {
    case HOST:
        
        gpuErrchk(cudaMalloc(&gpu_data_X, sizeof(float) * n_fits * n_points));
        gpuErrchk(cudaMalloc(&gpu_data_Y, sizeof(float) * n_fits * n_points));
        gpuErrchk(cudaMalloc(&gpu_maximum_pos, sizeof(float) * n_fits ));
        gpuErrchk(cudaMalloc(&gpu_maximum_amp, sizeof(float) * n_fits ));

        gpuErrchk(cudaMemcpy(gpu_data_X, data_X, sizeof(float) * n_fits * n_points, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_data_Y, data_Y, sizeof(float) * n_fits * n_points, cudaMemcpyHostToDevice));

        LaunchKernel::find_maxima(n_fits, n_points , gpu_data_X, gpu_data_Y, gpu_maximum_pos, gpu_maximum_amp);
        gpuErrchk(cudaMemcpy(maximum_position, gpu_maximum_pos, sizeof(float) * n_fits , cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(maximum_amplitude, gpu_maximum_amp, sizeof(float) * n_fits , cudaMemcpyDeviceToHost));
        cpu_maximum_pos = maximum_position;
        cpu_maximum_amp = maximum_amplitude;
        break;
    case DEVICE:
        LaunchKernel::find_maxima(n_fits, n_points, data_X, data_Y, maximum_position, maximum_amplitude);
        cpu_maximum_pos = new float[n_fits];
        cpu_maximum_amp = new float[n_fits];
        gpuErrchk(cudaMemcpy(cpu_maximum_pos, maximum_position, sizeof(float) * n_fits, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(cpu_maximum_amp, maximum_amplitude, sizeof(float) * n_fits, cudaMemcpyDeviceToHost));

        break;
    default:
        break;
    }

    //Compute other bound of conditions
    //TODO: do this on GPU ?
    float min_shift, max_shift;
    float min_amp, max_amp;

    float lower_shift, upper_shift;
    float lower_amp, upper_amp;

    min_shift = cpu_maximum_pos[0] - max_distance_to_maximum;
    max_shift = cpu_maximum_pos[0] + max_distance_to_maximum;
    min_amp = cpu_maximum_amp[0] * min_amplitude_of_maximum;
    max_amp = cpu_maximum_amp[0] * max_amplitude_of_maximum;

    for(int i = 1; i < n_fits ; i++){
        lower_shift = cpu_maximum_pos[i] - max_distance_to_maximum;
        upper_shift = cpu_maximum_pos[i] + max_distance_to_maximum;
        lower_amp = cpu_maximum_amp[i] * min_amplitude_of_maximum;
        upper_amp = cpu_maximum_amp[i] * max_amplitude_of_maximum;

        min_shift = (lower_shift < min_shift) ? lower_shift : min_shift;
        max_shift = (upper_shift > max_shift) ? upper_shift : max_shift;
        min_amp = (lower_amp < min_amp) ? lower_amp : min_amp;
        max_amp = (upper_amp > max_amp) ? upper_amp : max_amp;

    }

    //Apply conditions
    //Different for Rayleigh peak and Stokes/Antistokes : their shift is always > 0
    apply_fitting_constraints(min_amp, max_amp, min_shift, max_shift);


    //Cleanup
    if(storage == DEVICE){
        delete[] cpu_maximum_pos;
        delete[] cpu_maximum_amp;
    }

}

/* Given a array of X-coordinates and fitted function, computes the corresponding Y-coordinates. 
* Input is supposed to be on Host memory.
*/
void PeakFitting::compute_fitted_curve(float* cpu_x_coord, Fitted_Function* cpu_fitted, float* fitted_y){
    
    
    switch (storage)
    {
    case HOST:
        for(int i = 0; i < n_fits ; i ++){
            for(int x_n = 0; x_n < n_points ; x_n++){
                fitted_y[i * n_points + x_n] = evaluate(cpu_x_coord[i * n_points + x_n],
                    cpu_fitted->amplitude[i], cpu_fitted->shift[i], cpu_fitted->width[i], cpu_fitted->offset[i]);
            }
        
        }

        break;
    case DEVICE:
        float* gpu_y, *gpu_x, * gpu_amplitude, * gpu_shift, * gpu_width, * gpu_offset;
        gpuErrchk(cudaMalloc(&gpu_y, n_fits * n_points * sizeof(float)));
        gpuErrchk(cudaMalloc(&gpu_x, n_fits * n_points * sizeof(float)));
        gpuErrchk(cudaMalloc(&gpu_amplitude, n_fits * sizeof(float)));
        gpuErrchk(cudaMalloc(&gpu_shift, n_fits * sizeof(float)));
        gpuErrchk(cudaMalloc(&gpu_width, n_fits * sizeof(float)));
        gpuErrchk(cudaMalloc(&gpu_offset, n_fits * sizeof(float)));


        gpuErrchk(cudaMemcpy(gpu_x, cpu_x_coord, n_fits * n_points * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_amplitude, cpu_fitted->amplitude, n_fits  * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_shift, cpu_fitted->shift, n_fits  * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_width, cpu_fitted->width, n_fits  * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_offset, cpu_fitted->offset, n_fits  * sizeof(float), cudaMemcpyHostToDevice));
        
        evaluate_batch_gpu(gpu_x, gpu_y, gpu_amplitude, gpu_shift, gpu_width, gpu_offset);
        gpuErrchk(cudaMemcpy(fitted_y, gpu_y, n_fits * n_points * sizeof(float), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(gpu_x));
        gpuErrchk(cudaFree(gpu_y));
        gpuErrchk(cudaFree(gpu_amplitude));
        gpuErrchk(cudaFree(gpu_shift));
        gpuErrchk(cudaFree(gpu_width));
        gpuErrchk(cudaFree(gpu_offset));

        break;
    default:
        break;
    }

}


///////////////////////////////
/// Rayleigh (Gaussian or Lorentzian) Fitting
/////////////////////////////// 

/* Creates the user_info block for Gpufit.
* Necessary to pass the X-coordinates for the fit.
*/
void RayleighFitting::create_user_info(float* data_X, float* data_Y, float* initial_parameters, float* weights) {
    float* user_info_float;
    switch (storage)
    {
    case HOST:
        user_info_size = sizeof(float) * n_points * n_fits;
        user_info_float = (float*)malloc(user_info_size);
        memcpy(user_info_float, data_X, sizeof(float) * n_points * n_fits);
        user_info = (char*)user_info_float;
        break;

    case DEVICE:
        user_info_size = sizeof(float) * n_points * n_fits;
        gpuErrchk(cudaMalloc(&user_info_float, user_info_size));
        gpuErrchk(cudaMemcpy(user_info_float, data_X, sizeof(float) * n_points * n_fits, cudaMemcpyDeviceToDevice));
        user_info = (char*)user_info_float;
        break;

    default:
        break;
    }

}


/* Makes a sanity check on the previously performed fit.
* Retrieves the data is needed.
*/
void RayleighFitting::sanity_check(float* noise_level, float threshold, float* param){
    switch (storage)
    {
    case HOST:
        int* gpu_sanity;
        float* gpu_param, *gpu_SNR;
        float* gpu_noise_level;
        int* gpu_output_states, * gpu_n_iterations;
 
        gpuErrchk(cudaMalloc(&gpu_sanity, sizeof(int) * n_fits));
        gpuErrchk(cudaMalloc(&gpu_SNR, sizeof(float) * n_fits));
        gpuErrchk(cudaMalloc(&gpu_param, sizeof(float) * n_fits * n_parameters));
        gpuErrchk(cudaMalloc(&gpu_noise_level, sizeof(float) * n_fits ));
        gpuErrchk(cudaMalloc(&gpu_output_states, sizeof(int) * n_fits ));
        gpuErrchk(cudaMalloc(&gpu_n_iterations, sizeof(int) * n_fits ));
       
        gpuErrchk(cudaMemcpy(gpu_noise_level, noise_level, sizeof(float) * n_fits, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_param, param, sizeof(float) * n_fits * n_parameters, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_output_states, output_states, sizeof(float) * n_fits, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_n_iterations, output_n_iterations, sizeof(float) * n_fits, cudaMemcpyHostToDevice));

        LaunchKernel::sanity_check(gpu_noise_level, threshold, n_fits, gpu_param, gpu_output_states, gpu_n_iterations,max_iterations, gpu_sanity, model_id, gpu_SNR, 0, NULL);
        gpuErrchk(cudaMemcpy(sanity, gpu_sanity, sizeof(int) * n_fits, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(SNR, gpu_SNR, sizeof(float) * n_fits, cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(gpu_sanity));
        gpuErrchk(cudaFree(gpu_SNR));
        gpuErrchk(cudaFree(gpu_param));
        gpuErrchk(cudaFree(gpu_noise_level));
        gpuErrchk(cudaFree(gpu_output_states));
        gpuErrchk(cudaFree(gpu_n_iterations));
        break;
    case DEVICE: 
        LaunchKernel::sanity_check(noise_level, threshold, n_fits, param, output_states, output_n_iterations, max_iterations, sanity, model_id, SNR, 0, NULL);
        break;
    }
}

/* Applies the fit contraints for the following fits.
*/
void RayleighFitting::apply_fitting_constraints(float min_amp, float max_amp, float min_shift, float max_shift){
    parameterContext->parameter_width(1, LOWER_UPPER, min_width, max_width);
    parameterContext->parameter_amplitude(1, LOWER_UPPER, min_amp, max_amp);
    parameterContext->parameter_shift(1, LOWER_UPPER, min_shift, max_shift);
    parameterContext->parameter_offset(1, NONE, 0, 0);
    

}
///////////////////////////////
/// Stokes or AntiStokes Fitting
/////////////////////////////// 

/* Makes a sanity check on the previously performed fit.
* Retrieves the data is needed.
*/
void StokesOrAntiStokesFitting::sanity_check(float* noise_level, float threshold, float* param) {

    switch (storage)
    {
    case HOST:
        int* gpu_sanity;
        float* gpu_param, * gpu_SNR;
        float* angle_distribution_gpu;
        float* gpu_noise_level;
        int* gpu_output_states, * gpu_n_iterations;

        gpuErrchk(cudaMalloc(&gpu_sanity, sizeof(int) * n_fits));
        gpuErrchk(cudaMalloc(&gpu_SNR, sizeof(float) * n_fits));
        gpuErrchk(cudaMalloc(&gpu_param, sizeof(float) * n_fits * n_parameters));
        gpuErrchk(cudaMalloc(&angle_distribution_gpu, sizeof(float) * angle_distribution_n));
        gpuErrchk(cudaMalloc(&gpu_noise_level, sizeof(float) * n_fits));
        gpuErrchk(cudaMalloc(&gpu_output_states, sizeof(int) * n_fits));
        gpuErrchk(cudaMalloc(&gpu_n_iterations, sizeof(int) * n_fits));

        gpuErrchk(cudaMemcpy(gpu_param, param, sizeof(float) * n_fits * n_parameters, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(angle_distribution_gpu, angle_distribution, sizeof(float) * angle_distribution_n, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_noise_level, noise_level, sizeof(float) * n_fits, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_output_states, output_states, sizeof(int) * n_fits, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_n_iterations, output_n_iterations, sizeof(int) * n_fits, cudaMemcpyHostToDevice));
        
        LaunchKernel::sanity_check(gpu_noise_level, threshold, n_fits, gpu_param, gpu_output_states, gpu_n_iterations, max_iterations, gpu_sanity, model_id, gpu_SNR, angle_distribution_n, angle_distribution_gpu);

        gpuErrchk(cudaMemcpy(sanity, gpu_sanity, sizeof(int) * n_fits, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(SNR, gpu_SNR, sizeof(float) * n_fits, cudaMemcpyDeviceToHost));


        gpuErrchk(cudaFree(gpu_sanity));
        gpuErrchk(cudaFree(gpu_SNR));
        gpuErrchk(cudaFree(gpu_param));
        gpuErrchk(cudaFree(angle_distribution_gpu));
        gpuErrchk(cudaFree(gpu_noise_level));
        gpuErrchk(cudaFree(gpu_output_states));
        gpuErrchk(cudaFree(gpu_n_iterations));
        break;
    case DEVICE:
        LaunchKernel::sanity_check(noise_level, threshold, n_fits, param, output_states, output_n_iterations, max_iterations, sanity, model_id, SNR, angle_distribution_n, angle_distribution);

        break;
    }
}

/* Computes the angle distribution to performe the integration over the angle distribution
*/
void StokesOrAntiStokesFitting::initialisation(){
  
    switch (storage)
    {
    case HOST:
        angle_distribution = new float[angle_distribution_n];
        init_angle_distribution(NA_illum, NA_coll, angle, angle_distribution_n, angle_distribution);
        angle_distribution_cpu = angle_distribution;
        break;
    case DEVICE:
        gpuErrchk(cudaMalloc(&angle_distribution, sizeof(float) * angle_distribution_n));
        angle_distribution_cpu = new float[angle_distribution_n];
        init_angle_distribution(NA_illum, NA_coll, angle, angle_distribution_n, angle_distribution_cpu);
        gpuErrchk(cudaMemcpy(angle_distribution, angle_distribution_cpu, sizeof(float) * angle_distribution_n, cudaMemcpyHostToDevice));
        break;
    default:
        break;
    }
}

StokesOrAntiStokesFitting::StokesOrAntiStokesFitting(DataLocation location, size_t n_fits, int max_iterations,
    float tolerance, ModelID model_id, EstimatorID estimator_id, float start_freq, float end_freq, float SNR_threshold, 
    Curve_Extraction_Context* cec,
    float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction) :
    PeakFitting(location, n_fits, max_iterations, tolerance, model_id, estimator_id, start_freq, end_freq, SNR_threshold, cec),
    angle_distribution_n(angle_distribution_n),
    NA_illum(NA_illum),
    NA_coll(NA_coll),
    angle(angle),
    geometrical_correction(geometrical_correction)
{
    initialisation();
};

StokesOrAntiStokesFitting::StokesOrAntiStokesFitting(DataLocation location, size_t n_fits, int max_iterations,
    float tolerance, ModelID model_id, EstimatorID estimator_id, int n_points, float SNR_threshold, 
    float NA_illum, float NA_coll, float angle, int angle_distribution_n, float geometrical_correction) :
    PeakFitting(location, n_fits, max_iterations, tolerance, model_id, estimator_id, n_points, SNR_threshold),
    angle_distribution_n(angle_distribution_n),
    NA_illum(NA_illum),
    NA_coll(NA_coll),
    angle(angle),
    geometrical_correction(geometrical_correction)
{
    initialisation();
};

StokesOrAntiStokesFitting::~StokesOrAntiStokesFitting() {
    delete[] angle_distribution_cpu;
    switch (storage)
    {
    case HOST:
        //angle_distribution_cpu = angle_distribution so already freed
        break;
    case DEVICE:
        gpuErrchk(cudaFree(angle_distribution));
        break;
    default:
        break;
    }
};

/* Creates the user_info block for Gpufit.
* Necessary to pass the X-coordinates for the fit, *and* the angle distribution to compute the broadened Brillouin lineshape function
*/
void StokesOrAntiStokesFitting::create_user_info(float* data_X, float* data_Y, float* initial_parameters, float* weights) {
    float* user_info_float;
    switch (storage)
    {
    case HOST:
        user_info_size = sizeof(int) * 1 //angle_distrib_length
            + sizeof(float) * n_points * n_fits //x-coordinate data
            + sizeof(float) * angle_distribution_n; //angle_distribution
        user_info_float = (float*)malloc(user_info_size);
        memcpy(user_info_float, &angle_distribution_n, sizeof(int));
        memcpy(user_info_float + 1, angle_distribution, sizeof(float) * angle_distribution_n); //float and int have the same size
        memcpy(user_info_float + 1 + angle_distribution_n, data_X, sizeof(float) * n_fits * n_points);
        user_info = (char*)user_info_float;
        break;

    case DEVICE:
        user_info_size = sizeof(int) * 1 //angle_distrib_length
            + sizeof(float) * n_points * n_fits //x-coordinate data
            + sizeof(float) * angle_distribution_n;//angle_distribution
        gpuErrchk(cudaMalloc(&user_info_float, user_info_size));
        gpuErrchk(cudaMemcpy(user_info_float, &angle_distribution_n, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(user_info_float + 1, angle_distribution, sizeof(float) * angle_distribution_n, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(user_info_float + 1 + angle_distribution_n, data_X, sizeof(float) * n_fits * n_points, cudaMemcpyDeviceToDevice));
        user_info = (char*)user_info_float;
        break;

    default:
        break;
    }

}


void StokesOrAntiStokesFitting::dynamic_recenter(float* rayleigh_parameters, int* rayleigh_sanity){
    LaunchKernel::apply_rayleigh_X_offset(n_fits, x_coord, n_points, rayleigh_parameters, rayleigh_sanity);
}


/* Computes the angle distribution to performe the integration over the angle distribution
*/
void StokesOrAntiStokesFitting::init_angle_distribution(float NA_illum, float NA_coll, float angle, int N, float* angle_distrib) {
    double a_ill = asin(NA_illum);
    double a_col = asin(NA_coll);
    double sigma = sqrt((a_ill * a_ill + a_col * a_col) / 2);
    int count = 0;
    double step = ((1 - 1.f / N) - 1.f / N) / (N - 1);
    double n = 1.f / N;
    for (int count = 0; count < N; count++) {
        angle_distrib[count] = angle + normsInv(n, 0, sigma);
        n += step;
    }
}

/* Applies the fit contraints for the following fits.
*/
void StokesOrAntiStokesFitting::apply_fitting_constraints(float min_amp, float max_amp, float min_shift, float max_shift) {
    parameterContext->parameter_width(1, LOWER_UPPER, min_width, max_width);
    parameterContext->parameter_amplitude(1, LOWER_UPPER, min_amp / angle_distribution_n, max_amp / angle_distribution_n);

    float lower = (abs(min_shift) > abs(max_shift) ) ? abs(max_shift) : abs(min_shift) ;
    float upper = (abs(min_shift) < abs(max_shift)) ? abs(max_shift) : abs(min_shift) ;
    printf("DEBUG : %f %f %f => %f %f \n", lower, upper, geometrical_correction, lower * geometrical_correction, upper * geometrical_correction);
    parameterContext->parameter_shift(1, LOWER_UPPER, lower * geometrical_correction , upper * geometrical_correction);
    parameterContext->parameter_offset(1, NONE, 0, 0);
    

}

/* Function called to compute on the GPU in a batch for n_fits parameters n_points Y-coordinates given X-coordinates
*/
void StokesOrAntiStokesFitting::evaluate_batch_gpu(float* x, float* output_y, float* amplitude, float* shift, float* width, float* offset) {
    float* gpu_angle_distribution = angle_distribution;
    if (storage == HOST) {
        gpuErrchk(cudaMalloc(&gpu_angle_distribution, angle_distribution_n * sizeof(float)));
        gpuErrchk(cudaMemcpy(gpu_angle_distribution, angle_distribution, angle_distribution_n * sizeof(float), cudaMemcpyHostToDevice));
    }

    LaunchKernel::batch_evaluation(model_id, x, output_y, n_fits, n_points, amplitude, shift, width, offset, gpu_angle_distribution, angle_distribution_n);

    if (storage == HOST)
        gpuErrchk(cudaFree(gpu_angle_distribution));
}


#include "../other/asa241.h"
double StokesOrAntiStokesFitting::normsInv(double p, double mu, double sigma)
{
    return mu + sigma * r8_normal_01_cdf_inverse(p);
}