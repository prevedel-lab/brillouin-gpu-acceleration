function [g, frq_lut, start_ROI, end_ROI, a_gpu,b_gpu,c_gpu] = Initialize_GPU_fitting(ref_im, y_pos, rayleight_peak_pos_fit, interpolation)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


    libpath = 'GPU_Fitting\';
    libname = 'GPU_Fitting';
    hfilename = 'GPU_Fitting\DLL_wrapper.h';

    g = GPU_fitting_class(libpath, libname, hfilename); 
    w = size(ref_im,1); h = size(ref_im,2);


    %Parameters for the summed curve
    curve_extraction_context.max_n_peaks = 20;
    curve_extraction_context.n_points = 20 * 3 + 1;
    FSR=15;
    curve_extraction_context.starting_freq = -FSR/2 ;
    curve_extraction_context.step = FSR / (curve_extraction_context.n_points - 1);
    curve_extraction_context.interpolation = interpolation ;
    
    single_order = false;
    if single_order
        curve_extraction_context.starting_order = 1;
        curve_extraction_context.ending_order = curve_extraction_context.starting_order+1;        
    else
        curve_extraction_context.starting_order = 0;
        curve_extraction_context.ending_order = size(rayleight_peak_pos_fit,2);
    end
    
    %Parameters for the Fitting Algorithm
    fitting_context.n_fits = length(y_pos);
    fitting_context.max_iteration = 50;
    fitting_context.tolerance = 1e-4;
    fitting_context.SNR_threshold = 5;

    
    %Parameters for Stokes/antiStokes calculation
    angle_context.NA_illum = 0.01/1.33;
    angle_context.NA_coll = 0.01/1.33;
    angle_context.angle = pi;
    angle_context.angle_distribution_n = 1000;
    angle_context.geometrical_correction = 1;
    %pipeline_initialisation(g, w, h, fitting_context, curve_extraction_context, 'FIT_STOKES', 'FIT_LORENTZIAN', 'FIT_ANTISTOKES', [-6.9 -3.8], [-1.3 1.3], [3.4 6.1], angle_context);
    pipeline_initialisation(g, w, h, fitting_context, curve_extraction_context, 'FIT_LORENTZIAN', 'FIT_LORENTZIAN', 'FIT_LORENTZIAN', [-7 -3], [-1.5 1.5], [3 7], angle_context);
    

    x = round(y_pos);
    peak_numbers = zeros(w,1,'int32'); peak_numbers(x) = size(rayleight_peak_pos_fit,2);
    peak_original_positions = zeros(curve_extraction_context.max_n_peaks, w, 'single');
    peak_remapped_positions = zeros(curve_extraction_context.max_n_peaks, w, 'single');

    for i=x(1):x(end)
        remapped_peak_distance = (rayleight_peak_pos_fit(i - x(1) + 1,end)-rayleight_peak_pos_fit(i - x(1) +1,1))/(size(rayleight_peak_pos_fit,2) - 1);
        for j = 1:size(rayleight_peak_pos_fit,2)
            peak_original_positions(j,i) = rayleight_peak_pos_fit(i - x(1) + 1,j);
            peak_remapped_positions(j,i) = (j-1)*remapped_peak_distance + rayleight_peak_pos_fit(i - x(1) + 1,1);
        end
    end

    [translation_lut, frq_lut, start_ROI, end_ROI] = create_preprocessing_data(g, peak_numbers, peak_original_positions, peak_remapped_positions);
    pipeline_send_experiment_settings(g,  peak_numbers, peak_original_positions, translation_lut, frq_lut, start_ROI, end_ROI);


end



