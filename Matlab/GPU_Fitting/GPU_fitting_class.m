classdef GPU_fitting_class < handle
    %GPU_fitting_class encapsulates the GPU_fitting library to provide a
    %convinient interface to matlab

    
    properties (Access = private)
       lib
       
        %variables
        img_width; img_height;
        
        curve_extraction_context;
        fitting_context;
        angle_context;
        
        start_ROI; end_ROI;
        peak_numbers;
        frq_lut;
        translation_lut;
        
        stokes_range;
        antistokes_range;
        rayleigh_range;
        
        debug_stokes_function;
        debug_antistokes_function;
        debug_rayleigh_function;
        
    end
    
    methods
        function obj = GPU_fitting_class(libpath,libname,hfilename)
            obj.lib = libname;
            
            if not(libisloaded(libname))
                loadlibrary([libpath libname],hfilename, 'mfilename', 'somelibM')
            end
            
            %libfunctions(libname, '-full')
        end
        
        function delete(obj)
            unloadlibrary(obj.lib);
        end
        
        function pipeline_initialisation(obj,width, height, fitting_context, curve_extraction_context, stokes_fitting_function, rayleigh_fitting_function, antistokes_fitting_function, stokes_range, rayleigh_range, antistokes_range, angle_context)            
            obj.img_width = width; obj.img_height = height;
            
            obj.fitting_context = fitting_context;
            obj.curve_extraction_context = curve_extraction_context;
            obj.angle_context = angle_context;
            
            obj.stokes_range = stokes_range;
            obj.antistokes_range = antistokes_range;
            obj.rayleigh_range = rayleigh_range;
            
            %for debug
            obj.debug_stokes_function = stokes_fitting_function;
            obj.debug_antistokes_function = antistokes_fitting_function;
            obj.debug_rayleigh_function = rayleigh_fitting_function;
            
            calllib(obj.lib,'pipeline_initialisation', width, height, fitting_context, curve_extraction_context, stokes_fitting_function, rayleigh_fitting_function, antistokes_fitting_function, stokes_range, rayleigh_range, antistokes_range, angle_context);
        end
        function pipeline_close(obj)             
            clear pipeline_sum_and_fit;
            clear pipeline_get_gof;
            calllib(obj.lib,'pipeline_close');
        end  
        
        function pipeline_set_constraints(obj, use_constraints, min_width, max_width, max_distance_to_maximum, min_amplitude_of_maximum, max_amplitude_of_maximum)
            calllib(obj.lib, 'pipeline_set_constraints',  use_constraints, min_width, max_width, max_distance_to_maximum, min_amplitude_of_maximum, max_amplitude_of_maximum);
            
        end
        
        function [translation_lut, frq_lut, start_ROI, end_ROI] = create_preprocessing_data(obj, peak_numbers, peak_original_positions, peak_remapped_positions)            
            peak_original_positions = single(peak_original_positions);
            peak_remapped_positions = single(peak_remapped_positions);
            
            width = obj.img_width;height = obj.img_height;
            

            translation_lut = zeros(width+1,1, 'int32');
            frq_lut = zeros(width,height, 'single');
            start_ROI = zeros(width,1, 'int32');
            end_ROI = zeros(width,1,'int32');
            [~, ~, ~, ~, translation_lut, frq_lut, start_ROI, end_ROI] = calllib(obj.lib,'create_preprocessing_data', obj.curve_extraction_context, width, height, peak_numbers, peak_original_positions, peak_remapped_positions, translation_lut, frq_lut, start_ROI, end_ROI);
            
                        
            obj.peak_numbers = peak_numbers;
            obj.start_ROI = start_ROI; obj.end_ROI = end_ROI;
            obj.frq_lut = frq_lut;
            obj.translation_lut = translation_lut;
        end 
        
        function pipeline_send_experiment_settings(obj,  peak_numbers, peak_original_positions, translation_lut, frq_lut, start_ROI, end_ROI)
            calllib(obj.lib,'pipeline_send_experiment_settings', peak_numbers, peak_original_positions, translation_lut, frq_lut, start_ROI, end_ROI);
        end 
        
        function [r_Stokes, r_antiStokes, r_Rayleigh] = pipeline_sum_and_fit(obj, image, dynamic_recentring)  
            if ~isa(image, 'uint16')
                error('The image must be an "uint16"')
            end
            n_fits = obj.fitting_context.n_fits;
            persistent Stokes;
            if isempty(Stokes)
                Stokes = create_empty_Fitted_Function_struct(obj, n_fits);
            end
            
            persistent Rayleigh;
            if isempty(Rayleigh)
                Rayleigh = create_empty_Fitted_Function_struct(obj, n_fits);
            end
            
            persistent antiStokes;
            if isempty(antiStokes)
                antiStokes = create_empty_Fitted_Function_struct(obj, n_fits);
            end
               
            [~, r_Stokes, r_Rayleigh, r_antiStokes] = calllib(obj.lib,'pipeline_sum_and_fit', image, dynamic_recentring, Stokes, Rayleigh, antiStokes);
          
        end 

        function [r_Stokes, r_antiStokes, r_Rayleigh, r_timings] = pipeline_sum_and_fit_timed(obj, image, dynamic_recentring)  
            if ~isa(image, 'uint16')
                error('The image must be an "uint16"')
            end
            n_fits = obj.fitting_context.n_fits;
            persistent Stokes;
            if isempty(Stokes)
                Stokes = create_empty_Fitted_Function_struct(obj, n_fits);
            end
            
            persistent Rayleigh;
            if isempty(Rayleigh)
                Rayleigh = create_empty_Fitted_Function_struct(obj, n_fits);
            end
            
            persistent antiStokes;
            if isempty(antiStokes)
                antiStokes = create_empty_Fitted_Function_struct(obj, n_fits);
            end
            
            timings = zeros(8,1, 'single');
               
            [~, r_Stokes, r_Rayleigh, r_antiStokes, r_timings] = calllib(obj.lib,'pipeline_sum_and_fit_timed', image, dynamic_recentring, Stokes, Rayleigh, antiStokes, timings);
          
        end 

        
        function [r_Stokes_gof, r_antiStokes_gof, r_Rayleigh_gof] = pipeline_get_gof(obj)
            
            n_fits = obj.fitting_context.n_fits;
            persistent Stokes_gof;
            if isempty(Stokes_gof)
                Stokes_gof.states = zeros(n_fits,1, 'int32');
                Stokes_gof.chi_squares = zeros(n_fits,1, 'single');
                Stokes_gof.n_iterations = zeros(n_fits,1, 'int32');
                Stokes_gof.SNR = zeros(n_fits,1, 'single');
            end

            persistent Rayleigh_gof
            if isempty(Rayleigh_gof)
                Rayleigh_gof.states = zeros(n_fits,1, 'int32');
                Rayleigh_gof.chi_squares = zeros(n_fits,1, 'single');
                Rayleigh_gof.n_iterations = zeros(n_fits,1, 'int32');
                Rayleigh_gof.SNR = zeros(n_fits,1, 'single');
            end

            persistent antiStokes_gof;
            if isempty(antiStokes_gof)
                antiStokes_gof.states = zeros(n_fits,1, 'int32');
                antiStokes_gof.chi_squares = zeros(n_fits,1, 'single');
                antiStokes_gof.n_iterations = zeros(n_fits,1, 'int32');
                antiStokes_gof.SNR = zeros(n_fits,1, 'single');
            end
            
            [r_Stokes_gof, r_Rayleigh_gof, r_antiStokes_gof ] = calllib(obj.lib,'pipeline_get_gof', Stokes_gof, Rayleigh_gof, antiStokes_gof );
       
            
        end
        
                
        %functions for debug
        function [summed_curves, X, Y, X_fitted, Y_fitted, Stokes_par, antiStokes_par, Rayleigh_par] = debug_get_raw_and_fitted_spectra(obj, image, recentering)
            
            
            
            [Stokes_par, antiStokes_par, Rayleigh_par] = pipeline_sum_and_fit(obj, image, recentering); 
            
            summed_curves = debug_create_summed_curves(obj, image);
%             noise_deviation = debug_estimate_noise_deviation(obj, summed_curves);
            
            %get the full spectrum:
            [X, Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, [obj.stokes_range(1) obj.antistokes_range(2)]);
            X = reshape(X, n_points, obj.fitting_context.n_fits);
            Y = reshape(Y, n_points, obj.fitting_context.n_fits);
            
            X_fitted = [];
            Y_fitted = [];
            nan_spacer = nan+zeros(1,obj.fitting_context.n_fits, 'like', X);
            
            %Stokes
            if recentering
                [data_X, data_Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, obj.stokes_range, Rayleigh_par );
            else
                [data_X, data_Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, obj.stokes_range);
            end
            
            %[data_X, data_Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, obj.stokes_range);
%             Stokes_par = create_empty_Fitted_Function_struct(obj, obj.fitting_context.n_fits);
%             [~, ~, ~, ~, ~, Stokes_par] = calllib(obj.lib,'fitting', data_X, data_Y, noise_deviation, n_points, 'FIT_STOKES', obj.fitting_context, obj.angle_context, Stokes_par);
            data_Y_fitted = zeros(size(data_Y),'like',data_Y);
            [~, ~, ~, data_Y_fitted] = calllib(obj.lib,'calculate_fitted_curve', data_X, n_points, obj.fitting_context.n_fits, obj.debug_stokes_function, Stokes_par, obj.angle_context, data_Y_fitted);
            X_fitted = [X_fitted; nan_spacer; reshape(data_X, n_points, obj.fitting_context.n_fits)];
            Y_fitted = [Y_fitted; nan_spacer; reshape(data_Y_fitted, n_points, obj.fitting_context.n_fits)];
            
            %Rayleigh
            [data_X, data_Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, obj.rayleigh_range);
%             Rayleigh_par = create_empty_Fitted_Function_struct(obj, obj.fitting_context.n_fits);
%             [~, ~, ~, ~, ~, Rayleigh_par] = calllib(obj.lib,'fitting', data_X, data_Y, noise_deviation, n_points, 'FIT_LORENTZIAN', obj.fitting_context, obj.angle_context, Rayleigh_par);
            data_Y_fitted = zeros(size(data_Y),'like',data_Y);
            [~, ~, ~, data_Y_fitted] = calllib(obj.lib,'calculate_fitted_curve', data_X, n_points, obj.fitting_context.n_fits, obj.debug_rayleigh_function, Rayleigh_par, obj.angle_context, data_Y_fitted);
            X_fitted = [X_fitted; nan_spacer; reshape(data_X, n_points, obj.fitting_context.n_fits)];
            Y_fitted = [Y_fitted; nan_spacer; reshape(data_Y_fitted, n_points, obj.fitting_context.n_fits)];
            
            %antiStokes
            if recentering
                [data_X, data_Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, obj.antistokes_range, Rayleigh_par);
            else
                [data_X, data_Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, obj.antistokes_range);
            end
            
%             antiStokes_par = create_empty_Fitted_Function_struct(obj, obj.fitting_context.n_fits);
%             [~, ~, ~, ~, ~, antiStokes_par] = calllib(obj.lib,'fitting', data_X, data_Y, noise_deviation, n_points, 'FIT_ANTISTOKES', obj.fitting_context, obj.angle_context, antiStokes_par);
            data_Y_fitted = zeros(size(data_Y),'like',data_Y);
            [~, ~, ~, data_Y_fitted] = calllib(obj.lib,'calculate_fitted_curve', data_X, n_points, obj.fitting_context.n_fits, obj.debug_antistokes_function, antiStokes_par, obj.angle_context, data_Y_fitted);
            X_fitted = [X_fitted; nan_spacer; reshape(data_X, n_points, obj.fitting_context.n_fits)];
            Y_fitted = [Y_fitted; nan_spacer; reshape(data_Y_fitted, n_points, obj.fitting_context.n_fits)];
        end
        
        function [a, b, c] = linearize_pixelspace(obj, peak_numbers, original_peak_positions, remapped_peak_positions)
            a = zeros(obj.img_width, "single");
            b = zeros(obj.img_width, "single");
            c = zeros(obj.img_width, "single");
            [~,~,~,~,a,b,c] = calllib(obj.lib, 'linearize_pixelspace', obj.curve_extraction_context, obj.img_width, peak_numbers, original_peak_positions, remapped_peak_positions, a, b, c);
            
        end
        
        %Not tested
        function [frq_lut] = create_frq_lut(obj, peak_numbers, original_peak_positions, a, b, c)
            frq_lut = zeros([obj.img_width obj.img_height], "single");
            [~,frq_lut,~,~,~,~,~] = calllib(obj.lib, 'create_frq_lut', obj.curve_extraction_context, frq_lut, obj.img_height, obj.img_width, peak_numbers, original_peak_positions, a, b, c);
        end
        
        
        
        
    end
    
    methods (Access = private)
        
        function [summed_curves] = debug_create_summed_curves(obj, image)
            if ~isa(image, 'uint16')
                error('The image must be an "uint16"')
            end
            if size(image,1)~=obj.img_width || size(image,2)~=obj.img_height
                error('The image size must correspond to the one specified in the "pipeline_initialisation" function')
            end
            summed_curves = zeros(obj.img_width, obj.curve_extraction_context.n_points);
            [~, ~, ~, ~, ~, ~, summed_curves] = calllib(obj.lib,'create_summed_curves', obj.curve_extraction_context, image, obj.img_height, obj.img_width, obj.start_ROI, obj.end_ROI, obj.peak_numbers, obj.frq_lut, summed_curves);           
        end  
        function [data_X, data_Y, n_points] = debug_extract_data_for_fitting(obj, summed_curves, frq_range, Rayleigh_pos)
            start_frq=frq_range(1);
            end_frq=frq_range(2);
            
            n_points = int32(0);
            data_X = zeros(obj.curve_extraction_context.n_points*obj.fitting_context.n_fits,1, 'single');
            data_Y = data_X;
            if exist('Rayleigh_pos','var')
                %'Rayleigh_pos' must have the same structure as the variable
                %created by 'create_empty_Fitted_Function_struct'
                [~, ~, ~, n_points, data_X, data_Y] = calllib(obj.lib,'extract_data_for_fitting_recentering', obj.curve_extraction_context, obj.img_width, summed_curves, obj.translation_lut, start_frq,end_frq, n_points, data_X, data_Y, Rayleigh_pos);
            else
                [~, ~, ~, n_points, data_X, data_Y] = calllib(obj.lib,'extract_data_for_fitting', obj.curve_extraction_context, obj.img_width, summed_curves, obj.translation_lut, start_frq,end_frq, n_points, data_X, data_Y);
            end
            
            data_X = data_X(1:(obj.fitting_context.n_fits*n_points));
            data_Y = data_Y(1:(obj.fitting_context.n_fits*n_points));
        end

        
        
        function noise_deviation = debug_estimate_noise_deviation(obj, summed_curves)
            noise_deviation = zeros(obj.fitting_context.n_fits,1,'single');
            [~, ~, ~, noise_deviation] = calllib(obj.lib,'estimate_noise_deviation', obj.curve_extraction_context, obj.img_width, summed_curves, obj.translation_lut,  obj.stokes_range(1), obj.antistokes_range(2), noise_deviation);
        end
        
        function [Fitted_Function_struct] = create_empty_Fitted_Function_struct(~, size)
            Fitted_Function_struct.amplitude = zeros(size,1, 'single');
            Fitted_Function_struct.shift     = zeros(size,1, 'single');
            Fitted_Function_struct.width     = zeros(size,1, 'single');
            Fitted_Function_struct.offset    = zeros(size,1, 'single');
            Fitted_Function_struct.sanity    = zeros(size,1, 'int32');
        end
    end
end


