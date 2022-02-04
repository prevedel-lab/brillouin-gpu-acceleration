
%% Generate Synthetic signal
% Parameters to create synthetic signal
width = 500;
height = 300;
order = 5 ;

syn_rayleigh(1) = 10; %amplitude
syn_rayleigh(2) = 0; %center
syn_rayleigh(3) = 0.5; %width

syn_stokes(1) = 8; %amplitude
syn_stokes(2) = -5; %center
syn_stokes(3) = 1; %width

syn_antistokes(1) = 8; %amplitude
syn_antistokes(2) = 5; %center
syn_antistokes(3) = 1; %width

[synthetic_signal, y_pos, rayleigh_pos_fit, frq_lut] = Synthetic_signal(width ,height, order, syn_rayleigh, syn_stokes, syn_antistokes);
imagesc(synthetic_signal)


%% Fit the peaks with the GPU library

addpath('GPU_Fitting');
if exist('g','var')
    pipeline_close(g);
    clear g
end
[g, frq_lut, start_ROI, end_ROI] = Initialize_GPU_fitting(synthetic_signal, y_pos, rayleigh_pos_fit - 1, 'SPLINE');
[Stokes_par, antiStokes_par, Rayleigh_par] = pipeline_sum_and_fit(g, synthetic_signal, false);
[summed_curves, X, Y, X_fitted, Y_fitted, Stokes_par, antiStokes_par, Rayleigh_par] = debug_get_raw_and_fitted_spectra(g, synthetic_signal, false);            
pipeline_close(g);
clear g

%% Displaying result

subplot(3,2,[1,4])
title("synthetic signal and rayleigh peak")
hold on
imagesc(synthetic_signal);

legend_str = [];
for i=1:order-1
    plot(rayleigh_pos_fit(:,i), y_pos , "o");
    legend_str = [ legend_str, "Order "+string(i)];
end
legend(legend_str);
axis([0 width 0 height]);
hold off

subplot(3,2, [5,6])
title("Exemple of summed and fitted curve");
line = 200 ;
hold on
plot(X(:, line), Y(:, line));
plot(X_fitted(:, line), Y_fitted(:, line), "--");
legend("summed curve", "fitted curve");
hold off

title("frequency look up table")
imagesc(frq_lut)
