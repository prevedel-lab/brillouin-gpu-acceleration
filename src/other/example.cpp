#include "example.h"


#define DISPLAY_SYNTHETIC_SIGNAL

void main() {

	// ====
	// Perform 2 test to check if GPUfit works
	// Result gets displayed on the console
	// ====
	check_spline_fit();
	check_poly2_fit(10, 500);
	check_lorentzian_fit(10, 500);

	// ====
	// Initialize GPU Fitting Parameters
	// ====

	//Parameters concerning the order fusion steps
	Curve_Extraction_Context cec;
	cec.max_n_peaks = 20;
	cec.n_points = 20 * 3 + 1;
	cec.starting_freq = -15.0 / 2;
	cec.step = (7.5 - cec.starting_freq) / (cec.n_points - 1);
	cec.starting_order = 0;
	cec.ending_order = 5;
	cec.interpolation = SPLINE; //LINEAR or SPLINE

	//Parameters to compute broadened Brillouin lineshape
	// - not needed for Lorentzian fit, could be removed
	Stokes_antiStokes_Context angle_context;
	angle_context.angle = 3.1415f;
	angle_context.angle_distribution_n = 1000;
	angle_context.NA_coll = 0.01 / 1.33;
	angle_context.NA_illum = 0.01 / 1.33;
	angle_context.geometrical_correction = 1; //theoretical is sqrt(2)


	// Parameters for Gpufit
	Fitting_Context fitting_context;
	fitting_context.max_iteration = 50;
	fitting_context.n_fits = 400;
	fitting_context.tolerance = 1e-4;
	fitting_context.SNR_threshold = 5;

	//Frequency range (in GHz) of the 3 peaks
	float stokes_range[2] = { -7, -3 };
	float rayleigh_range[2] = { -1.5, 1.5 };
	float antistokes_range[2] = { 3, 7 };
	
	// ====
	// Creating a synthetic signal
	// ====
	float syn_rayleigh_center = 0;
	float syn_stokes_center = -5;
	float syn_antistokes_center = 5;

	SyntheticSignal signal(1000, 500, fitting_context.n_fits, cec.ending_order);
	signal.generate_frq_lut(-7.5, 7.5, 1.2);
	signal.set_signal_value(10); // Background value
	signal.generate_signal(FIT_LORENTZIAN, 5000, syn_rayleigh_center, 0.25, 0);
	signal.generate_signal(FIT_LORENTZIAN, 4000, syn_stokes_center, 0.5, 0); //stokes
	signal.generate_signal(FIT_LORENTZIAN, 4000, syn_antistokes_center, 0.5, 0); //antistokes
	uint16_t* synthetic_data = signal.get_transposed_signal_uint16(65500);

#ifdef DISPLAY_SYNTHETIC_SIGNAL
	//Displaying synthetic signal
	auto signal_to_be = cimg_library::CImg<uint16_t>(synthetic_data, signal.get_height(), signal.get_width(), 1, 1, true);
	cimg_library::CImgDisplay display;
	display.display(signal_to_be);
	while (!display.is_closed()) {
		display.wait();
	}
#endif // DISPLAY_SYNTHETIC_SIGNAL

	// ====
	// Pre-processing of the data : frq LUT and ROI allocation and creation
	// ====
	
	/* Do the pre - preprocessing (once per setup ) */
	cudaExtent dim{ signal.get_height(), signal.get_width(), 1 }; //! Transposition
	int* translation_lut = new int[dim.width + 1];
	float* frq_lut = new float[dim.width * dim.height];
	int* start_ROI = new int[dim.width];
	int* end_ROI = new int[dim.width];

	//Need to put the arrays in the correct shape
	int* peak_numbers = new int[dim.width];
	float* original_peak_positions = new float[dim.width * cec.max_n_peaks];
	float* remapped_peak_positions = new float[dim.width * cec.max_n_peaks];
	float step;
	for (int i = 0; i < dim.width; i++) {
		if (i < signal.y_pos[0] || i > signal.y_pos[signal.n_fit - 1]) {  
			// We are before the first or after the last column with detected signal
			// so there is no rayleigh peak
			peak_numbers[i] = 0;
			for (int j = 0; j < cec.max_n_peaks; j++) {
				original_peak_positions[j + i * cec.max_n_peaks] = 0;
				remapped_peak_positions[j + i * cec.max_n_peaks] = 0;

			}
		}
		else {
			// We know from the synthetic signal how many orders and the position of each rayleigh peak
			// on experimental data, this needs to come from a fit on the data
			peak_numbers[i] = signal.n_order;
			step = (signal.rayleigh_pos[(i - (signal.y_pos[0])) * signal.n_order + (signal.n_order - 1)]
				- signal.rayleigh_pos[(i - (signal.y_pos[0])) * signal.n_order + 0])
				/ (signal.n_order - 1);
			for (int j = 0; j < signal.n_order; j++) {
				original_peak_positions[j + i * cec.max_n_peaks] = signal.rayleigh_pos[(i - (signal.y_pos[0])) * signal.n_order + j];
				remapped_peak_positions[j + i * cec.max_n_peaks] = j * step + signal.rayleigh_pos[(i - (signal.y_pos[0])) * signal.n_order + 0];
			}

			for (int j = signal.n_order; j < cec.max_n_peaks; j++) {
				original_peak_positions[j + i * cec.max_n_peaks] = 0;
				remapped_peak_positions[j + i * cec.max_n_peaks] = 0;
			}
		}
	}
	
	//We can now create the frq LUT and the ROI, common to all the images
	DLL_wrapper::create_preprocessing_data(&cec, dim.width, dim.height, peak_numbers, original_peak_positions,
		remapped_peak_positions, translation_lut, frq_lut, start_ROI, end_ROI);

	// ====
	// Processing using the pipeline
	// ====	

	//Allocating memory for the fitting results
	int n_fits = fitting_context.n_fits;
	Fitted_Function* antistokes, * rayleigh, * stokes;

	antistokes = new Fitted_Function();
	antistokes->amplitude = new float[n_fits];
	antistokes->shift = new float[n_fits];
	antistokes->width = new float[n_fits];
	antistokes->offset = new float[n_fits];
	antistokes->sanity = new int[n_fits];

	rayleigh = new Fitted_Function();
	rayleigh->amplitude = new float[n_fits];
	rayleigh->shift = new float[n_fits];
	rayleigh->width = new float[n_fits];
	rayleigh->offset = new float[n_fits];
	rayleigh->sanity = new int[n_fits];

	stokes = new Fitted_Function();
	stokes->amplitude = new float[n_fits];
	stokes->shift = new float[n_fits];
	stokes->width = new float[n_fits];
	stokes->offset = new float[n_fits];
	stokes->sanity = new int[n_fits];

	/* Start of DLL_wrapper test */
	printf("\n=====\n Starting DLL wrapper test... \n===== \n");
	DLL_wrapper::pipeline_initialisation(dim.width, dim.height, &fitting_context, &cec, FIT_LORENTZIAN, FIT_LORENTZIAN, FIT_LORENTZIAN, stokes_range, rayleigh_range, antistokes_range, &angle_context);
	DLL_wrapper::pipeline_send_experiment_settings(peak_numbers, original_peak_positions, translation_lut, frq_lut, start_ROI, end_ROI);
	//DLL_wrapper::pipeline_set_constraints(true, 0.1, 3, 1, 0.1, 5);
	DLL_wrapper::pipeline_sum_and_fit(synthetic_data, true, stokes, rayleigh, antistokes); //Call this function once for each image to process
	//DLL_wrapper::pipeline_get_gof(stokes_gof, rayleigh_gof, antistokes_gof);
	DLL_wrapper::pipeline_close();

	//Display result of the fit
	for (int i = 0; i < n_fits; i++) {
		printf("== Fit #%d == \r\nParameter\t| Value (GHz)\t| Fit (GHz)\t| Difference (MHz)\r\n", i);
		printf("Rayleigh\t| %.2f\t| %.2f\t| %.2f \r\n",
			syn_rayleigh_center, rayleigh->shift[i], abs(syn_rayleigh_center - rayleigh->shift[i]) * 1000 );
		printf("Stokes  \t| %.2f\t| %.2f\t| %.2f \r\n",
			syn_stokes_center, stokes->shift[i], abs(syn_stokes_center - stokes->shift[i]) * 1000);
		printf("antiStokes\t| %.2f\t| %.2f\t| %.2f \r\n",
			syn_antistokes_center, antistokes->shift[i], abs(syn_antistokes_center - antistokes->shift[i]) * 1000);

	}


	// ====
	// Processing using the standalone functions
	// ====
	printf("\n=====\n Starting standalone functions test... \n===== \n");
	
	// Linearizing pixelspace
	float* standalone_a = new float[dim.width];
	float* standalone_b = new float[dim.width];
	float* standalone_c = new float[dim.width];
	DLL_wrapper::linearize_pixelspace(&cec, dim.width, peak_numbers, original_peak_positions, remapped_peak_positions,
		standalone_a, standalone_b, standalone_c);


	//creating frq lut
	float* standalone_frq_lut = new float[dim.width * dim.height];
	DLL_wrapper::create_frq_lut(&cec, standalone_frq_lut, dim.height, dim.width, peak_numbers, original_peak_positions,
		standalone_a, standalone_b, standalone_c);

	//creating ROI
	int* standalone_start_ROI = new int[dim.width];
	int* standalone_end_ROI = new int[dim.width];
	DLL_wrapper::create_ROI(&cec, dim.height, dim.width, peak_numbers, original_peak_positions, standalone_a, standalone_b,
		standalone_c, standalone_start_ROI, standalone_end_ROI);

	//creating summed curves
	float* standalone_summed_curves = new float[dim.width * dim.height];
	DLL_wrapper::create_summed_curves(&cec, synthetic_data, dim.height, dim.width, standalone_start_ROI, standalone_end_ROI,
		peak_numbers, standalone_frq_lut, standalone_summed_curves);
	
	//Create translation lut 
	//enables to know without testing which columns have to be fitted (>3 rayleigh peaks)
	int* sa_translation_lut = new int[dim.width + 1];
	int sa_n_fits = 0;
	for (int i = 0; i < dim.width; i++) {
		if (peak_numbers[i] >= 3) {
			sa_translation_lut[i] = sa_n_fits;
			sa_n_fits++;
		}
		else {
			sa_translation_lut[i] = -1;
		}

	}
	sa_translation_lut[dim.width] = sa_n_fits;

	int N = 1000;
	int* n_points = new int[1];
	//We don't know before hand how many points there are, so we allocate the maximum possible size
	float* data_x = new float[dim.width * cec.n_points];
	float* data_y = new float[dim.width * cec.n_points];
	float* noise_deviation = new float[dim.width * cec.n_points];

	// Calculation of noise deviation (for sanity check)
	float SNR_threshold = 2;
	DLL_wrapper::estimate_noise_deviation(&cec, dim.width, standalone_summed_curves, sa_translation_lut, -6.0, 6.0, noise_deviation);

	// Extrating and fitting Rayleigh
	Fitted_Function * sa_rayleigh;
	sa_rayleigh = new Fitted_Function();
	sa_rayleigh->amplitude = new float[sa_n_fits];
	sa_rayleigh->shift = new float[sa_n_fits];
	sa_rayleigh->width = new float[sa_n_fits];
	sa_rayleigh->offset = new float[sa_n_fits];
	sa_rayleigh->sanity = new int[sa_n_fits];
	DLL_wrapper::extract_data_for_fitting(&cec, dim.width, standalone_summed_curves, sa_translation_lut, rayleigh_range[0], rayleigh_range[1],
		n_points, data_x, data_y);
	DLL_wrapper::fitting(data_x, data_y, noise_deviation, n_points[0], FIT_LORENTZIAN, &fitting_context, &angle_context, sa_rayleigh);

	

	// Extrating and fitting Antistokes
	Fitted_Function *sa_antistokes;
	sa_antistokes = new Fitted_Function();
	sa_antistokes->amplitude = new float[sa_n_fits];
	sa_antistokes->shift = new float[sa_n_fits];
	sa_antistokes->width = new float[sa_n_fits];
	sa_antistokes->offset = new float[sa_n_fits];
	sa_antistokes->sanity = new int[sa_n_fits];
	DLL_wrapper::extract_data_for_fitting(&cec, dim.width, standalone_summed_curves, sa_translation_lut, antistokes_range[0], antistokes_range[1],
		n_points, data_x, data_y);
	DLL_wrapper::fitting(data_x, data_y, noise_deviation, n_points[0], FIT_LORENTZIAN, &fitting_context, &angle_context, sa_antistokes);
	

	// Extrating and fitting Stokes

	Fitted_Function *sa_stokes;
	sa_stokes = new Fitted_Function();
	sa_stokes->amplitude = new float[sa_n_fits];
	sa_stokes->shift = new float[sa_n_fits];
	sa_stokes->width = new float[sa_n_fits];
	sa_stokes->offset = new float[sa_n_fits];
	sa_stokes->sanity = new int[sa_n_fits];
	DLL_wrapper::extract_data_for_fitting(&cec, dim.width, standalone_summed_curves, sa_translation_lut, stokes_range[0], stokes_range[1],
		n_points, data_x, data_y);
	DLL_wrapper::fitting(data_x, data_y, noise_deviation, n_points[0], FIT_LORENTZIAN, &fitting_context, &angle_context, sa_stokes);

	//Display result of the fit
	for (int i = 0; i < n_fits; i++) {
		printf("== Fit #%d == \r\nParameter\t| Value (GHz)\t| Fit (GHz)\t| Difference (MHz)\r\n", i);
		printf("Rayleigh\t| %.2f\t| %.2f\t| %.2f \r\n",
			syn_rayleigh_center, sa_rayleigh->shift[i], abs(syn_rayleigh_center - sa_rayleigh->shift[i]) * 1000);
		printf("Stokes  \t| %.2f\t| %.2f\t| %.2f \r\n",
			syn_stokes_center, sa_stokes->shift[i], abs(syn_stokes_center - sa_stokes->shift[i]) * 1000);
		printf("antiStokes\t| %.2f\t| %.2f\t| %.2f \r\n",
			syn_antistokes_center, sa_antistokes->shift[i], abs(syn_antistokes_center - sa_antistokes->shift[i]) * 1000);

	}

	// ====
	// Cleanup
	// ====

	delete[] translation_lut;
	delete[] frq_lut;
	delete[] start_ROI;
	delete[] end_ROI;

	delete[] peak_numbers;
	delete[] original_peak_positions;
	delete[] remapped_peak_positions;

	delete[] antistokes->amplitude;
	delete[] antistokes->shift;
	delete[] antistokes->width;
	delete[] antistokes->offset;
	delete[] antistokes->sanity;
	delete antistokes;
	
	delete[] rayleigh->amplitude;
	delete[] rayleigh->shift;
	delete[] rayleigh->width;
	delete[] rayleigh->offset;
	delete[] rayleigh->sanity;
	delete rayleigh;

	delete[] stokes->amplitude;
	delete[] stokes->shift;
	delete[] stokes->width;
	delete[] stokes->offset;
	delete[] stokes->sanity;
	delete stokes ;

	delete[] standalone_a;
	delete[] standalone_b;
	delete[] standalone_c;

	delete[] standalone_frq_lut;

	delete[] standalone_start_ROI;
	delete[] standalone_end_ROI;

	delete[] standalone_summed_curves;
	delete[] sa_translation_lut;

	delete[] n_points;
	delete[] data_x;
	delete[] data_y;
	delete[] noise_deviation;

	delete[] sa_rayleigh->amplitude;
	delete[] sa_rayleigh->shift;
	delete[] sa_rayleigh->width;
	delete[] sa_rayleigh->offset;
	delete[] sa_rayleigh->sanity;
	delete sa_rayleigh;


	delete[] sa_antistokes->amplitude;
	delete[] sa_antistokes->shift;
	delete[] sa_antistokes->width;
	delete[] sa_antistokes->offset;
	delete[] sa_antistokes->sanity;
	delete sa_antistokes;

	delete[] sa_stokes->amplitude;
	delete[] sa_stokes->shift;
	delete[] sa_stokes->width;
	delete[] sa_stokes->offset;
	delete[] sa_stokes->sanity;
	delete sa_stokes;

	
}


/** 
Old test to make sure the spline code worked
*/
void debug_splines() {
	//Testing spline interpolation

	//Input
	int const points = 10;
	float x[points] = { -4,-3,-2,-1, 0, 1, 2, 3, 4 ,4.5 };
	float y[points] = { 10, 5, 2, 8, 9 };

	printf("data=[");
	for (int i = 0; i < points; i++) {
		//x[i] = rand() % 100;
		y[i] = rand() % 100;
		printf("%f %f \n", x[i], y[i]);
	}
	printf("]");

	//Output
	float a[points], b[points], c[points], d[points];
	Spline_Buffers spline_buffer;
	spline_buffer.A = new float[points];
	spline_buffer.l = new float[points];
	spline_buffer.h = new float[points];
	spline_buffer.u = new float[points];
	spline_buffer.z = new float[points];

	Functions::spline_coefficients(points, x, y, a, b, c, d, spline_buffer, 0);

	//Interpolation and displaying
	int n_points = 100;
	float* x_inter = new float[n_points];
	float* y_inter = new float[n_points];
	float start = x[0];
	int spline = 0;
	printf("spline=[");
	for (int i = 0; i < n_points; i++) {
		x_inter[i] = x[0] + i * (x[points - 1] - x[0]) / (n_points - 1.);
		if (x_inter[i] >= x[spline + 1])
			spline++;

		float spline_x = x_inter[i] - x[spline];
		y_inter[i] = d[spline] * spline_x * spline_x * spline_x
			+ c[spline] * spline_x * spline_x
			+ b[spline] * spline_x
			+ a[spline];

		printf("%f %d %f %f \n", x[spline], spline, x_inter[i], y_inter[i]);
	}
	printf("]");




	delete[] x_inter;
	delete[] y_inter;
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] spline_buffer.A;
	delete[] spline_buffer.l;
	delete[] spline_buffer.h;
	delete[] spline_buffer.u;
	delete[] spline_buffer.z;

	return;
}

void check_spline_fit() {
	//Testing spline interpolation

//Input
	int const points = 10;
	float x[points] = { -4,-3,-2,-1, 0, 1, 2, 3, 4 ,4.5 };
	float y[points] = { 10, 5, 2, 8, 9 };

	printf("data=[");
	for (int i = 0; i < points; i++) {
		//x[i] = rand() % 100;
		y[i] = rand() % 100;
		printf("%f %f \n", x[i], y[i]);
	}
	printf("]");

	//Output
	float a[points], b[points], c[points], d[points];
	Spline_Buffers spline_buffer;
	spline_buffer.A = new float[points];
	spline_buffer.l = new float[points];
	spline_buffer.h = new float[points];
	spline_buffer.u = new float[points];
	spline_buffer.z = new float[points];

	Functions::spline_coefficients(points, x, y, a, b, c, d, spline_buffer, 0);

	float a_2[points], b_2[points], c_2[points], d_2[points];
	Spline_Buffers spline_buffer_2;
	spline_buffer_2.A = new float[points];
	spline_buffer_2.l = new float[points];
	spline_buffer_2.h = new float[points];
	spline_buffer_2.u = new float[points];
	spline_buffer_2.z = new float[points];
	Functions::spline_coefficients_2(points, x, y, a_2, b_2, c_2, d_2, spline_buffer_2, 0);

	// Comparing output values
	printf("== a == \n");
	for (int i = 0; i < points; i++) {
		printf(" %f | %f | %f\% \n ", a[i] , a_2[i], (a[i] - a_2[i]) / a[i] * 100);
	}

	printf("== b == \n");
	for (int i = 0; i < points; i++) {
		printf(" %f | %f | %f\% \n ", b[i], b_2[i], (b[i] - b_2[i]) / b[i] * 100);
	}

	printf("== c == \n");
	for (int i = 0; i < points; i++) {
		printf(" %f | %f | %f\% \n ", c[i], c_2[i], (c[i] - c_2[i]) / c[i] * 100);
	}

	printf("== d == \n");
	for (int i = 0; i < points; i++) {
		printf(" %f | %f | %f\% \n ", d[i], d_2[i], (d[i] - d_2[i]) / d[i] * 100);
	}
	
	delete[] spline_buffer.A;
	delete[] spline_buffer.l;
	delete[] spline_buffer.h;
	delete[] spline_buffer.u;
	delete[] spline_buffer.z;

	delete[] spline_buffer_2.A;
	delete[] spline_buffer_2.l;
	delete[] spline_buffer_2.h;
	delete[] spline_buffer_2.u;
	delete[] spline_buffer_2.z;
}


/**
Testing if the fit of 2nd degree polynomials  using GPUfit works or not.
Create mock data in the form of a  2nd degree polynomials, passes it to
the custom GpuFit wrapper, does a bit of processing, and then gpufit
gets to do the fit and return the fitted parameters.
*/
void check_poly2_fit(int n_fits, int n_points) {
	
	printf("========== Testing the fit of 2nd degree polynomial ========== \r\n");
	
	printf("Creating mock data... \r\n");
	float *x = new float[n_fits * n_points];
	float *y = new float[n_fits * n_points];
	int *valid_data = new int[n_fits];

	float* a = new float[n_fits];
	float* b = new float[n_fits];
	float* c = new float[n_fits];

	for (int i = 0; i < n_fits; i++) {
		a[i] = (std::rand() - RAND_MAX / 2) / 100;
		b[i] = (std::rand() - RAND_MAX / 2) / 100;
		c[i] = (std::rand() - RAND_MAX / 2) / 100;
		valid_data[i] = n_points;
		for (int j = 0; j < n_points; j++) {
			x[i * n_points + j] = (std::rand() - RAND_MAX / 2) / 100.0;
			y[i * n_points + j] = Functions::poly2(x[i * n_points + j], a[i], b[i], c[i]) ;
		}
	}

	printf("Fitting... \r\n");
	float* fitted_a = new float[n_fits];
	float* fitted_b = new float[n_fits];
	float* fitted_c = new float[n_fits];

	Poly2Fitting poly(n_fits, n_points, 40, 1e-6);
	poly.fit(valid_data, x, y);
	poly.get_fitted_parameters(fitted_a, fitted_b, fitted_c);


	printf("Evaluating fit : \r\n");	
	for (int i = 0; i < n_fits; i++) {
		float err_a = abs(a[i] - fitted_a[i]) / a[i];
		float err_b = abs(b[i] - fitted_b[i]) / b[i];
		float err_c = abs(c[i] - fitted_c[i]) / c[i];
		printf("== Fit #%d == \r\n param\t| real\t| fitted\t| error \r\n", i);
		printf("a\t| %.1f\t| %.1f\t| %.4f %% \r\n", a[i], fitted_a[i], err_a*100);
		printf("b\t| %.1f\t| %.1f\t| %.4f %% \r\n", b[i], fitted_b[i], err_b*100);
		printf("c\t| %.1f\t| %.1f\t| %.4f %% \r\n", c[i], fitted_c[i], err_c*100);
	}

	//Cleaning memory
	delete[] x;
	delete[] y;
	delete[] valid_data;
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] fitted_a;
	delete[] fitted_b;
	delete[] fitted_c;

}

/**
Testing if the fit of lorentzian functions using GPUfit works or not.
Create mock data in the form of a lorentzian function, passes it to
the custom GpuFit wrapper, does a bit of processing, and then gpufit 
gets to do the fit and return the fitted parameters.
*/
void check_lorentzian_fit(int n_fits, int n_points) {
	printf("========== Testing the fit of Lorentzian function ========== \r\n");

	printf("Creating mock data... \r\n");
	float* x = new float[n_fits * n_points];
	float* y = new float[n_fits * n_points];
	int* valid_data = new int[n_fits];

	float* amplitude= new float[n_fits];
	float* gamma	= new float[n_fits];
	float* center	= new float[n_fits];
	float* offset	= new float[n_fits];

	for (int i = 0; i < n_fits; i++) {
		amplitude[i] = abs(std::rand() - RAND_MAX / 2) / 100;
		gamma[i] = abs(std::rand() - RAND_MAX / 2) / 100;
		center[i] = (std::rand() - RAND_MAX / 2) / 100;
		offset[i] = abs(std::rand() - RAND_MAX / 2) / 100;
		valid_data[i] = n_points;

		//We're sampling [center - n*gamma ; center + n*gamma ]
		int n = 5;
		for (int j = 0; j < n_points; j++) {
			x[i * n_points + j] = center[i] - n*gamma[i] + (2*n*gamma[i] / n_points * j) ;
			y[i * n_points + j] = 
				Functions::lorentzian(x[i * n_points + j], amplitude[i], center[i], gamma[i]) + offset[i] ;
		}
	}

	printf("Fitting... \r\n");
	float* fitted_amplitude = new float[n_fits];
	float* fitted_gamma		= new float[n_fits];
	float* fitted_center	= new float[n_fits];
	float* fitted_offset	= new float[n_fits];

	LorentzianFitting lorentzian(HOST, n_fits, 20, 1e-5, n_points, 5);
	lorentzian.get_initial_parameters(x, y, lorentzian.initial_parameters, HOST);
	lorentzian.determine_fitting_constraints(x, y);
	lorentzian.no_copy_fit(x, y, lorentzian.initial_parameters, NULL);
	lorentzian.export_fitted_parameters(fitted_amplitude, fitted_center, fitted_gamma, fitted_offset);


	printf("Evaluating fit : \r\n");
	for (int i = 0; i < n_fits; i++) {
		float err_ampl		= abs(amplitude[i] - abs(fitted_amplitude[i])) / amplitude[i];
		float err_gamma		= abs(gamma[i] - abs(fitted_gamma[i])) / gamma[i];
		float err_center	= abs(center[i] - fitted_center[i]) / center[i];
		float err_offset	= abs(offset[i] - fitted_offset[i]) / offset[i];
		printf("== Fit #%d == \r\n param\t| real\t| fitted\t| error \r\n", i);
		printf("ampl\t| %.1f\t| %.1f\t| %.4f %% \r\n", amplitude[i], fitted_amplitude[i], err_ampl * 100);
		printf("gamma\t| %.1f\t| %.1f\t| %.4f %% \r\n", gamma[i], fitted_gamma[i], err_gamma * 100);
		printf("center\t| %.1f\t| %.1f\t| %.4f %% \r\n", center[i], fitted_center[i], err_center * 100);
		printf("offset\t| %.1f\t| %.1f\t| %.4f %% \r\n", offset[i], fitted_offset[i], err_offset * 100);
	}

}
