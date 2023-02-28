/* Copyright (C) 2021  Sebastian Hambura
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

#include "SyntheticSignal.h"

SyntheticSignal::SyntheticSignal(int width, int height, int n_fit, int n_order) {
	
	//Caracteristics of the synthetic signal
	this->width = width;
	this->height = height;
	this->n_fit = n_fit;
	this->n_order = n_order;

	period_size = width / (n_order + 1);
	ROI_width_start = period_size / 2;
	ROI_width_end = width - period_size / 2;
	ROI_height_start = height / 2 - n_fit / 2;
	ROI_height_end = height/2 + (n_fit - n_fit / 2) ;

	//Allocate the different arrays
	frq_lut = new float[width * height];
	signal = new float[width * height];
	signal_16b = new uint16_t[width * height];
	rayleigh_pos = new float[n_fit*n_order];
	rayleigh_pos_remapped = new float[n_fit * n_order];
	y_pos = new int[n_fit];
	n_peaks = new int[height];

	// Put an initial value in the important arrays
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			signal[x + y * width] = 10; //non zero background
			frq_lut[x + y * width] = NAN ;
			n_peaks[y] = 0;
		} 
	}
}

SyntheticSignal::~SyntheticSignal() {
	// Free allocated memory
	delete[] frq_lut;
	delete[] signal;
	delete[] rayleigh_pos;
	delete[] y_pos;
	delete[] n_peaks;
	delete[] signal_16b;
	if (stokes_angle_distrib_used)
		delete[] stokes_angle_distrib;
}

// Creates a synthetic signal with a default non-linearity factor
void SyntheticSignal::generate_frq_lut(float starting_freq, float ending_freq) {
	generate_frq_lut(starting_freq, ending_freq, 1);

}

/* Computes the underlaying frq_lut for the synthetic signal. 
	Usually starting_freq = -7.5 and ending_freq = 7.5
	Nonlinearity increases the curvature of the signal.*/
void SyntheticSignal::generate_frq_lut(float starting_freq, float ending_freq, float nonlinearity) {

	/* NEW Method */
	int border = 50;
	float x_border = width * 0.12;

	int starting_order = 2;
	int construction_order = n_order + 1;	//first order can't be used
	float y_c = height / 2;					//center of the curvature
	float A = 500;							// spacing between the orders(more or less)
	float b = pow(height / 2,  2) / (A - (construction_order + starting_order) * (construction_order + starting_order)) + 1; // b >= (y - yc)² / (A - n²) | 120 = > impact y has on the whole = > +/ -curvature
	b *= 1/nonlinearity;

	float min_x = sqrt(A - (construction_order + starting_order) * (construction_order + starting_order) - (height / 2) * (height / 2) / (b * b));
	float max_x = sqrt(A - starting_order * starting_order);
	float x_c = x_border / 2;
	float a = (width - x_border - x_c) / (max_x - min_x);


	//create and map Rayleigh peaks
	int* peak_numbers = new int[n_fit];
	float* fit_a = new float[n_fit];
	float* fit_b = new float[n_fit];
	float* fit_c = new float[n_fit];
	float* temp_rayleigh_pos = new float[construction_order * n_fit];
	float* temp_rayleigh_pos_remapped = new float[construction_order * n_fit];
	


	for (int y = ROI_height_start; y < ROI_height_end; y++) {
		int fit = y - ROI_height_start;
		for (int n = 0; n < construction_order; n++) {
			float x = abs(a) * (sqrt(A - (n+ starting_order) * (n+ starting_order) - (y - y_c) * (y - y_c) / (b * b)) - min_x) + x_c;
			temp_rayleigh_pos[fit * construction_order + (construction_order - n - 1)] = x;

		}

		//Remap Rayleigh peak
		float step = (temp_rayleigh_pos[fit * construction_order + construction_order - 1] - temp_rayleigh_pos[fit * construction_order + 0]) / (construction_order - 1);

		for (int n = 0; n < construction_order; n++) {
			temp_rayleigh_pos_remapped[fit * construction_order + n] = temp_rayleigh_pos[fit * construction_order + 0] + n * step;
		}

		peak_numbers[fit] = construction_order;

	}


	//Fit poly 2 to the Rayleigh peaks
	Poly2Fitting poly(n_fit, construction_order, 20, 1e-5);
	poly.fit(peak_numbers, temp_rayleigh_pos, temp_rayleigh_pos_remapped);
	poly.get_fitted_parameters(fit_a, fit_b , fit_c );

	//Change the original position to the inverse of the remapped rayleigh peak
	for (int y = 0; y < n_fit; y++) {
		for (int n = 0; n < construction_order; n++) {
			temp_rayleigh_pos[y * construction_order + n] = Functions::from_linear_to_original_space(temp_rayleigh_pos_remapped[y * construction_order + n], fit_a[y], fit_b[y], fit_c[y]);
			
		}
	}

	
	//Computes the frq for all the pixels inside the ROI
	for (int y = ROI_height_start; y < ROI_height_end; y++) {
		int fit = y - ROI_height_start;
		float step = (temp_rayleigh_pos_remapped[fit * construction_order + construction_order - 1] - temp_rayleigh_pos_remapped[fit * construction_order + 0]) / (construction_order - 1);
		int start_column = floor(Functions::from_linear_to_original_space(temp_rayleigh_pos_remapped[fit * construction_order + 1] - step/2, fit_a[fit], fit_b[fit], fit_c[fit])); //Todo
		int end_column = ceil(Functions::from_linear_to_original_space(temp_rayleigh_pos_remapped[fit * construction_order + construction_order - 1] + step / 2, fit_a[fit], fit_b[fit], fit_c[fit]));

		for (int x = start_column; x < end_column; x++) {
			float x_remap = (fit_a[fit] * x + fit_b[fit]) * x + fit_c[fit];
			float rayleigh_pos_linearspace = ((fit_a[fit] * temp_rayleigh_pos[fit * construction_order + 0] + fit_b[fit]) * temp_rayleigh_pos[fit * construction_order + 0] + fit_c[fit]);
			float dist = x_remap - rayleigh_pos_linearspace;
			float pos = fmod(dist, step);

			float frq = pos / step;
			if (frq > 0.5)
				frq -= 1;
			if (frq < -0.5)
				frq += 1;

			frq += 0.5;
			frq = (ending_freq - starting_freq) * frq + starting_freq;

			frq_lut[x + y * width] = frq;

		}

	}

	

	// Set the data for the next processing steps
	// the first order can't be used because it's too close to the edge
	for (int y = 0; y < n_fit; y++)
		for (int n = 0; n < n_order; n++) {
			rayleigh_pos[y * n_order + n] = temp_rayleigh_pos[y * construction_order + n + 1];
			y_pos[y] = y + ROI_height_start;
		}
		

	//for (int y = 0; y < n_fit; y++)
	//	for (int n = 0; n < n_order; n++)
	//		frq_lut[(y+ ROI_height_start) * width + int(round(rayleigh_pos[y * n_order + n]))] = 20.1;

	//Free memory
	delete[] peak_numbers;
	delete[] fit_a;
	delete[] fit_b;
	delete[] fit_c;
	delete[] temp_rayleigh_pos_remapped;
	delete[] temp_rayleigh_pos;

	Hz_per_pixel = (ending_freq - starting_freq) / period_size;
}

/* Applies the choosen function on the synthetic signal.
	Uses the underlaying frq_lut to determine the frq of each pixel.
	For stokes and antiStokes functions, a context has to be given previous to calling this function.*/
void SyntheticSignal::generate_signal(Fitting_Function_t function, float amplitude, float shift, float width, float offset, int line) {
	int y = line;
	int pixel;
	for (int x = ROI_width_start; x < ROI_width_end; x++) {
		pixel = x + y * this->width;
		if (!isnan(frq_lut[pixel]))
		switch (function)
		{
		case FIT_LORENTZIAN :
			signal[pixel] += Functions::lorentzian(frq_lut[pixel], amplitude, shift, width) + offset;
			break;
		case FIT_ANTISTOKES:
			signal[pixel] += Functions::anti_stokes(frq_lut[pixel], amplitude, shift, width, stokes_angle_distrib, stokes_angle_n) + offset;
			break;
		case FIT_STOKES:
			signal[pixel] += Functions::stokes(frq_lut[pixel], amplitude, shift, width, stokes_angle_distrib, stokes_angle_n) + offset;
			break;
		default:
			signal[pixel] = 0; //Do nothing, but maybe we should throw an error here
			break;
		}
		
	}
}

void SyntheticSignal::generate_signal(Fitting_Function_t function, float amplitude, float shift, float width, float offset) {
	for (int y = ROI_height_start; y < ROI_height_end; y++) {
		generate_signal(function, amplitude, shift, width, offset, y);
	}
}

void SyntheticSignal::generate_signal(Fitting_Function_t function, Fitted_Function function_parameters) {
	int n = 0;
	for (int y = ROI_height_start; y < ROI_height_end; y++) {
		generate_signal(function, function_parameters.amplitude[n], function_parameters.shift[n],
			function_parameters.width[n], function_parameters.offset[n], y);
		n++;
	}
}

/* (Re)sets the synthetic signal value to the given value.*/
void SyntheticSignal::set_signal_value(float val)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			signal[x + y * width] = val;

		}
	}
}

void SyntheticSignal::set_stokes_antistokes_context(Stokes_antiStokes_Context* context) {
	stokes_angle_n = context->angle_distribution_n;
	stokes_angle_distrib = new float[stokes_angle_n];
	StokesOrAntiStokesFitting::init_angle_distribution(context->NA_illum, context->NA_coll, context->angle, context->angle_distribution_n, stokes_angle_distrib);
	stokes_angle_distrib_used = true;
}

//Converts the synthetic signal into 16-bit image, and applies a transposition
uint16_t* SyntheticSignal::get_transposed_signal_uint16(uint16_t max_intensity){
	float max = *std::max_element(signal, signal + width * height);
	float min = std::min(*std::min_element(signal, signal + width * height), (float) 0);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			signal_16b[y + x * height] = (signal[x + y * width] - min) / (max - min) * max_intensity;
			//signal_16b[x + y * width] = signal[x + y * width] / max * max_intensity;

		}
	}
	return signal_16b;
}

// Prints the ypos array to the console
void SyntheticSignal::display_ypos(){
	printf("Y_pos :\n");
	for (int n = 0; n < n_fit; n++) {
		printf("%d ", y_pos[n]);
	}
	printf("\n");
}

// Prints the rayleigh pos array to the console
void SyntheticSignal::display_rayleigh_pos() {
	printf("Rayleigh pos :\n");
	for (int n = 0; n < n_fit; n++) {
		for (int i = 0; i < n_order; i++) {
			printf("%f ", rayleigh_pos[i + n*n_order]);
		}
		printf("\n");
		
	}
	printf("\n");
}

// Prints the signal to the console
void SyntheticSignal::display_uint16_t_value(){
	printf("uint16 signal:\n");
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf("%d ", signal_16b[y + x * height]);
		}
		printf("\n");
	}
}

// Enables the visualisation of the ROI
void SyntheticSignal::draw_ROI() {
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			if (x == ROI_width_start || x == ROI_width_end || y == ROI_height_start || y == ROI_height_end) {
				signal[x + y * width] = 20;
				frq_lut[x + y * width] = 20;
			}
		}
	}
}
