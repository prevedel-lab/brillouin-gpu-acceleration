/*****************************************************************//**
 * \file   SyntheticSignal.h
 * \brief  Creates a synthetic Brillouin image
 *
 * Creates a Synthetic Brillouin signal.
 * First creates a frq LUT table, with a given curvature. 
 * Then we can add the function we want on top of that frq : 
 * for example, a lorentzian centered at 0GHz, and 2 other lorentzian centered 
 * at -5GHz and 5GHz.
 * This gives a Brillouin image, without noise, which can be processed by the 
 * library.
 *
 * \author Sebastian
 * \date   November 2021
 *********************************************************************/

#pragma once
#include "../DLL_struct.h"
#include "../cuda/kernel.cuh" // to get the defintion of the math functions
#include "../gpufit_wrapper/PeakFitting.h"
#include "../gpufit_wrapper/FunctionFitting.h" 

class SyntheticSignal
{
public:
	//Dimension of image
	int width;
	int height;

	//The Region of interest, where the actual signal gets generated
	int ROI_width_start;
	int ROI_width_end;
	int ROI_height_start;
	int ROI_height_end;

	//Array of size width*height containing the data
	float* frq_lut;	
	float* signal;
	uint16_t* signal_16b;

	//Caracterisation of the signal
	int period_size;	
	float Hz_per_pixel;

	int n_fit;		// number of fits in the image
	int n_order;	// number of orders in the signal

	// Different array containing information about the position of the rayleigh peak in the synthetic signal
	float* rayleigh_pos;
	float* rayleigh_pos_remapped;
	int* y_pos;
	int* n_peaks;

	// In case the Stokes and antiStokes peak are to be generated with a broadened Brillouin lineshape. Not tested for some time.
	bool stokes_angle_distrib_used = false;
	float* stokes_angle_distrib;
	float stokes_angle_n;

	
public:
	SyntheticSignal(int width, int height, int n_fit, int n_order);	
	~SyntheticSignal();


	/* Generate frq lut */

	void generate_frq_lut(float starting_freq, float ending_freq);
	void generate_frq_lut(float starting_freq, float ending_freq, float nonlinearity);


	/* Generate signal */

	void generate_signal(Fitting_Function_t function,  float amplitude, float shift, float width, float offset, int line);	
	void generate_signal(Fitting_Function_t function, float amplitude, float shift, float width, float offset);
	void generate_signal(Fitting_Function_t function, Fitted_Function function_parameters);
	void set_signal_value(float val);
	void set_stokes_antistokes_context(Stokes_antiStokes_Context* context);


	/* output */

	uint16_t* get_transposed_signal_uint16(uint16_t max_intensity);
	float* get_signal() { return signal; };
	float* get_frq_lut() { return frq_lut; };
	int get_width() { return width; };
	int get_height() { return height; };
	int get_n_order() { return n_order; };
	int* get_n_peaks() { return n_peaks; };
	

	/* Displaying function : usefull to forward to Matlab by copy-pasting the output into Matlab's console */

	void display_ypos();
	void display_rayleigh_pos();
	void display_uint16_t_value();


	/* Aux */

	void draw_ROI();
};

