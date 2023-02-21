/*****************************************************************//**
 * \file   DLL_struct.h
 * \brief  A File containing all the data structures used to communicate with the DLL.
 *
 * 
 * \author Sebastian Hambura
 * \date   August 2020
 *********************************************************************/
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

#pragma once

/**
* \defgroup structures I/O structure format
* \brief Structures and enumerations to easily communicate between MatLab and the library.
 */



/**  \brief An enum to be able to chose which mathematical model to use as fitting function.
 */
typedef enum {
	FIT_LORENTZIAN,		///< Lorentzian function
	FIT_GAUSSIAN,		///< Gaussian function
	FIT_STOKES,			///< Stokes function
	FIT_ANTISTOKES,		///< AntiStokes function
	FIT_PHASOR			///< Phasor analysis : not implemented yet

} Fitting_Function_t;


/** \brief An enum to be able to chose betwen linear and spline interpolation
*/
typedef enum {
	LINEAR,		// Linear interpolation
	SPLINE		// Natural spline interpolation
} Interpolation_type;

/** \brief A data structure containing all the information relating to the angle distribution and geometry of the setup
 * to compute the Stokes and the antiStokes functions.
 * 
 * \ingroup structures
 */
typedef struct {
	float NA_illum;					/**< NA of the illumination lens. In our setup, set it to 0.25. */
	float NA_coll;					/**< NA of the collection lens. In our setup, set it to 0.7. */ 
	int angle_distribution_n;		/**< Number of sums to simulate the integration over all angles. 
									1 000 is fine, 10 000 is slow. */
	float angle;					/**<Angle of the collection geometry (pi for epidetection, pi/2 for 90 degrees,...).
									In our setup, set it to PI / 2. */
	float geometrical_correction;	/**< Due to optical/geometrical considerations, there is a factor between the actual
									frequency and the shift used to compute the Stokes/antiStokes. Theoretically, it 
									should be sqrt(2) (=1.414...), but experimentally I found it to be closer to 1.27.

									Code extract from the anti_stokes function :
									\code 
									void StokesOrAntiStokesFitting::apply_fitting_constraints(float min_amp, float max_amp, float min_shift, float max_shift) {
										parameterContext->parameter_width(1, LOWER_UPPER, min_width, max_width);
										parameterContext->parameter_amplitude(1, LOWER_UPPER, min_amp / angle_distribution_n, max_amp / angle_distribution_n);

										float lower = (abs(min_shift) > abs(max_shift) ) ? abs(max_shift) : abs(min_shift) ;
										float upper = (abs(min_shift) < abs(max_shift)) ? abs(max_shift) : abs(min_shift) ;
										parameterContext->parameter_shift(1, LOWER_UPPER, lower * geometrical_correction , upper * geometrical_correction);
										parameterContext->parameter_offset(1, NONE, 0, 0);
									}
									\endcode
									*/
} Stokes_antiStokes_Context;


/** \brief A structure containing all information related to the function's fit on the data.
 * 
 * \ingroup structures
 */
typedef struct {
	int n_fits;				///< Number of fits per batch
	int max_iteration;		///< Maximum number of iterations
	float tolerance;		///< Tolerance criteria to validate the convergence of the fit. Recommended between 1e-4 and 1e-6
	float SNR_threshold;	///< The Signal-to-Noise Ratio above which the fit is considered insane.
} Fitting_Context;


/** \brief A structure containing all the parameters of a fitted peak function.
 * 
 *	Every array has a length of n_fits.
 * 
 *  Internally in this library and in GPUFit, the parameters are stored in a single array like this :
 *  [amplitude, shift, width, offset, amplitude, shit, width, offset, ...., width, offset ].
 *  
 * \ingroup structures
 */
typedef struct {
	float* amplitude;	///< The amplitude of the fitted function.
	float* shift;		///< The shift or center of the fitted function.
	float* width;		///< The width (or sigma) of the fitted function.
	float* offset;		///< The offset of the fitted function.

	int* sanity;		///< The sanity of the fit : 0 = insane, 1 = sane.
} Fitted_Function;


/** \brief A structure containing the different Goodness of fit parameters the fit can return.
 * 
 *	Every array has a length of n_fits.
 * 
 * \ingroup structures
 */
typedef struct {
	int* states;			/**< \brief The state as returned by GPUFit.
							
							For the states array, the following convention applies (from GPUfit library) : 
								*	- 0 :	The fit converged, tolerance is satisfied, the maximum number of iterations is not exceeded
								*	- 1 :	Maximum number of iterations exceeded
								*	- 2 :	During the Gauss-Jordan elimination the Hessian matrix is indicated as singular
								*	- 3 :	Non-positive curve values have been detected while using MLE (MLE requires only positive curve values)
								*	- 4 :	State not read from GPU Memory
							 
							States 3 and 4 should never occur.							
							*/
	float* chi_squares;		///< Last chi square value computed, returned by GPUFit.
	int* n_iterations;		///< The total number of iterations done for this fit, returned by GPUFit.
	float* SNR;				///< The exact value of the Signal-to-Noise ratio
} Goodness_Of_Fit;


/** \brief A structure containing the information to go from the image to the summed curve.
 *
 * \ingroup structures
 */
typedef struct {
	int starting_order;		/**< The first order which should be taken into account for the creation of the summed curve.
							Included. */		
							
	int ending_order;		/**< The last order which should be taken into account for the creation of the summed curve.
							Excluded. */					
							
	int max_n_peaks;		/**< Maximum number of Rayleigh peaks found in the image. 
							Works fine with 20. */

	int n_points;			/**< The number of points the summed curves has. 
							Should be a bit higher than the number of pixels between 2 Rayleigh peaks. 
							In our setup, works good with 60 (= 20 * 3) */

	float starting_freq;	/**< Starting frequency of the summed curve.
							In our setup, there is a 15GHz shift between each Rayleigh peak, so we want the 
							summed curve to be from -7.5GHz to 7.5GHz. Therefore set this to -7.5 */

	float step;				/**< Difference of frequency between to consecutive points in the summed curve.
							Calculate as follow : step = (ending_freq - starting_freq) / (n_points - 1) */
	Interpolation_type interpolation; /**< Type of interpolation : linear or spline
									 */
} Curve_Extraction_Context;


/** \brief A struct used to pass pointers to the various buffers needed by the spline interpolation to the GPU. 
*	You shouldn't have to use it for outside.
* 
*/
typedef struct {
	float* data_x;
	float* data_y;
	float* a;
	float* b;
	float* c;
	float* d;
	float* A;
	float* h;
	float* l;
	float* u;
	float* z;

} Spline_Buffers;
