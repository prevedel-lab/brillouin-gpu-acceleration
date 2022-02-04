#ifndef GPUFIT_ANTI_STOKES_CUH_INCLUDED
#define GPUFIT_ANTI_STOKES_CUH_INCLUDED

/* Description of the calculate_anti_stokes function
* ==============================================
* Added by Sebastian Hambura the 06/2020
*
* This function calculates the values of one-dimensional antistokes model functions
* and their partial derivatives with respect to the model parameters.
*
* This function makes use of the user information data to pass in the
* independent variables (X values) corresponding to the data and the angle distribution
* used to do the integration.  The X values must be of type REAL.
*
*   user_info is structured as follows :
*       - an int to indicate the number of angle in the angle distribution
*       - the angle distribution as REALs
*       - the x-coordinates distribution : either none (default [0, 1, ..., n_fits -1] ), or 
*           one x-scale for all fits, or one x-scale per fit. The corect case is deduced by looking at
*           ( user_info_size - angle_distribution_n*sizeof(REAL) ) / sizeof(REAL)
* 
*
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: A = amplitude of the Brillouin peak
*             p[1]: s = shift (GHz) of the Brillouin peak
*             p[2]: w = intrinsic linewidth (GHz) of the Brillouin peak
*             p[3]: offset / background signal
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_gauss1d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_anti_stokes(
    REAL const* parameters,
    int const n_fits,
    int const n_points,
    REAL* value,
    REAL* derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char* user_info,
    std::size_t const user_info_size)
{
    /* Converting user_info to usefull data*/

    // Getting the angle distribution
    int n_angles = ((int*)user_info)[0];
    REAL* angle_distribution = (REAL*) (user_info  + sizeof(REAL));
    REAL* x_distribution = (REAL*)(user_info + sizeof(REAL) + n_angles * sizeof(REAL));

    //Getting x-coordinate
    size_t size_x_distrib = user_info_size - n_angles * sizeof(REAL);
    REAL x;
    if (size_x_distrib / sizeof(REAL) == n_points) { 

        //1 x-scale for all fits
        x = x_distribution[point_index];

    } 
    else if (size_x_distrib / sizeof(REAL) > n_points) {

        //1 x-scale per fit
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = x_distribution[chunk_begin + fit_begin + point_index];

    } 
    else {

        //default x-scale
        x = point_index;
    }

    /* parameters */
    REAL const * p = parameters;
    REAL A = p[0] ;
    REAL s = p[1] ;
    REAL w = p[2];
    REAL offset = p[3];

    /* Calculations */
    REAL psi;
    REAL val, deriv_s, deriv_w, deriv_A;
    REAL alpha, beta, gamma;
    val = 0;
    deriv_s = 0;
    deriv_w = 0;
    deriv_A = 0;
    for (int i = 0; i < n_angles; i++) {
        psi = angle_distribution[i] / 2;
        alpha = x - s * sin(psi);
        beta = w * sin(psi) * sin(psi);
        gamma = 2*alpha / beta;
        gamma = gamma * gamma;
        
        val += 1 / (1 + gamma) ;
        deriv_s += sin(psi) * alpha / ((1 + gamma) * (1 + gamma) * beta * beta) ;
        deriv_w += gamma * gamma / ((1 + gamma) * (1 + gamma)) ;
    }
    deriv_A = val ;
    val = val * A  + offset;
    deriv_s = 8 * A * deriv_s ;
    deriv_w = (2 * A / w) * deriv_w ;

    /* Normalisation */
    //deriv_A = deriv_A / n_angles;
    //val = val / n_angles;
    //deriv_s = deriv_s / n_angles;
    //deriv_w = deriv_w / n_angles;

    // value
    value[point_index] = val;

    // derivative
    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = deriv_A;
    current_derivative[1 * n_points] = deriv_s;
    current_derivative[2 * n_points] = deriv_w;
    current_derivative[3 * n_points] = 1;

}

#endif