#ifndef GPUFIT_CAUCHY_LORENTZ_1D_CUH_INCLUDED
#define GPUFIT_CAUCHY_LORENTZ_1D_CUH_INCLUDED

/* Description of the calculate_cauchy_lorentz_1d function
* ===================================================
* Added by Sebastian Hambura the 06/2020
*
* This function calculates the values of the Cauchy distribution model functions
* (also known as Lorentz distribution, Cauchy�Lorentz distribution or Lorentz(ian) function)
* and their partial derivatives with respect to the model parameters.
* https://en.wikipedia.org/wiki/Cauchy_distribution
* The function isn't expected to be normalized nor of 0-offset : f(x) =  A / (1 + [(x-x0)/gamma]**2) + offset
*
* This function makes use of the user information data to pass in the
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* Note that if no user information is provided, the (X) coordinate of the
* first data value is assumed to be (0.0).  In this case, for a fit size of
* M data points, the (X) coordinates of the data are simply the corresponding
* array index values of the data array, starting from zero.
*
* There are three possibilities regarding the X values:
*
*   No X values provided:
*
*       If no user information is provided, the (X) coordinate of the
*       first data value is assumed to be (0.0).  In this case, for a
*       fit size of M data points, the (X) coordinates of the data are
*       simply the corresponding array index values of the data array,
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then
*       the same X values will be used for all fits.  In this case, the
*       size of the user_info array (in bytes) must equal
*       sizeof(REAL) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array
*       (in bytes) must equal sizeof(REAL) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters 
*             p[0]: amplitude 
*             p[1]: x0 
*             p[2]: gamma
*             p[3]: offset
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_linear1d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_cauchy_lorentz_1d(
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
    // indices

    REAL* user_info_float = (REAL*)user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters
    REAL A = parameters[0];
    REAL x0 = parameters[1];
    REAL gamma = parameters[2];
    REAL offset = parameters[3];

    // value
    REAL denominator = gamma * gamma + (x - x0) * (x - x0);
    value[point_index] = A * gamma * gamma / denominator + offset;

    // derivatives
    REAL squarred_denominator = denominator * denominator;
    REAL* current_derivatives = derivative + point_index;

    current_derivatives[0 * n_points] = gamma * gamma / denominator;                                // derivative A
    current_derivatives[1 * n_points] = 2 * A * gamma * gamma * (x - x0) / squarred_denominator;    // derivative x0
    current_derivatives[2 * n_points] = 2 * A * gamma * (x - x0) * (x - x0) / squarred_denominator; // derivative gamma
    current_derivatives[3 * n_points] = 1;
}

#endif
