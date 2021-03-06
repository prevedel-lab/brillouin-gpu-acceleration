#ifndef GPU_FIT_H_INCLUDED
#define GPU_FIT_H_INCLUDED

#ifdef __linux__
#define VISIBLE __attribute__((visibility("default")))
#endif

#ifdef _WIN32
#define VISIBLE
#endif

#include <cstddef>
#include <stdexcept>
#include "constants.h"
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

VISIBLE int gpufit
(
    size_t n_fits,
    size_t n_points,
    REAL * data,
    REAL * weights,
    int model_id,
    REAL * initial_parameters,
    REAL tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    size_t user_info_size,
    char * user_info,
    REAL * output_parameters,
    int * output_states,
    REAL * output_chi_squares,
    int * output_n_iterations
) ;

VISIBLE int gpufit_constrained
(
    size_t n_fits,
    size_t n_points,
    REAL * data,
    REAL * weights,
    int model_id,
    REAL * initial_parameters,
    REAL * constraints,
    int * constraint_types,
    REAL tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    size_t user_info_size,
    char * user_info,
    REAL * output_parameters,
    int * output_states,
    REAL * output_chi_squares,
    int * output_n_iterations
);

VISIBLE int gpufit_cuda_interface
(
    size_t n_fits,
    size_t n_points,
    REAL * gpu_data,
    REAL * gpu_weights,
    int model_id,
    REAL tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    size_t user_info_size,
    char * gpu_user_info,
    REAL * gpu_fit_parameters,
    int * gpu_output_states,
    REAL * gpu_output_chi_squares,
    int * gpu_output_n_iterations
);

VISIBLE int gpufit_cuda_interface_constrained
(
    size_t n_fits,
    size_t n_points,
    REAL* gpu_data,
    REAL* gpu_weights,
    int model_id,
    REAL tolerance,
    int max_n_iterations,
    int* parameters_to_fit,
    REAL* constraints,
    int* constraint_types,
    int estimator_id,
    size_t user_info_size,
    char* gpu_user_info,
    REAL* gpu_fit_parameters,
    int* gpu_output_states,
    REAL* gpu_output_chi_squares,
    int* gpu_output_n_iterations
);

VISIBLE char const * gpufit_get_last_error() ;

// returns 1 if cuda is available and 0 otherwise
VISIBLE int gpufit_cuda_available();

VISIBLE int gpufit_get_cuda_version(int * runtime_version, int * driver_version);

VISIBLE int gpufit_portable_interface(int argc, void *argv[]);

VISIBLE int gpufit_constrained_portable_interface(int argc, void *argv[]);

#ifdef __cplusplus
}
#endif

#endif // GPU_FIT_H_INCLUDED
