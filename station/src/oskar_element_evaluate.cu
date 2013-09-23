/*
 * Copyright (c) 2012-2013, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <private_element.h>
#include <oskar_element.h>

#include <oskar_apply_element_taper_cosine.h>
#include <oskar_apply_element_taper_gaussian.h>
#include <oskar_evaluate_dipole_pattern.h>
#include <oskar_cuda_check_error.h>

#define PIf 3.14159265358979323846f
#define PI  3.14159265358979323846

#ifdef __cplusplus
extern "C" {
#endif

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_cudak_hor_lmn_to_modified_theta_phi_f(const int num_points,
        const float* l, const float* m, const float* n,
        const float delta_phi, float* theta, float* phi)
{
    float x, y, z, p;

    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    /* Get the data. */
    x = l[i];
    y = m[i];
    z = n[i];

    /* Cartesian to spherical (with orientation offset). */
    p = atan2f(y, x) + delta_phi;
    p = fmodf(p, 2.0f * PIf);
    x = sqrtf(x*x + y*y);
    y = atan2f(x, z); /* Theta. */
    if (p < 0) p += 2.0f * PIf; /* Get phi in range 0 to 2 pi. */
    phi[i] = p;
    theta[i] = y;
}

/* Double precision. */
__global__
void oskar_cudak_hor_lmn_to_modified_theta_phi_d(const int num_points,
        const double* l, const double* m, const double* n,
        const double delta_phi, double* theta, double* phi)
{
    double x, y, z, p;

    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    /* Get the data. */
    x = l[i];
    y = m[i];
    z = n[i];

    /* Cartesian to spherical (with orientation offset). */
    p = atan2(y, x) + delta_phi;
    p = fmod(p, 2.0 * PI);
    x = sqrt(x*x + y*y);
    y = atan2(x, z); /* Theta. */
    if (p < 0) p += 2.0 * PI; /* Get phi in range 0 to 2 pi. */
    phi[i] = p;
    theta[i] = y;
}

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_hor_lmn_to_modified_theta_phi_cuda_f(int num_points,
        const float* d_l, const float* d_m, const float* d_n,
        float delta_phi, float* d_theta, float* d_phi)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_cudak_hor_lmn_to_modified_theta_phi_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_l, d_m, d_n,
            delta_phi, d_theta, d_phi);
}

/* Double precision. */
void oskar_hor_lmn_to_modified_theta_phi_cuda_d(int num_points,
        const double* d_l, const double* d_m, const double* d_n,
        double delta_phi, double* d_theta, double* d_phi)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_cudak_hor_lmn_to_modified_theta_phi_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_l, d_m, d_n,
            delta_phi, d_theta, d_phi);
}

/* Wrapper. */
void oskar_hor_lmn_to_modified_theta_phi(oskar_Mem* theta, oskar_Mem* phi,
        double delta_phi, int num_points, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!theta || !phi || !l || !m || !n || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get data type and location. */
    type = oskar_mem_type(theta);
    location = oskar_mem_location(theta);

    /* Compute modified theta and phi coordinates. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_hor_lmn_to_modified_theta_phi_cuda_f(num_points,
                    ((const float*)l->data), ((const float*)m->data),
                    ((const float*)n->data), (float)delta_phi,
                    ((float*)theta->data), ((float*)phi->data));
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_hor_lmn_to_modified_theta_phi_cuda_d(num_points,
                    ((const double*)l->data), ((const double*)m->data),
                    ((const double*)n->data), delta_phi,
                    ((double*)theta->data), ((double*)phi->data));
            oskar_cuda_check_error(status);
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

void oskar_element_evaluate(const oskar_Element* model,
        oskar_Mem* output, double orientation_x, double orientation_y,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* n, oskar_Mem* theta, oskar_Mem* phi, int* status)
{
    int spline_x = 0, spline_y = 0, computed_angles = 0;

    /* Check all inputs. */
    if (!model || !output || !l || !m || !n || !theta || !phi || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if spline data is present for x or y dipole. */
    spline_x = oskar_splines_has_coeffs(model->theta_re_x) &&
            oskar_splines_has_coeffs(model->theta_im_x) &&
            oskar_splines_has_coeffs(model->phi_re_x) &&
            oskar_splines_has_coeffs(model->phi_im_x);
    spline_y = oskar_splines_has_coeffs(model->theta_re_y) &&
            oskar_splines_has_coeffs(model->theta_im_y) &&
            oskar_splines_has_coeffs(model->phi_re_y) &&
            oskar_splines_has_coeffs(model->phi_im_y);

    /* Check that the output array is complex. */
    if (!oskar_mem_is_complex(output))
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Ensure there is enough space in the theta and phi work arrays. */
    if ((int)oskar_mem_length(theta) < num_points)
        oskar_mem_realloc(theta, num_points, status);
    if ((int)oskar_mem_length(phi) < num_points)
        oskar_mem_realloc(phi, num_points, status);

    /* Resize output array if required. */
    if ((int)oskar_mem_length(output) < num_points)
        oskar_mem_realloc(output, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if element type is isotropic. */
    if (model->element_type == OSKAR_ELEMENT_TYPE_ISOTROPIC)
        oskar_mem_set_value_real(output, 1.0, status);

    /* Evaluate polarised response if output array is matrix type. */
    if (oskar_mem_is_matrix(output))
    {
        double delta_phi_x, delta_phi_y;

        /* Check if spline data present for dipole X. */
        if (spline_x)
        {
            /* Compute modified theta and phi coordinates for dipole X. */
            delta_phi_x = PI/2 - orientation_x;
            oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                    delta_phi_x, num_points, l, m, n, status);
            computed_angles = 1;

            /* Evaluate spline pattern for dipole X. */
            oskar_splines_evaluate(output, 0, 8, model->theta_re_x,
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 1, 8, model->theta_im_x,
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 2, 8, model->phi_re_x,
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 3, 8, model->phi_im_x,
                    num_points, theta, phi, status);
        }
        else if (model->element_type == OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE)
        {
            /* Compute modified theta and phi coordinates for dipole X. */
            delta_phi_x = orientation_x - PI/2; /* TODO check the order. */
            oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                    delta_phi_x, num_points, l, m, n, status);
            computed_angles = 1;

            /* Evaluate dipole pattern for dipole X. */
            oskar_evaluate_dipole_pattern(output, num_points, theta, phi, 1,
                    status);
        }

        /* Check if spline data present for dipole Y. */
        if (spline_y)
        {
            /* Compute modified theta and phi coordinates for dipole Y. */
            delta_phi_y = -orientation_y;
            oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                    delta_phi_y, num_points, l, m, n, status);
            computed_angles = 1;

            /* Evaluate spline pattern for dipole Y. */
            oskar_splines_evaluate(output, 4, 8, model->theta_re_y,
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 5, 8, model->theta_im_y,
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 6, 8, model->phi_re_y,
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 7, 8, model->phi_im_y,
                    num_points, theta, phi, status);
        }
        else if (model->element_type == OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE)
        {
            /* Compute modified theta and phi coordinates for dipole X. */
            delta_phi_y = orientation_y - PI/2; /* TODO check the order. */
            oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                    delta_phi_y, num_points, l, m, n, status);
            computed_angles = 1;

            /* Evaluate dipole pattern for dipole Y. */
            oskar_evaluate_dipole_pattern(output, num_points, theta, phi, 0,
                    status);
        }
    }

    /* Compute theta values for tapering, if not already done. */
    if (model->taper_type != OSKAR_ELEMENT_TAPER_NONE && !computed_angles)
    {
        oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                0, num_points, l, m, n, status);
        computed_angles = 1;
    }

    /* Apply element tapering, if specified. */
    if (model->taper_type == OSKAR_ELEMENT_TAPER_COSINE)
    {
        oskar_apply_element_taper_cosine(output, num_points,
                model->cos_power, theta, status);
    }
    else if (model->taper_type == OSKAR_ELEMENT_TAPER_GAUSSIAN)
    {
        oskar_apply_element_taper_gaussian(output, num_points,
                model->gaussian_fwhm_rad, theta, status);
    }
}

#ifdef __cplusplus
}
#endif
