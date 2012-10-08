/*
 * Copyright (c) 2012, The University of Oxford
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

#include "math/oskar_spline_data_evaluate.h"
#include "station/oskar_element_model_evaluate.h"
#include "station/oskar_evaluate_dipole_pattern.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_cuda_check_error.h"

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
    type = theta->type;
    location = theta->location;

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

int oskar_element_model_evaluate(const oskar_ElementModel* model, oskar_Mem* G,
        int use_polarised, double orientation_x, double orientation_y,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* n, oskar_Mem* theta, oskar_Mem* phi)
{
    int error = 0;

    /* Check that the output array is complex. */
    if (!oskar_mem_is_complex(G->type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that all arrays are on the GPU. */
    if (l->location != OSKAR_LOCATION_GPU ||
            m->location != OSKAR_LOCATION_GPU ||
            n->location != OSKAR_LOCATION_GPU ||
            G->location != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Ensure there is enough space in the theta and phi work arrays. */
    if (theta->num_elements < num_points)
    {
        oskar_mem_realloc(theta, num_points, &error);
        if (error) return error;
    }
    if (phi->num_elements < num_points)
    {
        oskar_mem_realloc(phi, num_points, &error);
        if (error) return error;
    }

    /* Resize output array if required. */
    if (G->num_elements < num_points)
    {
        oskar_mem_realloc(G, num_points, &error);
        if (error) return error;
    }

    /* Evaluate polarised response if output array is matrix type. */
    if (oskar_mem_is_matrix(G->type))
    {
        if (use_polarised)
        {
            double delta_phi_x, delta_phi_y;

            /* Check if spline data present for dipole X. */
            if (model->theta_re_x.coeff.data && model->theta_im_x.coeff.data &&
                    model->phi_re_x.coeff.data && model->phi_im_x.coeff.data)
            {
                /* Compute modified theta and phi coordinates for dipole X. */
                delta_phi_x = PI/2 - orientation_x;
                oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                        delta_phi_x, num_points, l, m, n, &error);

                /* Evaluate spline pattern for dipole X. */
                oskar_spline_data_evaluate(G, 0, 8, &model->theta_re_x,
                        num_points, theta, phi, &error);
                oskar_spline_data_evaluate(G, 1, 8, &model->theta_im_x,
                        num_points, theta, phi, &error);
                oskar_spline_data_evaluate(G, 2, 8, &model->phi_re_x,
                        num_points, theta, phi, &error);
                oskar_spline_data_evaluate(G, 3, 8, &model->phi_im_x,
                        num_points, theta, phi, &error);
                if (error) return error;
            }
            else
            {
                /* Compute modified theta and phi coordinates for dipole X. */
                delta_phi_x = orientation_x - PI/2; /* TODO check the order. */
                oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                        delta_phi_x, num_points, l, m, n, &error);

                /* Evaluate tapered dipole pattern for dipole X. */
                oskar_evaluate_dipole_pattern(G, num_points,
                        theta, phi, model->cos_power, model->gaussian_fwhm_rad,
                        1, &error);
                if (error) return error;
            }

            /* Check if spline data present for dipole Y. */
            if (model->theta_re_y.coeff.data && model->theta_im_y.coeff.data &&
                    model->phi_re_y.coeff.data && model->phi_im_y.coeff.data)
            {
                /* Compute modified theta and phi coordinates for dipole X. */
                delta_phi_y = -orientation_y;
                oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                        delta_phi_y, num_points, l, m, n, &error);

                /* Evaluate spline pattern for dipole Y. */
                oskar_spline_data_evaluate(G, 4, 8, &model->theta_re_y,
                        num_points, theta, phi, &error);
                oskar_spline_data_evaluate(G, 5, 8, &model->theta_im_y,
                        num_points, theta, phi, &error);
                oskar_spline_data_evaluate(G, 6, 8, &model->phi_re_y,
                        num_points, theta, phi, &error);
                oskar_spline_data_evaluate(G, 7, 8, &model->phi_im_y,
                        num_points, theta, phi, &error);
                if (error) return error;
            }
            else
            {
                /* Compute modified theta and phi coordinates for dipole X. */
                delta_phi_y = orientation_y - PI/2; /* TODO check the order. */
                oskar_hor_lmn_to_modified_theta_phi(theta, phi,
                        delta_phi_y, num_points, l, m, n, &error);

                /* Evaluate tapered dipole pattern for dipole Y. */
                oskar_evaluate_dipole_pattern(G, num_points,
                        theta, phi, model->cos_power, model->gaussian_fwhm_rad,
                        0, &error);
                if (error) return error;
            }
        }

        /* Don't use polarised element model. */
        else
        {
        }
    }

    /* Evaluate scalar response if output array is scalar type. */
    else if (oskar_mem_is_scalar(G->type))
    {
        /* Not yet implemented. */
        return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return 0;
}

#ifdef __cplusplus
}
#endif
