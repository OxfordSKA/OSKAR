/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_SKY_MODEL_H_
#define OSKAR_SKY_MODEL_H_

/**
 * @file oskar_SkyModel.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"


/**
 * @struct oskar_SkyModel
 *
 * @brief Structure to hold a sky model used by the OSKAR simulator.
 *
 * @details
 * The structure holds source parameters for the global sky model used by the
 * OSKAR simulator.
 */
#ifdef __cplusplus
extern "C"
#endif
struct oskar_SkyModel
{
    int num_sources;          /**< Number of sources in the sky model. */
    oskar_Mem RA;             /**< Right ascension, in radians. */
    oskar_Mem Dec;            /**< Declination, in radians. */
    oskar_Mem I;              /**< Stokes-I, in Jy. */
    oskar_Mem Q;              /**< Stokes-Q, in Jy. */
    oskar_Mem U;              /**< Stokes-U, in Jy. */
    oskar_Mem V;              /**< Stokes-V, in Jy. */
    oskar_Mem reference_freq; /**< Reference frequency for the source flux, in Hz. */
    oskar_Mem spectral_index; /**< Spectral index. */
    oskar_Mem rel_l;          /**< Phase centre relative direction-cosines. */
    oskar_Mem rel_m;          /**< Phase centre relative direction-cosines. */
    oskar_Mem rel_n;          /**< Phase centre relative direction-cosines. */

#ifdef __cplusplus
    /**
     * @brief Constructs and allocates memory for a sky model.
     *
     * @param type         Data type for the sky model (either OSKAR_SINGLE,
     *                     or OSKAR_DOUBLE)
     * @param location     Memory location of the sky model (either
     *                     OSKAR_LOCATION_CPU or OSKAR_LOCATION_GPU)
     * @param num_sources  Number of sources in the sky model.
     */
    oskar_SkyModel(int type, int location, int num_sources = 0);

    /**
     * @brief Constructs a sky model, loading it from the specified file.
     *
     * @details
     * Loads the sky model from a OSKAR source text file.
     *
     * @param filename     Path to a file containing an OSKAR sky model.
     * @param type         Data type for the sky model (either OSKAR_SINGLE,
     *                     or OSKAR_DOUBLE)
     * @param location     Memory location of the constructed sky model
     *                     (either OSKAR_LOCATION_CPU or OSKAR_LOCATION_GPU)
     */
    oskar_SkyModel(const char* filename, int type, int location);

    /**
     * @brief Constructs a sky model by copying the supplied sky
     * model to the specified location.
     *
     * @param sky          oskar_SkyModel to copy.
     * @param location     Memory location of the constructed sky model
     *                     (either OSKAR_LOCATION_CPU or OSKAR_LOCATION_GPU)
     */
    oskar_SkyModel(const oskar_SkyModel* other, int location);

    /**
     * @brief Destroys the sky model.
     */
    ~oskar_SkyModel();

    /**
     * @brief Appends the specified sky model the the current sky model.
     *
     * @param other Sky model to append.
     *
     * @return error code.
     */
    int append(const oskar_SkyModel* other);

    /**
     * @brief Computes the source l,m,n direction cosines relative to phase
     * centre.
     *
     * @details
     * Assumes that the RA and Dec arrays have already been populated, and
     * that the data is on the GPU.
     *
     * @param[in] ra0 Right Ascension of phase centre, in radians.
     * @param[in] dec0 Declination of phase centre, in radians.
     *
     * @return error code.
     */
    int compute_relative_lmn(double ra0, double dec0);

    /**
     * @brief Loads an OSKAR source text file into the current sky structure.
     * Sources from the file are appended to the end of the current structure.
     *
     * @param filename  Path to a file containing an oskar sky model.
     *
     * @return error code.
     */
    int load(const char* filename);

    /**
     * @brief Resizes the sky model the specified number of sources.
     *
     * @param num_sources Number of sources to resize to.
     *
     * @return error code.
     */
    int resize(int num_sources);

    /**
     * @brief Sets the parameters for the source at the specified
     * index in the sky model.
     *
     * @param index             Source index.
     * @param ra                Right ascension, in radians.
     * @param dec               Declination, in radians.
     * @param I                 Stokes-I, in Jy.
     * @param Q                 Stokes-Q, in Jy.
     * @param U                 Stokes-U, in Jy.
     * @param V                 Stokes-V, in Jy.
     * @param ref_frequency     Reference frequency, in Hz.
     * @param spectral_index    Spectral index.
     *
     * @return error code.
     */
    int set_source(int index, double ra, double dec, double I, double Q,
            double U, double V, double ref_frequency, double spectral_index);

    /**
     * @brief Returns the memory type for memory in the sky structure
     * or error code if the types are inconsistent.
     */
    int type() const;

    /**
     * @brief Returns the memory location for memory in the sky structure
     * or error code if the types are inconsistent.
     */
    int location() const;

    /**
     * @brief Returns true if the memory in the sky structure is
     * double precision.
     */
    bool is_double() const;
#endif
};
typedef struct oskar_SkyModel oskar_SkyModel;









#ifdef __cplusplus
extern "C" {
#endif

/* DEPRECATED */
struct oskar_SkyModelGlobal_f
{
    int num_sources;

    float* RA;             /**< The source Right Ascensions, in radians. */
    float* Dec;            /**< The source Declinations, in radians. */

    float* I;              /**< The source equatorial Stokes I values, in Jy. */
    float* Q;              /**< The source equatorial Stokes Q values, in Jy. */
    float* U;              /**< The source equatorial Stokes U values, in Jy. */
    float* V;              /**< The source equatorial Stokes V values, in Jy. */

    float* reference_freq; /**< The reference frequency for the source brightness, in Hz. */
    float* spectral_index; /**< The source spectral index. */

    float* rel_l;          /**< Source l-direction-cosines relative to the phase centre. */
    float* rel_m;          /**< Source m-direction-cosines relative to the phase centre. */
    float* rel_n;          /**< Source n-direction-cosines relative to the phase centre. */
};
typedef struct oskar_SkyModelGlobal_f oskar_SkyModelGlobal_f;

/* DEPRECATED */
struct oskar_SkyModelGlobal_d
{
    int num_sources;

    double* RA;             /**< The source Right Ascensions, in radians. */
    double* Dec;            /**< The source Declinations, in radians. */

    double* I;              /**< The source equatorial Stokes I values, in Jy. */
    double* Q;              /**< The source equatorial Stokes Q values, in Jy. */
    double* U;              /**< The source equatorial Stokes U values, in Jy. */
    double* V;              /**< The source equatorial Stokes V values, in Jy. */

    double* reference_freq; /**< The reference frequency for the source brightness, in Hz. */
    double* spectral_index; /**< The source spectral index. */

    double* rel_l;          /**< Source l-direction-cosines relative to the phase centre. */
    double* rel_m;          /**< Source m-direction-cosines relative to the phase centre. */
    double* rel_n;          /**< Source n-direction-cosines relative to the phase centre. */
};
typedef struct oskar_SkyModelGlobal_d oskar_SkyModelGlobal_d;

/* DEPRECATED */
struct oskar_SkyModelLocal_f
{
    int num_sources;

    float* RA;          /**< Source Right Ascensions, in radians. */
    float* Dec;         /**< Source Declinations, in radians. */

    float* I;           /**< Source horizontal Stokes I values, in Jy. */
    float* Q;           /**< Source horizontal Stokes Q values, in Jy. */
    float* U;           /**< Source horizontal Stokes U values, in Jy. */
    float* V;           /**< Source horizontal Stokes V values, in Jy. */

    float* hor_l;       /**< Source horizontal l-direction-cosines. */
    float* hor_m;       /**< Source horizontal m-direction-cosines. */
    float* hor_n;       /**< Source horizontal n-direction-cosines. */

    float* rel_l;      /**< Source l-direction-cosines relative to the phase centre. */
    float* rel_m;      /**< Source m-direction-cosines relative to the phase centre. */
    float* rel_n;      /**< Source n-direction-cosines relative to the phase centre. */
};
typedef struct oskar_SkyModelLocal_f oskar_SkyModelLocal_f;

/* DEPRECATED */
struct oskar_SkyModelLocal_d
{
    int num_sources;

    double* RA;         /**< Source Right Ascensions in radians. */
    double* Dec;        /**< Source Declinations in radians. */

    double* I;          /**< Source horizontal Stokes I values, in Jy. */
    double* Q;          /**< Source horizontal Stokes Q values, in Jy. */
    double* U;          /**< Source horizontal Stokes U values, in Jy. */
    double* V;          /**< Source horizontal Stokes V values, in Jy. */

    double* hor_l;      /**< Source horizontal l-direction-cosines. */
    double* hor_m;      /**< Source horizontal m-direction-cosines. */
    double* hor_n;      /**< Source horizontal n-direction-cosines. */

    double* rel_l;      /**< Source l-direction-cosines relative to the phase centre. */
    double* rel_m;      /**< Source m-direction-cosines relative to the phase centre. */
    double* rel_n;      /**< Source n-direction-cosines relative to the phase centre. */
};
typedef struct oskar_SkyModelLocal_d oskar_SkyModelLocal_d;


OSKAR_EXPORT
void oskar_sky_model_global_copy_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
        oskar_SkyModelGlobal_d* hd_sky);

OSKAR_EXPORT
void oskar_sky_model_global_copy_to_gpu_f(const oskar_SkyModelGlobal_f* h_sky,
        oskar_SkyModelGlobal_f* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_allocate_gpu_d(const int num_sources,
        oskar_SkyModelLocal_d* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_allocate_gpu_f(const int num_sources,
        oskar_SkyModelLocal_f* hd_sky);

OSKAR_EXPORT
void oskar_global_sky_model_free_gpu_d(oskar_SkyModelGlobal_d* hd_sky);

OSKAR_EXPORT
void oskar_global_sky_model_free_gpu_f(oskar_SkyModelGlobal_f* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_free_gpu_d(oskar_SkyModelLocal_d* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_free_gpu_f(oskar_SkyModelLocal_f* hd_sky);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_MODEL_H_ */
