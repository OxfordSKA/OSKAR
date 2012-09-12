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

#ifndef OSKAR_SKY_MODEL_H_
#define OSKAR_SKY_MODEL_H_

/**
 * @file oskar_SkyModel.h
 */

#include "oskar_global.h"
#ifdef __cplusplus
#include "utility/oskar_Log.h"
#endif
#include "utility/oskar_Mem.h"
#include <stdlib.h>

/**
 * @struct oskar_SkyModel
 *
 * @brief Structure to hold a sky model used by the OSKAR simulator.
 *
 * @details
 * The structure holds source parameters for the global sky model used by the
 * OSKAR simulator.
 */
struct OSKAR_EXPORT oskar_SkyModel
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

    int use_extended;         /**< Enable use of extended sources */
    oskar_Mem FWHM_major;     /**< Major axis FWHM for gaussian sources */
    oskar_Mem FWHM_minor;     /**< Minor axis FWHM for gaussian sources */
    oskar_Mem position_angle; /**< Position angle for gaussian sources */
    oskar_Mem gaussian_a;     /**< Gaussian source width parameter */
    oskar_Mem gaussian_b;     /**< Gaussian source width parameter */
    oskar_Mem gaussian_c;     /**< Gaussian source width parameter */

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
    oskar_SkyModel(int type = OSKAR_DOUBLE, int location = OSKAR_LOCATION_CPU,
            int num_sources = 0);

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
     * @brief Filter sources by flux.
     *
     * @param min_I Minimum value of Stokes I.
     * @param max_I Maximum value of Stokes I.
     *
     * @return An error code.
     */
    int filter_by_flux(double min_I, double max_I);

    /**
     * @brief This function removes sources from a sky model that lie within
     * \p inner_radius or beyond \p outer_radius.
     *
     * @param[out] sky Pointer to sky model.
     * @param[in] inner_radius Inner radius in radians.
     * @param[in] outer_radius Outer radius in radians.
     * @param[in] ra0 Right ascension of the phase centre in radians.
     * @param[in] dec0 Declination of the phase centre in radians.
     *
     * @return An error code.
     */
    int filter_by_radius(double inner_radius, double outer_radius,
            double ra0, double dec0);

    /**
     * @brief Loads an OSKAR source text file into the current sky structure.
     * Sources from the file are appended to the end of the current structure.
     *
     * @param filename  Path to a file containing an OSKAR sky model.
     *
     * @return An error code.
     */
    int load(const char* filename);

    /**
     * @brief Loads a GSM text file into the current sky structure.
     * Pixels from the file are appended to the end of the current structure.
     *
     * @param filename  Path to a GSM file.
     *
     * @return An error code.
     */
    int load_gsm(oskar_Log* log, const char* filename);

    /**
     * @brief Returns the memory location for memory in the sky structure
     * or error code if the types are inconsistent.
     */
    int location() const;

    /**
     * @brief
     * Scales all current source brightnesses according to the spectral index
     * for the given frequency.
     *
     * @details
     * This function scales all the existing source brightnesses using the
     * spectral index and the given frequency.
     *
     * Note that the scaling is done relative to the current brightness, and
     * should therefore be performed only on a temporary copy of the original
     * sky model.
     *
     * @param[in] frequency The frequency, in Hz.
     *
     * @return An OSKAR or CUDA error code.
     */
    int scale_by_spectral_index(double frequency);

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
     * @param FWHM_major        Major axis FWHM, radians.
     * @param FWHM_minor        Minor axis FWHM, radians.
     * @param position_angle    Position angle, radians.
     *
     * @return error code.
     */
    int set_source(int index, double ra, double dec, double I, double Q,
            double U, double V, double ref_frequency, double spectral_index,
            double FWHM_major = 0.0, double FWHM_minor = 0.0,
            double position_angle = 0.0);

    /**
     * @brief Returns the memory type for memory in the sky structure
     * or error code if the types are inconsistent.
     */
    int type() const;

    /**
     * @brief Writes the current contents of teh sky structure to an OSKAR
     * source text file.
     *
     * @param filename Path to a file to write to.
     *
     * @return An error code.
     */
    int write(const char* filename);
#endif
};
typedef struct oskar_SkyModel oskar_SkyModel;

#endif /* OSKAR_SKY_MODEL_H_ */
