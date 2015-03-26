/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#ifndef OSKAR_SETTINGS_IMAGE_H_
#define OSKAR_SETTINGS_IMAGE_H_

/**
 * @file oskar_SettingsImage.h
 */

/**
 * @struct oskar_SettingsImage
 *
 * @brief Structure to hold image settings.
 *
 * @details
 * The structure holds parameters for imaging.
 */
struct oskar_SettingsImage
{
    double fov_deg;
    int size;

    int image_type;        /* Polarisation selection or PSF
                            * enum (value from OSKAR_IMAGE_TYPE_XXX ) */

    int channel_snapshots; /* bool, false = frequency synthesis */
    int channel_range[2];  /* channel range (e.g. 1-1 = 1 channel,
                            * 1-2 = 2 channels)*/

    int time_snapshots;    /* bool, false = time synthesis */
    int time_range[2];     /* time range */

    int transform_type;    /* enum (transform type) */

    char* input_vis_data;  /* Path to input visibility data (used when running
                            * the imager as a standalone binary */

    int direction_type;    /* enum, specification of imaging direction */
    double ra_deg;         /* custom image pointing direction */
    double dec_deg;        /* custom image pointing direction */

    char* fits_image;      /* FITS file name path */
};
typedef struct oskar_SettingsImage oskar_SettingsImage;

enum OSKAR_IMAGE_TRANSFORM_TYPE
{
    OSKAR_IMAGE_DFT_2D,
    OSKAR_IMAGE_DFT_3D,
    OSKAR_IMAGE_FFT
};

enum OSKAR_IMAGE_DIRECTION
{
    OSKAR_IMAGE_DIRECTION_OBSERVATION, /* Observation (primary beam) direction */
    OSKAR_IMAGE_DIRECTION_RA_DEC
};


#endif /* OSKAR_SETTINGS_IMAGE_H_ */
