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

#include "fits/oskar_uvfits_writer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_uvfits_create(const char* filename, oskar_uvfits* fits)
{
    /* If the file exists, remove it. */
    FILE* file;
    if ((file = fopen(filename, "r")) != NULL)
    {
        fclose(file);
        remove(filename);
    }

    /* Defaults. */
    fits->status    = 0;
    fits->decimals  = 10;
    fits->num_axes  = 6; /* 6 => 0(==GROUPS),AMP(RE,IM,WGT),STOKES, FREQ, RA, DEC */
    fits->num_param = 5; /* 5 => UU, VV, WW, DATE, BASELINE */

    /* Create a new empty output FITS file. */
    /* fits_create_file */
    ffinit(&(fits->fptr), filename, &(fits->status));
    oskar_uvfits_check_status(fits->status, "Opening file");
}

void oskar_uvfits_check_status(const int status, const char* message)
{
    char line[80];

    /* No status error, return. */
    if (!status) return;

    memset(line, '*', 79);
    fprintf(stderr, "%s\n", line);

    /* Print user supplied message. */
    if (message == NULL || strlen(message) > 0)
        fprintf(stderr, "UVFITS ERROR: %s.", message);

    /* Print the CFITSIO error message. */
    fits_report_error(stderr, status);
    fprintf(stderr, "%s\n", line);
}

void oskar_uvfits_close(fitsfile* fits_file)
{
    int status = 0;
    if (fits_file == NULL) return;
    /* fits_close_file */
    ffclos(fits_file, &status);
    oskar_uvfits_check_status(status, "Closing file");
}

void oskar_uvfits_write_groups_header(fitsfile* fits_file, long long num_vis)
{
    /* Number of axes (6 are required, a 7th (BAND axis) is optional). */
    int num_axes = 6;
    long axis_dim[6];
    int simple = TRUE;          /* This file does conform to FITS standard. */
    int bitpix = FLOAT_IMG;     /* FLOAT_IMG=-32: AIPS doesn't use double! */
    int extend = TRUE;          /* Allow use of extensions. */
    int status = 0;
    /* Number of parameters.
     * 1. UU        (u baseline coordinate)
     * 2. VV        (v baseline coordinate)
     * 3. WW        (w baseline coordinate)
     * 4. DATE      (Julian date)
     * 5. BASELINE  (Baseline number = ant1 * 256 + ant2) */
    int num_param = 5;
    long long pcount = num_param;  /* Number of parameters per group. */
    long long gcount = num_vis;    /* Number of groups (i.e. visibilities) */
    long num_stokes = 1; /* Scalar only (I, Q, U, or V e.t.c.) */
    long num_freqs  = 1; /* Only write one frequency. */
    long num_ra     = 1; /* One pointing */
    long num_dec    = 1; /* One pointing */

    axis_dim[0] = 0;           /* No standard image just group */
    axis_dim[1] = 3;           /* (required) real, imaginary, weight. */
    axis_dim[2] = num_stokes;  /* (required) Stokes parameters. */
    axis_dim[3] = num_freqs;   /* (required) Frequency (spectral channel). */
    axis_dim[4] = num_ra;      /* (required) Right ascension of phase centre. */
    axis_dim[5] = num_dec;     /* (required) Declination of phase centre. */

    /* Write random groups description header. */
    fits_write_grphdr(fits_file, simple, bitpix, num_axes, axis_dim,
            pcount, gcount, extend, &status);

    /* Check the CFITSIO error status. */
    oskar_uvfits_check_status(status, "Write groups header");
}

void oskar_uvfits_write_header(fitsfile* fits_file, const char* filename,
		double ra0, double dec0, double frequency0, double date0)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE], name[FLEN_COMMENT];
    int decimals = 10;
    int status = 0;
    double invFreq;
    fits_write_date(fits_file, &status);

    strcpy(key, "TELESCOP");
    strcpy(value, "OSKAR SIM (0.0.0)");
    fits_write_key_str(fits_file,  key, value, NULL, &status);
    strcpy(key, "BUNIT");
    strcpy(value, "JY");
    fits_write_key_str(fits_file, key, value, "Units of flux", &status);
    fits_write_key_dbl(fits_file, "EQUINOX", 2000.0, decimals, "Epoch of RA DEC", &status);
    fits_write_key_dbl(fits_file, "OBSRA", ra0, decimals, "Antenna pointing RA", &status);
    fits_write_key_dbl(fits_file, "OBSDEC", dec0, decimals, "Antenna pointing DEC", &status);

    /* Axis description headers (Note: axis 1 = empty). */
    oskar_uvfits_write_axis_header(fits_file, 2, "COMPLEX", "1=real, 2=imag, 3=weight",
            1.0, 1.0, 1.0, 1.0);
    oskar_uvfits_write_axis_header(fits_file, 3, "STOKES", "==scalar (I/Q/U/V)",
            1.0, 1.0, 1.0, 1.0);
    oskar_uvfits_write_axis_header(fits_file, 4, "FREQ", "Frequency in Hz.",
            frequency0, 0.0, 1.0, 0.0);
    oskar_uvfits_write_axis_header(fits_file, 5, "RA", "Right Ascension in deg.",
            ra0, 0.0, 1.0, 1.0);
    oskar_uvfits_write_axis_header(fits_file, 6, "DEC", "Declination in deg.",
            dec0, 0.0, 1.0, 1.0);
    oskar_uvfits_check_status(status, "");

    /* Parameter headers. */
    invFreq = 1.0 / frequency0;
    oskar_uvfits_write_param_header(fits_file, 1, "UU--",     "", invFreq, 0.0);
    oskar_uvfits_write_param_header(fits_file, 2, "VV--",     "", invFreq, 0.0);
    oskar_uvfits_write_param_header(fits_file, 3, "WW--",     "", invFreq, 0.0);
    oskar_uvfits_write_param_header(fits_file, 4, "DATE",     "", 1.0,     date0);
    oskar_uvfits_write_param_header(fits_file, 5, "BASELINE", "", 1.0,     0.0);
    oskar_uvfits_check_status(status, "");

    /* Write a name that is picked up by AIPS. */
    strcat(name, "AIPS   IMNAME='");
    strcat(name, filename);
    strcat(name, "'");
    fits_write_history(fits_file, name, &status);
}

void oskar_uvfits_write_axis_header(fitsfile* fits_file, int id,
        const char* ctype, const char* comment, double crval,
        double cdelt, double crpix, double crota)
{
    char s_key[FLEN_KEYWORD], s_value[FLEN_VALUE], s_comment[FLEN_COMMENT];
    int status = 0;
    int decimals = 10;

    strcpy(s_comment, comment);
    strcpy(s_value, ctype);

    fits_make_keyn("CTYPE", id, s_key, &status);
    fits_write_key_str(fits_file, s_key, s_value, s_comment, &status);

    fits_make_keyn("CRVAL", id, s_key, &status);
    fits_write_key_dbl(fits_file, s_key, crval, decimals, NULL, &status);

    fits_make_keyn("CDELT", id, s_key, &status);
    fits_write_key_dbl(fits_file, s_key, cdelt, decimals, NULL, &status);

    fits_make_keyn("CRPIX", id, s_key, &status);
    fits_write_key_dbl(fits_file, s_key, crpix, decimals, NULL, &status);

    fits_make_keyn("CROTA", id, s_key, &status);
    fits_write_key_dbl(fits_file, s_key, crota, decimals, NULL, &status);
}

void oskar_uvfits_write_param_header(fitsfile* fits_file, int id,
        const char* type, const char* comment, double scale,
        double zero)
{
    char s_key[FLEN_KEYWORD], s_value[FLEN_VALUE], s_comment[FLEN_COMMENT];
    int status = 0;
    int decimals = 10;

    fits_make_keyn("PTYPE", id, s_key, &status);
    strcpy(s_value, type);
    strcpy(s_comment, comment);
    fits_write_key_str(fits_file, s_key, s_value, s_comment, &status);

    fits_make_keyn("PSCAL", id, s_key, &status);
    fits_write_key_dbl(fits_file, s_key, scale, decimals, NULL, &status);

    fits_make_keyn("PZERO", id, s_key, &status);
    fits_write_key_dbl(fits_file, s_key, zero, decimals, NULL, &status);
}

/* FIXME This needs fixing to use the new visibility structure. */
#if 0
void oskar_uvfits_write_data(fitsfile* fits_file, const oskar_VisData_d* vis,
        const double* weight, const double* date, const double* baseline)
{
	int i, j;
    int status = 0;

    /* fixme: work out how to read this from the already written header. */
    int num_axes = 6;
    long axis_dim[6];
    axis_dim[0] = 0;  /* No standard image just group */
    axis_dim[1] = 3;  /* (required) real, imaginary, weight. */
    axis_dim[2] = 1;  /* (required) Stokes parameters. */
    axis_dim[3] = 1;  /* (required) Frequency (spectral channel). */
    axis_dim[4] = 1;  /* (required) Right ascension of phase centre. */
    axis_dim[5] = 1;  /* (required) Declination of phase centre. */

    /* Setup compressed axis dimensions vector. */
    long naxes[5]; /* length = num_axis-1 */
    for (i = 0; i < num_axes - 1; ++i)
        naxes[i] = axis_dim[i+1];

    /* length = num_axes */
    long fpixel[5] = {1, 1, 1, 1, 1};
    long lpixel[5] = {3, 1, 1, 1, 1}; /* == naxes */

    int num_values_per_group = 1;
    for (i = 1; i < num_axes; ++i)
        num_values_per_group *= axis_dim[i];
    printf("num values per group = %i\n", num_values_per_group);

    int num_param = 5;
    long firstelem = 1;
    long nelements = num_param;

    float p_temp[5]; /* length = num_param */
    float *g_temp = (float*) malloc(num_values_per_group * sizeof(float));

    for (i = 0; i < vis->num_samples; ++i)
    {
        long group = (long)i + 1;

        /* Write the parameters. */
        p_temp[0] = vis->u[i];
        p_temp[1] = vis->v[i];
        p_temp[2] = vis->w[i];
        p_temp[3] = date[i];
        p_temp[4] = baseline[i];

        printf("- writing group %li\n", group);
        for (j = 0; j < nelements; ++j)
            printf("   param %i = %f\n", j+1, p_temp[j]);

        fits_write_grppar_flt(fits_file, group, firstelem, nelements,
                p_temp, &status);
        oskar_uvfits_check_status(status, "");

        /* Write the data. */
        g_temp[0] = vis->amp[i].x; /* re */
        g_temp[1] = vis->amp[i].y; /* im */
        g_temp[2] = weight[i];

        for (j = 0; j < num_values_per_group; ++j)
            printf("   data %i = %f\n", j+1, g_temp[j]);

        fits_write_subset_flt(fits_file, group, 5, naxes, fpixel, lpixel, g_temp, &status);

        oskar_uvfits_check_status(status, "");
    }
    free(g_temp);
}
#endif

int oskar_uvfits_baseline_id(int ant1, int ant2)
{
    return ant1 * 256 + ant2;
}

#ifdef __cplusplus
}
#endif
