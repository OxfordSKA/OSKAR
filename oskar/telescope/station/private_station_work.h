/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_STATION_WORK_H_
#define OSKAR_PRIVATE_STATION_WORK_H_

#include <mem/oskar_mem.h>

struct oskar_StationWork
{
    oskar_Mem* weights;          /* Complex scalar. */
    oskar_Mem* weights_scratch;  /* Complex scalar. */
    oskar_Mem* horizon_mask;     /* Integer. */
    oskar_Mem* source_indices;   /* Integer. */
    oskar_Mem* enu[3];           /* Real scalar. Direction cosines. */
    oskar_Mem* temp_dir_in[3];
    oskar_Mem* temp_dir_out[3];
    oskar_Mem* lmn[3];
    oskar_Mem* theta_modified;   /* Real scalar. */
    oskar_Mem* phi_x;            /* Real scalar. */
    oskar_Mem* phi_y;            /* Real scalar. */
    oskar_Mem* beam_out_scratch; /* Output scratch array. */

    /* TEC screen. */
    char screen_type;
    int isoplanatic_screen;
    int previous_time_index;
    int screen_num_pixels_x, screen_num_pixels_y, screen_num_pixels_t;
    double screen_height_km;
    double screen_pixel_size_m;
    double screen_time_interval_sec;
    oskar_Mem *tec_screen_path, *tec_screen;
    oskar_Mem *screen_output;

    int num_depths;
    oskar_Mem** beam;            /* For hierarchical stations. */

    /* HARP data. */
    oskar_Mem *poly, *ee, *qq, *dd, *phase_fac, *beam_coeffs, *pth, *pph;
};

#ifndef OSKAR_STATION_WORK_TYPEDEF_
#define OSKAR_STATION_WORK_TYPEDEF_
typedef struct oskar_StationWork oskar_StationWork;
#endif /* OSKAR_STATION_WORK_TYPEDEF_ */

#endif /* include guard */
