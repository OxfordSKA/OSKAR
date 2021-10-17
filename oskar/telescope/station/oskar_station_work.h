/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STATION_WORK_H_
#define OSKAR_STATION_WORK_H_

/**
 * @file oskar_station_work.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_StationWork;
#ifndef OSKAR_STATION_WORK_TYPEDEF_
#define OSKAR_STATION_WORK_TYPEDEF_
typedef struct oskar_StationWork oskar_StationWork;
#endif /* OSKAR_STATION_WORK_TYPEDEF_ */

/**
 * @brief Creates a station work buffer structure.
 *
 * @details
 * This function initialises a structure to hold work buffers that are
 * used when calculating station beam data.
 *
 * @param[in,out] work      Pointer to structure to be initialised.
 * @param[in]     type      OSKAR memory type ID.
 * @param[in]     location  OSKAR memory location ID.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
oskar_StationWork* oskar_station_work_create(int type, int location,
        int* status);

/**
 * @brief Frees memory in a station work buffer structure.
 *
 * @details
 * This function frees memory in a data structure used to hold work buffers
 * for calculating a station beam.
 *
 * @param[in,out]  work   Pointer to structure containing memory to free.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_station_work_free(oskar_StationWork* work, int* status);

/* Accessors. */

OSKAR_EXPORT
oskar_Mem* oskar_station_work_horizon_mask(oskar_StationWork* work);

OSKAR_EXPORT
oskar_Mem* oskar_station_work_source_indices(oskar_StationWork* work);

OSKAR_EXPORT
oskar_Mem* oskar_station_work_enu_direction(oskar_StationWork* work, int dim,
        int num_points, int* status);

OSKAR_EXPORT
oskar_Mem* oskar_station_work_lmn_direction(oskar_StationWork* work, int dim,
        int num_points, int* status);

OSKAR_EXPORT
void oskar_station_work_set_isoplanatic_screen(oskar_StationWork* work,
        int flag);

OSKAR_EXPORT
void oskar_station_work_set_tec_screen_common_params(oskar_StationWork* work,
        char screen_type, double screen_height_km, double screen_pixel_size_m,
        double screen_time_interval_sec);

OSKAR_EXPORT
void oskar_station_work_set_tec_screen_path(oskar_StationWork* work,
        const char* path);

OSKAR_EXPORT
const oskar_Mem* oskar_station_work_evaluate_tec_screen(oskar_StationWork* work,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        double station_u_m, double station_v_m, int time_index,
        double frequency_hz, int* status);

OSKAR_EXPORT
oskar_Mem* oskar_station_work_beam_out(oskar_StationWork* work,
        const oskar_Mem* output_beam, size_t length, int* status);

OSKAR_EXPORT
oskar_Mem* oskar_station_work_beam(oskar_StationWork* work,
        const oskar_Mem* output_beam, size_t length, int depth, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
