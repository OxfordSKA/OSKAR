/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_HARP_H_
#define OSKAR_HARP_H_

/**
 * @file oskar_harp.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Harp;
#ifndef OSKAR_HARP_TYPEDEF_
#define OSKAR_HARP_TYPEDEF_
typedef struct oskar_Harp oskar_Harp;
#endif /* OSKAR_HARP_TYPEDEF_ */

OSKAR_EXPORT
oskar_Harp* oskar_harp_create(int precision);

OSKAR_EXPORT
oskar_Harp* oskar_harp_create_copy(const oskar_Harp* other, int* status);

OSKAR_EXPORT
const oskar_Mem* oskar_harp_coeffs(const oskar_Harp* h, int feed);

OSKAR_EXPORT
void oskar_harp_evaluate_smodes(
        const oskar_Harp* h,
        int num_dir,
        const oskar_Mem* theta,
        const oskar_Mem* phi,
        oskar_Mem* poly,
        oskar_Mem* ee,
        oskar_Mem* qq,
        oskar_Mem* dd,
        oskar_Mem* pth,
        oskar_Mem* pph,
        int* status);

OSKAR_EXPORT
void oskar_harp_evaluate_station_beam(
        const oskar_Harp* h,
        int num_dir,
        const oskar_Mem* theta,
        const oskar_Mem* phi,
        double frequency_hz,
        int feed,
        int num_antennas,
        const oskar_Mem* antenna_x,
        const oskar_Mem* antenna_y,
        const oskar_Mem* antenna_z,
        const oskar_Mem* weights,
        const oskar_Mem* pth,
        const oskar_Mem* pph,
        oskar_Mem* phase_fac,
        oskar_Mem* beam_coeffs,
        int offset_out,
        oskar_Mem* beam,
        int* status);

OSKAR_EXPORT
void oskar_harp_evaluate_element_beam(
        const oskar_Harp* h,
        int num_dir,
        const oskar_Mem* theta,
        const oskar_Mem* phi,
        double frequency_hz,
        int feed,
        int i_antenna,
        int num_antennas,
        const oskar_Mem* antenna_x,
        const oskar_Mem* antenna_y,
        const oskar_Mem* antenna_z,
        const oskar_Mem* coeffs,
        const oskar_Mem* pth,
        const oskar_Mem* pph,
        oskar_Mem* phase_fac,
        int offset_out,
        oskar_Mem* beam,
        int* status);

OSKAR_EXPORT
void oskar_harp_evaluate_element_beams(
        const oskar_Harp* h,
        int num_dir,
        const oskar_Mem* theta,
        const oskar_Mem* phi,
        double frequency_hz,
        int feed,
        int num_antennas,
        const oskar_Mem* antenna_x,
        const oskar_Mem* antenna_y,
        const oskar_Mem* antenna_z,
        const oskar_Mem* coeffs,
        const oskar_Mem* pth,
        const oskar_Mem* pph,
        oskar_Mem* phase_fac,
        int offset_out,
        oskar_Mem* beam,
        int* status);

OSKAR_EXPORT
void oskar_harp_free(oskar_Harp* h);

OSKAR_EXPORT
void oskar_harp_open_hdf5(oskar_Harp* h, const char* path, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
