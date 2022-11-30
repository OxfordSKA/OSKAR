/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_BEAM_UTILS_H_
#define OSKAR_BEAM_UTILS_H_

#ifdef __cplusplus

#include <complex>

namespace oskar{

/// C++ alias of the double4c type. The types should have the same alignment.
struct alignas(4 * sizeof(std::complex<double>)) Double4C {
    std::complex<double> a, b, c, d;
};

namespace beam_utils{

/* Functions used by EveryBeam. */

void evaluate_dipole_pattern_double(
        const int num_points,
        const double* theta,
        const double* phi,
        const double freq_hz,
        const double dipole_length_m,
        std::complex<double>* pattern);

void evaluate_spherical_wave_sum_double(
        double theta,
        double phi_x,
        double phi_y,
        int l_max,
        const Double4C* alpha,
        Double4C* pattern);
} // namespace beam_utils
} // namespace oskar
#endif /* __cplusplus */

#endif /* include guard */
