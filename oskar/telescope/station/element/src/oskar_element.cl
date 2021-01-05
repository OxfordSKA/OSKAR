/* Copyright (c) 2018-2021, The OSKAR Developers. See LICENSE file. */

OSKAR_ELEMENT_TAPER_COSINE_SCALAR( M_CAT(apply_element_taper_cosine_scalar_, Real), Real, Real2)
OSKAR_ELEMENT_TAPER_COSINE_MATRIX( M_CAT(apply_element_taper_cosine_matrix_, Real), Real, Real4c)
OSKAR_ELEMENT_TAPER_GAUSSIAN_SCALAR( M_CAT(apply_element_taper_gaussian_scalar_, Real), Real, Real2)
OSKAR_ELEMENT_TAPER_GAUSSIAN_MATRIX( M_CAT(apply_element_taper_gaussian_matrix_, Real), Real, Real4c)
/*OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN( M_CAT(evaluate_geometric_dipole_pattern_, Real), Real, Real2)*/
OSKAR_EVALUATE_DIPOLE_PATTERN( M_CAT(evaluate_dipole_pattern_, Real), Real, Real2)
OSKAR_EVALUATE_DIPOLE_PATTERN_SCALAR( M_CAT(evaluate_dipole_pattern_scalar_, Real), Real, Real2, Real4c)
OSKAR_EVALUATE_SPHERICAL_WAVE_SUM( M_CAT(evaluate_spherical_wave_sum_, Real), Real, Real2, Real4c)
