/* Copyright (c) 2018-2020, The OSKAR Developers. See LICENSE file. */

OSKAR_JONES_R( M_CAT(evaluate_jones_R_, Real), Real, Real4c)
OSKAR_JONES_APPLY_STATION_GAINS_C( M_CAT(jones_apply_station_gains_complex_, Real), Real2)
OSKAR_JONES_APPLY_STATION_GAINS_M( M_CAT(jones_apply_station_gains_matrix_, Real), Real4c)
