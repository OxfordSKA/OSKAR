/* Copyright (c) 2019, The University of Oxford. See LICENSE file. */

OSKAR_MEM_NORMALISE_REAL_GPU(    M_CAT(mem_norm_real_, Real), Real)
OSKAR_MEM_NORMALISE_COMPLEX_GPU( M_CAT(mem_norm_complex_, Real), Real, Real2)
OSKAR_MEM_NORMALISE_MATRIX_GPU(  M_CAT(mem_norm_matrix_, Real), Real, Real4c)
