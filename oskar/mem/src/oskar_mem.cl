/* Copyright (c) 2018-2022, The OSKAR Developers. See LICENSE file. */

OSKAR_MEM_ADD( M_CAT(mem_add_, Real), Real)
OSKAR_MEM_CONJ( M_CAT(mem_conj_, Real), Real2)
OSKAR_MEM_MUL_RR_R( M_CAT(mem_mul_rr_r_, Real), Real)
OSKAR_MEM_MUL_CC_C( M_CAT(mem_mul_cc_c_, Real), Real2)
OSKAR_MEM_MUL_CC_M( M_CAT(mem_mul_cc_m_, Real), Real, Real2, Real4c)
OSKAR_MEM_MUL_CM_M( M_CAT(mem_mul_cm_m_, Real), Real2, Real4c)
OSKAR_MEM_MUL_MC_M( M_CAT(mem_mul_mc_m_, Real), Real2, Real4c)
OSKAR_MEM_MUL_MM_M( M_CAT(mem_mul_mm_m_, Real), Real2, Real4c)
OSKAR_MEM_SCALE_REAL( M_CAT(mem_scale_, Real), Real)
OSKAR_MEM_SET_VALUE_REAL_REAL(    M_CAT(mem_set_value_real_r_, Real), Real)
OSKAR_MEM_SET_VALUE_REAL_COMPLEX( M_CAT(mem_set_value_real_c_, Real), Real, Real2)
OSKAR_MEM_SET_VALUE_REAL_MATRIX(  M_CAT(mem_set_value_real_m_, Real), Real, Real4c)
