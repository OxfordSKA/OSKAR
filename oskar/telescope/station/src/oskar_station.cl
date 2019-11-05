/* Copyright (c) 2018-2019, The University of Oxford. See LICENSE file. */

OSKAR_BLANK_BELOW_HORIZON_SCALAR( M_CAT(blank_below_horizon_scalar_, Real), Real, Real2)
OSKAR_BLANK_BELOW_HORIZON_MATRIX( M_CAT(blank_below_horizon_matrix_, Real), Real, Real4c)
OSKAR_ELEMENT_WEIGHTS_DFT( M_CAT(evaluate_element_weights_dft_, Real), Real, Real2)
OSKAR_ELEMENT_WEIGHTS_ERR( M_CAT(evaluate_element_weights_errors_, Real), Real, Real2)
OSKAR_EVALUATE_TEC_SCREEN( M_CAT(evaluate_tec_screen_, Real), Real, Real2)
OSKAR_EVALUATE_VLA_BEAM_PBCOR_SCALAR( M_CAT(evaluate_vla_beam_pbcor_scalar_, Real), Real, Real2)
OSKAR_EVALUATE_VLA_BEAM_PBCOR_MATRIX( M_CAT(evaluate_vla_beam_pbcor_matrix_, Real), Real, Real4c)
