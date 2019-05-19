/* Copyright (c) 2018-2019, The University of Oxford. See LICENSE file. */

OSKAR_DFT_C2R_GPU(  M_CAT(dft_c2r_2d_,  Real), false, Real, Real2)
OSKAR_DFT_C2R_GPU(  M_CAT(dft_c2r_3d_,  Real), true, Real, Real2)
OSKAR_DFTW_C2C_GPU( M_CAT(dftw_c2c_2d_, Real), false, Real, Real2)
OSKAR_DFTW_C2C_GPU( M_CAT(dftw_c2c_3d_, Real), true, Real, Real2)
OSKAR_DFTW_M2M_GPU( M_CAT(dftw_m2m_2d_, Real), false, Real, Real2, Real4c)
OSKAR_DFTW_M2M_GPU( M_CAT(dftw_m2m_3d_, Real), true, Real, Real2, Real4c)
OSKAR_DFTW_O2C_GPU( M_CAT(dftw_o2c_2d_, Real), false, Real, Real2)
OSKAR_DFTW_O2C_GPU( M_CAT(dftw_o2c_3d_, Real), true, Real, Real2)
