/* Copyright (c) 2018-2021, The OSKAR Developers. See LICENSE file. */

OSKAR_CONVERT_AZ_EL_TO_ENU_DIR( M_CAT(convert_az_el_to_enu_directions_, Real), Real)
OSKAR_CONVERT_CIRS_REL_DIR_TO_ENU_DIR( M_CAT(convert_cirs_relative_directions_to_enu_directions_, Real), Real)
OSKAR_CONVERT_ECEF_TO_STATION_UVW( M_CAT(convert_ecef_to_station_uvw_, Real), Real)
OSKAR_CONVERT_ENU_DIR_TO_AZ_EL( M_CAT(convert_enu_directions_to_az_el_, Real), Real)
/*OSKAR_CONVERT_ENU_DIR_TO_CIRS_REL_DIR( M_CAT(convert_enu_directions_to_cirs_relative_directions_, Real), Real)*/
OSKAR_CONVERT_ENU_DIR_TO_LOCAL( M_CAT(convert_enu_directions_to_local_, Real), Real)
OSKAR_CONVERT_ENU_DIR_TO_REL_DIR( M_CAT(convert_enu_directions_to_relative_directions_, Real), Real)
OSKAR_CONVERT_ENU_DIR_TO_THETA_PHI( M_CAT(convert_enu_directions_to_theta_phi_, Real), Real)
OSKAR_CONVERT_LON_LAT_TO_REL_DIR( M_CAT(convert_lon_lat_to_relative_directions_2d_, Real), 0, Real)
OSKAR_CONVERT_LON_LAT_TO_REL_DIR( M_CAT(convert_lon_lat_to_relative_directions_3d_, Real), 1, Real)
OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI( M_CAT(convert_ludwig3_to_theta_phi_, Real), Real, Real2)
OSKAR_CONVERT_REL_DIR_TO_ENU_DIR( M_CAT(convert_relative_directions_to_enu_directions_, Real), Real)
OSKAR_CONVERT_REL_DIR_TO_LON_LAT( M_CAT(convert_relative_directions_to_lon_lat_2d_, Real), 0, Real)
OSKAR_CONVERT_REL_DIR_TO_LON_LAT( M_CAT(convert_relative_directions_to_lon_lat_3d_, Real), 1, Real)
OSKAR_CONVERT_STATION_UVW_TO_BASELINE_UVW( M_CAT(convert_station_uvw_to_baseline_uvw_, Real), Real)
OSKAR_CONVERT_THETA_PHI_TO_LUDWIG3( M_CAT(convert_theta_phi_to_ludwig3_, Real), Real, Real2, Real4c)
OSKAR_CONVERT_RA_DEC_TO_ENU_DIR( M_CAT(convert_apparent_ra_dec_to_enu_directions_, Real), Real)
