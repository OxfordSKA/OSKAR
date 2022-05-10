#
# oskar/convert/CMakeLists.txt
#

set(name convert)
set(${name}_SRC
    define_convert_az_el_to_enu_directions.h
    define_convert_apparent_ra_dec_to_enu_directions.h
    define_convert_cirs_relative_directions_to_enu_directions.h
    define_convert_ecef_to_station_uvw.h
    define_convert_enu_directions_to_az_el.h
    #define_convert_enu_directions_to_cirs_relative_directions.h
    define_convert_enu_directions_to_local_tangent_plane.h
    define_convert_enu_directions_to_relative_directions.h
    define_convert_enu_directions_to_theta_phi.h
    define_convert_lon_lat_to_relative_directions.h
    define_convert_ludwig3_to_theta_phi_components.h
    define_convert_relative_directions_to_enu_directions.h
    define_convert_relative_directions_to_lon_lat.h
    define_convert_station_uvw_to_baseline_uvw.h
    define_convert_theta_phi_to_ludwig3_components.h
    src/oskar_convert_any_to_enu_directions.c
    src/oskar_convert_apparent_ra_dec_to_enu_directions.c
    src/oskar_convert_az_el_to_enu_directions.c
    src/oskar_convert_brightness_to_jy.c
    src/oskar_convert_cellsize_to_fov.c
    src/oskar_convert_cirs_relative_directions_to_enu_directions.c
    src/oskar_convert_date_time_to_mjd.c
    src/oskar_convert_ecef_to_enu.c
    src/oskar_convert_ecef_to_geodetic_spherical.c
    src/oskar_convert_ecef_to_station_uvw.c
    src/oskar_convert_enu_directions_to_az_el.c
    #src/oskar_convert_enu_directions_to_cirs_relative_directions.c
    src/oskar_convert_enu_directions_to_local_tangent_plane.c
    src/oskar_convert_enu_directions_to_relative_directions.c
    src/oskar_convert_enu_directions_to_theta_phi.c
    src/oskar_convert_enu_to_ecef.c
    src/oskar_convert_enu_to_offset_ecef.c
    src/oskar_convert_fov_to_cellsize.c
    src/oskar_convert_galactic_to_fk5.c
    src/oskar_convert_geodetic_spherical_to_ecef.c
    src/oskar_convert_healpix_ring_to_theta_phi.c
    src/oskar_convert_lon_lat_to_relative_directions.c
    src/oskar_convert_lon_lat_to_xyz.c
    src/oskar_convert_ludwig3_to_theta_phi_components.c
    src/oskar_convert_mjd_to_gast_fast.c
    src/oskar_convert_mjd_to_gmst.c
    #src/oskar_convert_mjd_ut1_to_era.c
    #src/oskar_convert_mjd_utc_to_mjd_tt.c
    src/oskar_convert_offset_ecef_to_ecef.c
    src/oskar_convert_relative_directions_to_enu_directions.c
    src/oskar_convert_relative_directions_to_lon_lat.c
    src/oskar_convert_station_uvw_to_baseline_uvw.c
    src/oskar_convert_theta_phi_to_enu_directions.c
    src/oskar_convert_theta_phi_to_ludwig3_components.c
    src/oskar_convert_xyz_to_lon_lat.c
    src/oskar_convert.cl
    #src/oskar_evaluate_diurnal_aberration.c
    src/oskar_equation_of_equinoxes_fast.c)

if (CUDA_FOUND)
    list(APPEND ${name}_SRC src/oskar_convert.cu)
endif()

set(${name}_SRC "${${name}_SRC}" PARENT_SCOPE)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
