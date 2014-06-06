#
# src/measures/CMakeLists.txt
#

# Add the source files in this directory.
set(name measures)
set(${name}_SRC
    src/oskar_convert_apparent_ha_dec_to_enu_direction_cosines.c
    src/oskar_convert_apparent_ra_dec_to_az_el.c
    src/oskar_convert_apparent_ra_dec_to_relative_direction_cosines.c
    src/oskar_convert_apparent_ra_dec_to_enu_direction_cosines.c
    src/oskar_convert_date_time_to_mjd.c
    src/oskar_convert_relative_direction_cosines_to_apparent_ra_dec.c
    src/oskar_convert_ecef_to_baseline_uvw.c
    src/oskar_convert_ecef_to_enu.c
    src/oskar_convert_ecef_to_geodetic_spherical.c
    src/oskar_convert_ecef_to_station_uvw.c
    src/oskar_convert_enu_direction_cosines_to_az_el.c
    src/oskar_convert_enu_direction_cosines_to_theta_phi.c
    src/oskar_convert_enu_direction_cosines_to_relative_direction_cosines.c
    src/oskar_convert_enu_to_ecef.c
    src/oskar_convert_enu_to_offset_ecef.c
    src/oskar_convert_galactic_to_fk5.c
    src/oskar_convert_geodetic_spherical_to_ecef.c
    src/oskar_convert_healpix_ring_to_theta_phi.c
    src/oskar_convert_lon_lat_to_xyz.c
    src/oskar_convert_ludwig3_to_theta_phi_components.c
    src/oskar_convert_mjd_to_gast_fast.c
    src/oskar_convert_mjd_to_gmst.c
    src/oskar_convert_offset_ecef_to_ecef.c
    src/oskar_convert_relative_direction_cosines_to_enu_direction_cosines.c
    src/oskar_convert_station_uvw_to_baseline_uvw.c
    src/oskar_convert_theta_phi_to_enu_direction_cosines.c
    src/oskar_convert_theta_phi_to_healpix_ring.c
    src/oskar_convert_xyz_to_lon_lat.c
    src/oskar_equation_of_equinoxes_fast.c
)

if (CUDA_FOUND)
    list(APPEND ${name}_SRC
        src/oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda.cu
        src/oskar_convert_ecef_to_station_uvw_cuda.cu
        src/oskar_convert_enu_direction_cosines_to_relative_direction_cosines_cuda.cu
        src/oskar_convert_enu_direction_cosines_to_theta_phi_cuda.cu
        src/oskar_convert_ludwig3_to_theta_phi_components_cuda.cu
        src/oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cuda.cu
        src/oskar_convert_station_uvw_to_baseline_uvw_cuda.cu
    )
endif()

set(${name}_SRC "${${name}_SRC}" PARENT_SCOPE)

add_subdirectory(test)
