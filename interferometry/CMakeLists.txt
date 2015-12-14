#
# src/interferometry/CMakeLists.txt
#

# Add the source files in this directory.
set(interferometry_SRC
    src/oskar_telescope_accessors.c
    src/oskar_telescope_analyse.c
    src/oskar_telescope_create.c
    src/oskar_telescope_create_copy.c
    src/oskar_telescope_duplicate_first_station.c
    src/oskar_telescope_free.c
    src/oskar_telescope_load_pointing_file.c
    src/oskar_telescope_load_station_coords_ecef.c
    src/oskar_telescope_load_station_coords_horizon.c
    src/oskar_telescope_resize.c
    src/oskar_telescope_save_layout.c
    src/oskar_telescope_set_station_coords.c
    src/oskar_vis_block_accessors.c
    src/oskar_vis_block_add_system_noise.c
    src/oskar_vis_block_clear.c
    src/oskar_vis_block_copy.c
    src/oskar_vis_block_create.c
    src/oskar_vis_block_free.c
    src/oskar_vis_block_read.c
    src/oskar_vis_block_write.c
    src/oskar_vis_header_accessors.c
    src/oskar_vis_header_create.c
    src/oskar_vis_header_create_copy.c
    src/oskar_vis_header_free.c
    src/oskar_vis_header_read.c
    src/oskar_vis_header_write.c

    # Deprecated:
    src/oskar_vis_accessors.c
    src/oskar_vis_create.c
    src/oskar_vis_free.c
    src/oskar_vis_read.c
    src/oskar_vis_write.c
    
)

set(interferometry_SRC "${interferometry_SRC}" PARENT_SCOPE)

add_subdirectory(test)
