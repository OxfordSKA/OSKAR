#
# src/imaging/CMakeLists.txt
#

# Add the source files in this directory.
set(imaging_SRC
    src/oskar_evaluate_image_lon_lat_grid.c
    src/oskar_evaluate_image_lm_grid.c
    src/oskar_evaluate_image_lmn_grid.c
    src/oskar_evaluate_image_ranges.c
    src/oskar_get_image_vis_amps.cpp
    src/oskar_get_image_baseline_coords.cpp
    src/oskar_get_image_stokes.c
    src/oskar_image_accessors.c
    src/oskar_image_create.c
    src/oskar_image_free.c
    src/oskar_image_resize.c
    src/oskar_make_image_dft.c
)
set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

add_subdirectory(test)
