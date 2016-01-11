#
# src/imaging/CMakeLists.txt
#

# Add the source files in this directory.
set(imaging_SRC
    src/oskar_evaluate_image_lon_lat_grid.c
    src/oskar_evaluate_image_lm_grid.c
    src/oskar_evaluate_image_lmn_grid.c
    src/oskar_evaluate_image_ranges.c
)
set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

add_subdirectory(test)
