#
# src/imaging/CMakeLists.txt
#

include_directories(${oskar-lib_SOURCE_DIR}/imaging)

add_subdirectory(matlab)

# Add the source files in this directory.
set(imaging_SRC
    src/oskar_evaluate_image_lm_grid.c
    src/oskar_image_free.c
    src/oskar_image_init.c
    src/oskar_image_resize.c
    src/oskar_image_write.c
)
set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

#add_subdirectory(test)

