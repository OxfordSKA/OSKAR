#
# src/interferometry/CMakeLists.txt
#
#==============================================================================

include_directories(${oskar-lib_SOURCE_DIR}/interferometry)

# Add the source files in this directory.
set(interferometry_SRC
    src/oskar_interferometry_baselines.c
    src/oskar_interferometry_xyz2uvw.c
)
set(interferometry_SRC "${interferometry_SRC}" PARENT_SCOPE)
