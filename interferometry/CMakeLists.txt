#
# src/interferometry/CMakeLists.txt
#
#==============================================================================

include_directories(${oskar-lib_SOURCE_DIR}/interferometry)

# Add the source files in this directory.
set(interferometry_SRC
    src/oskar_synthesis_baselines.c
    src/oskar_synthesis_xyz2uvw.c
)

cuda_add_library(oskar_interferometry ${interferometry_SRC})


set_target_properties(oskar_interferometry PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
