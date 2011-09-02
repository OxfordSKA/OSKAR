#
# src/imaging/CMakeLists.txt
#

#include_directories(${QT_INCLUDE_DIR})

# ==== Create list of source files for the oskar-lib imaging module.
set(imaging_SRC
    src/oskar_evaluate_gridding_kernels.c
    src/oskar_fft_imager.c
    src/oskar_fft_utility.c
    src/oskar_gridding.c
    src/oskar_grid_correct.c
    src/oskar_VisGrid.c
)
#set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

# ==== Build the oskar-lib imaging module.
add_library(oskar_imaging SHARED ${imaging_SRC})
set_target_properties(oskar_imaging PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
target_link_libraries(oskar_imaging
    ${FFTW3_FFTWF_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
)
set_target_properties(oskar_imaging PROPERTIES PREFIX "lib")

#install(TARGETS oskar_imaging LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
#set_target_properties(oskar_imaging PROPERTIES SOVERSION ${oskar-lib_SOVERSION})
#set_target_properties(oskar_imaging PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# ==== Recurse into test subdirectory.
add_subdirectory(test)

