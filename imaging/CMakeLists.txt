#
# src/imaging/CMakeLists.txt
#
#==============================================================================

if (FFTW3_FOUND)

    include_directories(${QT_INCLUDE_DIR})

    # === Create list of source files for the oskar-lib imaging module.
    set(imaging_SRC
        src/ConvFunc.cpp
        src/FFTUtility.cpp
        src/GridCorrection.cpp
        src/Gridder.cpp
        src/WProjConvFunc.cpp
    )
    set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

    # === Build the oskar-lib imaging module.
    add_library(oskar_imaging SHARED ${imaging_SRC})
     set_target_properties(oskar_imaging PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    )
    set_target_properties(oskar_imaging PROPERTIES PREFIX "lib")
    target_link_libraries(oskar_imaging
        ${QT_QTCORE_LIBRARY}
        ${FFTW3_FFTWF_LIBRARY}
    )

    # === Recurse into test subdirectory.
    add_subdirectory(test)

endif (FFTW3_FOUND)
