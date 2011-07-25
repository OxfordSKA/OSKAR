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

    if (NOT WIN32)
    install(TARGETS oskar_imaging LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    set_target_properties(oskar_imaging PROPERTIES
        SOVERSION ${oskar-lib_SOVERSION})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN "src" EXCLUDE
        PATTERN "test" EXCLUDE
        PATTERN ".svn" EXCLUDE)
    endif ()

endif (FFTW3_FOUND)
