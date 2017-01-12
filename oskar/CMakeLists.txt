#
# oskar/CMakeLists.txt
#

# Build OSKAR sub-libraries.
add_subdirectory(apps)
add_subdirectory(binary)
add_subdirectory(ms)
add_subdirectory(settings)

# Build OSKAR main library.
set(libname oskar)
set(${libname}_folders
    beam_pattern
    convert
    correlate
    imager
    log
    math
    mem
    simulator
    sky
    splines
    telescope
    utility
    vis)
foreach (module ${${libname}_folders})
    add_subdirectory(${module})
    foreach (file ${${module}_SRC})
        list(APPEND ${libname}_SRC ${module}/${file})
    endforeach()
endforeach()

if (CUDA_FOUND)
    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)
    cuda_add_library(${libname} ${${libname}_SRC})
else()
    add_library(${libname} ${${libname}_SRC})
endif()

# Link libraries.
target_link_libraries(${libname}
    cfitsio
    oskar_binary)

# Link with oskar_ms if we have casacore.
if (CASACORE_FOUND)
    target_link_libraries(${libname} oskar_ms)
endif()

# Link with cuFFT if we have CUDA.
if (CUDA_FOUND)
    target_link_libraries(${libname} ${CUDA_CUFFT_LIBRARIES})
endif()

if (MSVC)
    set_target_properties(${libname} PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        SOVERSION ${OSKAR_VERSION})
else ()
    set_target_properties(${libname} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}"
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
endif()

install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

if (CUDA_FOUND)
    cuda_build_clean_target()
endif()

# === Install header tree.
if (NOT ${CMAKE_INSTALL_PREFIX} MATCHES "/usr/local")
    string(REGEX MATCH "[^/|\\][a-zA-z0-9|_|-]+$" bin_dir ${CMAKE_BINARY_DIR})
    string(REGEX MATCH "[^/|\\]([a-zA-z0-9|_|-]+$)" install_dir
        ${CMAKE_INSTALL_PREFIX})
    string(COMPARE EQUAL ${CMAKE_INSTALL_PREFIX} ${CMAKE_BINARY_DIR}
        install_in_build_dir)
    if ($install_in_build_dir)
        set(install_include_dir "${bin_dir}/include")
    else ()
        set(install_include_dir "${bin_dir}/${install_dir}/include")
    endif()
else()
    set(install_include_dir "${CMAKE_INSTALL_PREFIX}/include")
endif ()
# TODO move install command into subdirectories to make this more modular,
# and give subdirectories control over which headers are installed.
install(DIRECTORY "${PROJECT_SOURCE_DIR}/oskar"
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "oskar*.h"
    PATTERN ${install_include_dir} EXCLUDE
    PATTERN *dierckx* EXCLUDE
    PATTERN src  EXCLUDE
    PATTERN test EXCLUDE
    PATTERN tests EXCLUDE
    PATTERN .git EXCLUDE)
install(FILES ${PROJECT_BINARY_DIR}/oskar/oskar_version.h
    DESTINATION "${OSKAR_INCLUDE_INSTALL_DIR}/oskar")
