#
# oskar/CMakeLists.txt
#

include(oskar_build_macros)
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
endif()

# Try to build MS library first.
add_subdirectory(ms)
if (NOT CASACORE_FOUND)
    add_definitions(-DOSKAR_NO_MS)
endif()

# Build OSKAR sub-libraries.
add_subdirectory(apps)
add_subdirectory(binary)
add_subdirectory(settings)

# Build OSKAR main library.
set(libname oskar)
set(${libname}_folders
    beam_pattern
    convert
    correlate
    gains
    harp
    imager
    interferometer
    log
    math
    mem
    sky
    telescope
    utility
    vis)
foreach (module ${${libname}_folders})
    add_subdirectory(${module})
    foreach (file ${${module}_SRC})
        get_filename_component(fileext ${file} EXT)
        if ((fileext STREQUAL ".cl") OR (fileext STREQUAL ".h"))
            list(APPEND cl_SRC ${module}/${file})
        else()
            list(APPEND ${libname}_SRC ${module}/${file})
        endif()
    endforeach()
endforeach()

if (OpenCL_FOUND)
    OSKAR_WRAP_CL(${libname}_SRC ${cl_SRC})
endif()

# Build the main library.
add_library(${libname} ${${libname}_SRC})

# Link libraries.
target_link_libraries(${libname} PRIVATE cfitsio Threads::Threads)
target_link_libraries(${libname} PUBLIC oskar_binary)
if (WIN32)
    add_definitions(-DPSAPI_VERSION=1 -D_CRT_SECURE_NO_WARNINGS)
    target_link_libraries(${libname} PRIVATE Psapi)
endif()

# Link with oskar_ms if we have casacore.
if (CASACORE_FOUND)
    target_link_libraries(${libname} PUBLIC oskar_ms)
endif()

# Link with other optional dependencies if they're available.
if (OpenMP_C_FOUND)
    target_link_libraries(${libname} PRIVATE OpenMP::OpenMP_C)
endif()
if (OpenMP_CXX_FOUND)
    target_link_libraries(${libname} PRIVATE OpenMP::OpenMP_CXX)
endif()
if (HDF5_FOUND)
    target_link_libraries(${libname} PRIVATE ${HDF5_LIBRARIES})
endif()
if (OpenCL_FOUND)
    target_link_libraries(${libname} PRIVATE ${OpenCL_LIBRARIES})
endif()
if (CUDAToolkit_FOUND)
    target_link_libraries(${libname} PRIVATE CUDA::cufft CUDA::cudart)
endif()
if (HARP_FOUND)
    target_link_libraries(${libname} PRIVATE ${HARP_LIBRARY})
endif()
if (ska-sdp-func_FOUND)
    target_link_libraries(${libname} PRIVATE ska-sdp-func::ska_sdp_func)
endif()

set_target_properties(${libname} PROPERTIES
    SOVERSION ${OSKAR_VERSION}
    VERSION ${OSKAR_VERSION})

install(TARGETS ${libname}
    ARCHIVE DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime
)

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
    COMPONENT headers
    FILES_MATCHING REGEX "(oskar.*h$)|(.*hpp$)"
    PATTERN ${install_include_dir} EXCLUDE
    PATTERN define* EXCLUDE
    PATTERN private* EXCLUDE
    PATTERN cmake EXCLUDE
    PATTERN src  EXCLUDE
    PATTERN test EXCLUDE
    PATTERN tests EXCLUDE
    PATTERN xml EXCLUDE
    PATTERN .git EXCLUDE)
install(FILES ${PROJECT_BINARY_DIR}/oskar/oskar_version.h
    DESTINATION "${OSKAR_INCLUDE_INSTALL_DIR}/oskar"
    COMPONENT headers
)
