#
# oskar/CMakeLists.txt
#

include(oskar_build_macros)

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
    interferometer
    log
    math
    mem
    sky
    splines
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

if (CUDA_FOUND)
    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)
    cuda_add_library(${libname} ${${libname}_SRC})
else()
    add_library(${libname} ${${libname}_SRC})
endif()

# Link libraries.
target_link_libraries(${libname} cfitsio oskar_binary Threads::Threads)
if (WIN32)
    add_definitions(-DPSAPI_VERSION=1 -D_CRT_SECURE_NO_WARNINGS)
    target_link_libraries(${libname} Psapi)
endif()

# Link with oskar_ms if we have casacore.
if (CASACORE_FOUND)
    target_link_libraries(${libname} oskar_ms)
else()
    add_definitions(-DOSKAR_NO_MS)
endif()

# Link with HDF5 if we have it.
if (HDF5_FOUND)
    target_link_libraries(${libname} ${HDF5_LIBRARIES})
    add_definitions(-DOSKAR_HAVE_HDF5)
endif()

# Link with OpenCL if we have it.
if (OpenCL_FOUND)
    target_link_libraries(${libname} ${OpenCL_LIBRARIES})
endif()

# Link with cuFFT if we have CUDA.
if (CUDA_FOUND)
    target_link_libraries(${libname} ${CUDA_CUFFT_LIBRARIES})
    if (WIN32)
        file(GLOB lib_cudart_ "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64*")
        file(GLOB lib_cufft_ "${CUDA_TOOLKIT_ROOT_DIR}/bin/cufft64*")
        install(FILES ${lib_cudart_} ${lib_cufft_}
            DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime)
    endif()
endif()

set_target_properties(${libname} PROPERTIES
    SOVERSION ${OSKAR_VERSION}
    VERSION ${OSKAR_VERSION})

install(TARGETS ${libname}
    ARCHIVE DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime
)

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
    COMPONENT headers
    FILES_MATCHING REGEX "(oskar.*h)|(.*hpp)"
    PATTERN ${install_include_dir} EXCLUDE
    PATTERN define* EXCLUDE
    PATTERN private* EXCLUDE
    PATTERN *dierckx* EXCLUDE
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
