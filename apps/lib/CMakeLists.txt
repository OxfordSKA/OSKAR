#
# src/apps/lib/CMakeLists.txt
#

set(apps_SRC
    src/oskar_load_stations.cpp
    src/oskar_load_telescope.cpp
    src/oskar_Settings.cpp
    src/oskar_imager_dft.cu
    src/oskar_write_ms.cpp
    src/oskar_file_utils.cpp
)
set(apps_SRC "${apps_SRC}" PARENT_SCOPE)

# ==== Macros to build the oskar-lib platform module.
include_directories(${QT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_apps.dir/src)
cuda_add_library(oskar_apps ${apps_SRC})
target_link_libraries(oskar_apps oskar oskar_ms ${CASACORE_LIBRARIES}
    ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
set_target_properties(oskar_apps PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(oskar_apps PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS oskar_apps DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar_apps PROPERTIES SOVERSION ${oskar-lib_SOVERSION})

# ==== Recurse into test subdirectory.
add_subdirectory(test)
