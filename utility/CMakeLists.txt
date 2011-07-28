#
# src/utility/CMakeLists.txt
#

# ==== Needed?
include_directories(${oskar-lib_SOURCE_DIR}/utility)

# ==== Utility library source
set(utility_SRC
    src/oskar_util_cuda_memory.cu
)
set(utility_SRC "${utility_SRC}" PARENT_SCOPE)


# ==== Build the utility library.
#cuda_add_library(oskar_utility ${utility_SRC})
#target_link_libraries(oskar_utility
#    ${CUDA_CUBLAS_LIBRARIES}
#    ${CUDA_LIBRARIES})
#set_target_properties(oskar_utility PROPERTIES
#    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
#    INSTALL_RPATH_USE_LINK_PATH TRUE)
#
#
# ==== Install target.
#if (NOT WIN32)
#    install(TARGETS oskar_utility LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
#    set_target_properties(oskar_utility PROPERTIES
#        SOVERSION ${oskar-lib_SOVERSION})
#    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
#        FILES_MATCHING PATTERN "*.h"
#        PATTERN "src" EXCLUDE
#        PATTERN "test" EXCLUDE
#        PATTERN ".svn" EXCLUDE)
#endif ()


# === Recurse into test directory.
add_subdirectory(test)
