#
# src/python/CmakeLists.txt
#

if (PYTHON_FOUND AND CUDA_FOUND)

    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${NUMPY_INCLUDE_DIRS})

    oskar_py_module(oskar oskar_module.c LIBS oskar)

endif()