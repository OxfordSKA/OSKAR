#
# src/math/modules/CMakeLists.txt
#
#==============================================================================

# === Source files for the math modules sub-module.
set(math_modules_SRC
    src/Gridder.cpp
    src/GriddingKernels.cpp
)
set(math_modules_SRC "${math_modules_SRC}" PARENT_SCOPE)

# === Build a modules sub-module for testing (this is NOT installed).
include_directories(${QT_QTCORE_INCLUDE_DIR})
add_library(oskar_math_modules ${math_modules_SRC})
set_target_properties(oskar_math_modules PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(oskar_math_modules
    ${QT_QTCORE_LIBRARY}
)

# === Recurse into test subdirectory.
add_subdirectory(test)
