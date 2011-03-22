#
# /src/math/core/CMakeLists.txt
#
#==============================================================================

# === Source files for the math core sub-module.
set(math_core_SRC
    #
)
set(math_core_SRC "${math_core_SRC}" PARENT_SCOPE)

# === Build a core sub-module for testing (this is NOT installed).
#include_directories(${QT_QTCORE_INCLUDE_DIR})
#add_library(oskar_math_core ${math_modules_SRC})
#set_target_properties(oskar_math_core PROPERTIES
#    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
#    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
#)
#target_link_libraries(oskar_math_core
#    ${QT_QTCORE_LIBRARY}
#)

# === Recurse into test subdirectory.
add_subdirectory(test)
