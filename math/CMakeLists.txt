#
# src/math/CMakeLists.txt
#
#==============================================================================

# === Recurse into the oskar-lib beamforming math sub-module.
include_directories(${oskar-lib_SOURCE_DIR}/math)

add_subdirectory(beamforming)

# === Recurse into the oskar-lib core math sub-module.
add_subdirectory(core)

# === Recurse into the oskar-lib synthesis math sub-module.
add_subdirectory(synthesis)

# === Get list of source files required to build the math module.
set(math_SRC)
foreach (file ${math_beamforming_SRC})
    list(APPEND math_SRC beamforming/${file})
endforeach ()
foreach (file ${math_core_SRC})
    list(APPEND math_SRC core/${file})
endforeach ()
foreach (file ${math_modules_SRC})
    list(APPEND math_SRC modules/${file})
endforeach ()
foreach (file ${math_synthesis_SRC})
    list(APPEND math_SRC synthesis/${file})
endforeach ()

# === Macros to build the oskar-lib math module.
add_library(oskar_math ${math_SRC})
set_target_properties(oskar_math PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

# === Install target for the oskar-lib math library module.
if (NOT WIN32)
    install(TARGETS oskar_math LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    set_target_properties(oskar_math PROPERTIES
        SOVERSION ${oskar-lib_SOVERSION})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN "src" EXCLUDE
        PATTERN "test" EXCLUDE
        PATTERN ".svn" EXCLUDE)
endif ()

# === Recurse into test directory.
add_subdirectory(test)

add_subdirectory(matlab)
