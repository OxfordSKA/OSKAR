#
# /src/math/core/CMakeLists.txt
#
#==============================================================================

# === Source files for the math core sub-module.
set(math_core_SRC
    src/oskar_math_core_ctrimat.c
)
set(math_core_SRC "${math_core_SRC}" PARENT_SCOPE)

# === Recurse into test subdirectory.
add_subdirectory(test)
