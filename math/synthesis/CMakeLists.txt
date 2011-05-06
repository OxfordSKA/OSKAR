#
# src/math/synthesis/CMakeLists.txt
#
#==============================================================================

# === Source files for the math synthesis sub-module.
set(math_synthesis_SRC
    src/oskar_math_synthesis_baselines.c
    src/oskar_math_synthesis_xyz2uvw.c
)
set(math_synthesis_SRC "${math_synthesis_SRC}" PARENT_SCOPE)

# === Recurse into test subdirectory.
#add_subdirectory(test)
