#
# src/modules/CMakeLists.txt
#


include_directories(${oskar-lib_SOURCE_DIR}/modules)
include_directories(${oskar-lib_SOURCE_DIR}/math)

add_subdirectory(cuda)

# === Get list of source files required to build the math module.
set(modules_SRC)
foreach (file ${modules_cuda_SRC})
    list(APPEND modules_SRC cuda/${file})
endforeach ()

set(modules_SRC "${modules_SRC}" PARENT_SCOPE)

