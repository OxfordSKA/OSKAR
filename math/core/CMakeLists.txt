#
# /src/math/core
#

set(math_core_SRC
    src/dummy.cpp
)

include_directories()

add_library(math-core ${math_core_SRC})

target_link_libraries(math-core
)

# Recurse into test directory.
add_subdirectory(test)
