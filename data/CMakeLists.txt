#
# src/data
#

set(data_SRC
)

include_directories(
    ${QT_QTCORE_INCLUDE_DIR}
)

add_library(data ${data_SRC})

target_link_libraries(data
    ${QT_QTCORE_LIBRARY}
)

# Recurse into test subdirectory.
add_subdirectory(test)
