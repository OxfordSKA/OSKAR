#
# src/apps/CMakeLists.txt
#

set(apps_SRC
    src/oskar_load_stations.cpp
)
set(apps_SRC "${apps_SRC}" PARENT_SCOPE)

# ==== Macros to build the oskar-lib platform module.
include_directories(${QT_INCLUDE_DIR})
add_library(oskar_apps STATIC ${apps_SRC})
target_link_libraries(oskar_apps ${QT_QTCORE_LIB} oskar)
set_target_properties(oskar_apps PROPERTIES LINKER_LANGUAGE CXX)

# ==== Recurse into test subdirectory.
add_subdirectory(test)
