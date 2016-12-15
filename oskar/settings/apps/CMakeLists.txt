#
# settings/apps/CMakeLists.txt
#

set(name oskar_settings_apps)
set(${name}_SRC
    src/oskar_settings_log.cpp
    src/oskar_settings_to_sky.cpp
    src/oskar_settings_to_telescope.cpp
)
add_library(${name} STATIC ${${name}_SRC})
target_link_libraries(${name} oskar_settings_containers)
