#
# settings/types/CMakeLists.txt
#

set(name oskar_settings_types)
file(GLOB ${name}_SRC src/*.cpp)
add_library(${name} STATIC ${${name}_SRC})
target_link_libraries(${name} oskar_settings_utility)
add_subdirectory(test)

