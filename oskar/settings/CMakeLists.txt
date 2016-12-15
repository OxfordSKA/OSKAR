#
# settings/CMakeLists.txt
#

add_subdirectory(utility)
add_subdirectory(types)
add_subdirectory(containers)
add_subdirectory(apps)

# DEPRECATED Settings load library (uses Qt)
if (QT4_FOUND)
    include_directories(${QT_INCLUDE_DIR})

    set(name oskar_settings_load)
    set(${name}_SRC
        load/src/oskar_settings_load_tid_parameter_file.c
        load/src/oskar_settings_load_ionosphere.cpp
        load/src/oskar_settings_load_observation.cpp
        load/src/oskar_settings_load.cpp
    )

    add_library(${name} STATIC ${${name}_SRC})
    target_link_libraries(${name} ${QT_QTCORE_LIBRARY})
endif()
