include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/oskar

    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include
    ${PROJECT_SOURCE_DIR}/extern/rapidxml-1.13
    ${PROJECT_SOURCE_DIR}/extern/cfitsio-3.37
    ${PROJECT_SOURCE_DIR}/extern/Random123
    ${PROJECT_SOURCE_DIR}/extern/Random123/features

    ${PROJECT_SOURCE_DIR}/apps
    ${PROJECT_SOURCE_DIR}/apps/gui/widgets
    ${PROJECT_SOURCE_DIR}/oskar

    ${PROJECT_SOURCE_DIR}/oskar/apps
    ${PROJECT_SOURCE_DIR}/oskar/settings
    ${PROJECT_SOURCE_DIR}/oskar/settings/containers
    ${PROJECT_SOURCE_DIR}/oskar/settings/load
    ${PROJECT_SOURCE_DIR}/oskar/settings/struct
    ${PROJECT_SOURCE_DIR}/oskar/settings/types
    ${PROJECT_SOURCE_DIR}/oskar/settings/utility
)

set(GTEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include/gtest)
set(EZOPT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/ezOptionParser-0.2.0)
