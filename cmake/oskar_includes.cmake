include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/apps
    ${PROJECT_SOURCE_DIR}/apps/lib
    ${PROJECT_SOURCE_DIR}/apps/log
    ${PROJECT_SOURCE_DIR}/convert
    ${PROJECT_SOURCE_DIR}/correlate
    ${PROJECT_SOURCE_DIR}/element
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include
    ${PROJECT_SOURCE_DIR}/extern/rapidxml-1.13
    ${PROJECT_SOURCE_DIR}/extern/cfitsio-3.37
    ${PROJECT_SOURCE_DIR}/extern/Random123
    ${PROJECT_SOURCE_DIR}/extern/Random123/features
    ${PROJECT_SOURCE_DIR}/imaging
    ${PROJECT_SOURCE_DIR}/interferometry
    ${PROJECT_SOURCE_DIR}/jones
    ${PROJECT_SOURCE_DIR}/math
    ${PROJECT_SOURCE_DIR}/ms
    ${PROJECT_SOURCE_DIR}/settings
    ${PROJECT_SOURCE_DIR}/settings/containers
    ${PROJECT_SOURCE_DIR}/settings/load
    ${PROJECT_SOURCE_DIR}/settings/struct
    ${PROJECT_SOURCE_DIR}/settings/types
    ${PROJECT_SOURCE_DIR}/settings/utility
    ${PROJECT_SOURCE_DIR}/settings/widgets
    ${PROJECT_SOURCE_DIR}/sky
    ${PROJECT_SOURCE_DIR}/splines
    ${PROJECT_SOURCE_DIR}/station
    ${PROJECT_SOURCE_DIR}/utility
    ${PROJECT_SOURCE_DIR}/utility/binary
)

set(GTEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include/gtest)
set(EZOPT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/ezOptionParser-0.2.0)
