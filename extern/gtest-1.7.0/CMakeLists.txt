#
# extern/gtest-1.7.0/CMakeLists.txt
#

include(cmake/internal_utils.cmake)
config_compiler_and_linker()

include_directories(
  ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0
  ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include
)

cxx_library(gtest "${cxx_strict}" STATIC src/gtest-all.cc)
cxx_library(gtest_main "${cxx_strict}" STATIC src/gtest_main.cc)
target_link_libraries(gtest_main gtest)
