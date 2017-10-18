#
# oskar/vis/test/CMakeLists.txt
#

set(name vis_test)
set(${name}_SRC
    main.cpp
    Test_Visibilities.cpp
)

if (CASACORE_FOUND)
    list(APPEND ${name}_SRC
        Test_write_ms.cpp
    )
endif ()

add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(${name} ${name})
