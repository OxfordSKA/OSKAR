#
# doc/CMakeLists.txt
#

find_package(Sphinx)
find_package(PythonInterp)

# Define a function to add build targets for the Sphinx documentation.
# It will create build targets based on the target root name,
# plus "_html" and "_pdf" to build the HTML and PDF versions.
#
# add_sphinx_doc(
#     TARGET_ROOT <target root name>
#     DOC_SOURCE_DIR <source directory>
# )
#
function(add_sphinx_doc)
    if (Sphinx_FOUND AND PythonInterp_FOUND)
        set(options)
        set(oneValueArgs TARGET_ROOT DOC_SOURCE_DIR)
        set(multiValueArgs)
        cmake_parse_arguments(SPHINX
            "${options}"
            "${oneValueArgs}"
            "${multiValueArgs}"
            ${ARGN})

        if (NOT SPHINX_TARGET_ROOT)
            message(FATAL_ERROR "ERROR: add_sphinx_doc() requires a valid TARGET_ROOT.")
        endif()
        if (NOT SPHINX_DOC_SOURCE_DIR)
            message(FATAL_ERROR "ERROR: add_sphinx_doc() requires a valid DOC_SOURCE_DIR.")
        endif()
        if (SPHINX_UNPARSED_ARGUMENTS)
            message(FATAL_ERROR "ERROR: Unexpected arguments passed to add_sphinx_doc(): '${SPHINX_UNPARSED_ARGUMENTS}'")
        endif()

        set(SPHINX_BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
        set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

        configure_file("${SPHINX_DOC_SOURCE_DIR}/conf.py"
            "${SPHINX_BINARY_BUILD_DIR}/conf.py" @ONLY)

        # message("-- Using Python: ${PYTHON_EXECUTABLE}")
        add_custom_target(${SPHINX_TARGET_ROOT}_html
            COMMAND "${PYTHON_EXECUTABLE}"
                "${PROJECT_BINARY_DIR}/docs/settings/xml_to_rst.py"
            COMMAND "${SPHINX_EXECUTABLE}"
                -M html
                "${SPHINX_DOC_SOURCE_DIR}"
                "${SPHINX_BINARY_BUILD_DIR}"
                -c "${SPHINX_BINARY_BUILD_DIR}"
                -d "${SPHINX_CACHE_DIR}"
            COMMENT "Building HTML documentation with Sphinx"
        )
        add_custom_target(${SPHINX_TARGET_ROOT}_pdf
            COMMAND "${PYTHON_EXECUTABLE}"
                "${PROJECT_BINARY_DIR}/docs/settings/xml_to_rst.py"
            COMMAND "${SPHINX_EXECUTABLE}"
                -M latexpdf
                "${SPHINX_DOC_SOURCE_DIR}"
                "${SPHINX_BINARY_BUILD_DIR}"
                -c "${SPHINX_BINARY_BUILD_DIR}"
                -d "${SPHINX_CACHE_DIR}"
            COMMENT "Building PDF documentation with Sphinx"
        )
    else()
        message("-- WARNING: Sphinx or Python not found - cannot build documentation")
    endif()
endfunction()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/settings/xml_to_rst.py
    ${CMAKE_CURRENT_BINARY_DIR}/settings/xml_to_rst.py @ONLY)

add_sphinx_doc(
    TARGET_ROOT doc
    DOC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
)
