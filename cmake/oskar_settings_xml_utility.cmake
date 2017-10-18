# Macro to combine settings XML into a single file by processing
# <import> nodes.
macro(process_import_nodes FILENAME)

    if (NOT EXISTS ${FILENAME})
        message(FATAL_ERROR "Specified XML file '${filename}' not found!")
    endif()

    # Read the file into a string.
    file(READ ${FILENAME} xml_)

    set(iter 0)
    set(import_nodes_remaining YES)
    while (${import_nodes_remaining})

        # Obtain a list and number of files to import.
        unset(import_files_)
        string(REGEX MATCHALL "([-/A-Za-z0-9_\\.]+\\.xml)" import_files_ "${xml_}")
        list(LENGTH import_files_ n)

        if (${n} EQUAL 0)
            set(import_nodes_remaining NO)
        else()
            set(iIn 0)
            foreach(i IN LISTS import_files_)

                # Get the full file path and check that it exists
                get_filename_component(import_xml_file ${i} ABSOLUTE)
                if (NOT EXISTS ${import_xml_file})
                     message(FATAL_ERROR "XML import file '${i}' not found!")
                endif()

                # Read the import file into the string xml_new_
                file(READ ${import_xml_file} new_xml_)
                get_filename_component(import_name_ ${import_xml_file} NAME_WE)

                # Extract the top level node from the import file
                string(REGEX MATCH "(<s.*s>)" import_node_ "${new_xml_}")
                # Create an import node string by prefixing with a comment
                set(import_node_ "
<!-- BEGIN [${import_name_} iter:${iter},import:${iIn}] -->
${import_node_}
<!-- END [${import_name_}] -->
")

                # Update the xml_ string by replacing the import node.
                #string(REGEX REPLACE <regex> <replace_expression> <output variable> <input>)
                string(REGEX REPLACE "(<import filename=\"${i}\"[ ]?/>)" "${import_node_}" xml_ "${xml_}")

                math(EXPR iIn "${iIn}+1")
            endforeach()
        endif ()

        # Fail if import node iteration (depth) is > 10
        if (${iter} EQUAL 10)
            set(import_nodes_remaining NO)
            message(FATAL_ERROR "Settings XML concatenation failed")
        endif()

        math(EXPR iter "${iter}+1")
    endwhile()

    # Write out the combined XML file
    get_filename_component(name_ ${FILENAME} NAME_WE)
    set(xml_file ${CMAKE_CURRENT_BINARY_DIR}/${name_}_all.xml)
    #message(STATUS "Writing combined settings XML file: ${xml_file}")
    file(WRITE ${xml_file} "${xml_}")

    # Write a C header with the combined XML as a string
    get_filename_component(name_ ${FILENAME} NAME_WE)
    set(xml_file ${CMAKE_CURRENT_BINARY_DIR}/${name_}_xml_all.h)
    string(REPLACE "\"" "\\\"" xml_h_ "${xml_}")
    string(REPLACE "\n" "\"\n\"\\n" xml_h_ "${xml_h_}")
    set(xml_h_
"
static const char xml[] =
\"${xml_h_}\";
")
    #message(STATUS "Writing settings XML header: ${xml_file}")
    file(WRITE ${xml_file} "${xml_h_}")

endmacro()
