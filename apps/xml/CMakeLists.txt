#
# apps/xml/CMakeLists.txt
#

include(oskar_settings_xml_utility)

file(GLOB SettingsXML *.xml)

# Copy settings XML file(s) to the build directory.
foreach (file_ ${SettingsXML})
    get_filename_component(name_ ${file_} NAME)
    set(xml_file_ ${CMAKE_CURRENT_BINARY_DIR}/${name_})
    configure_file(${file_} ${xml_file_} @ONLY)
endforeach()

# Process import nodes in application XML files.
process_import_nodes(${CMAKE_CURRENT_BINARY_DIR}/oskar_sim_interferometer.xml)
process_import_nodes(${CMAKE_CURRENT_BINARY_DIR}/oskar_sim_beam_pattern.xml)
process_import_nodes(${CMAKE_CURRENT_BINARY_DIR}/oskar_sim_tec_screen.xml)
process_import_nodes(${CMAKE_CURRENT_BINARY_DIR}/oskar_imager.xml)
process_import_nodes(${CMAKE_CURRENT_BINARY_DIR}/oskar_fit_element_data.xml)
