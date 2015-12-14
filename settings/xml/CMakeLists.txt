#
# src/settings/xml/CMakeLists.txt
#

include(oskar_settings_xml_utility)

file(GLOB SettingsXML *.xml)

# Copy settings XML file(s) to the build directory.
foreach (file_ ${SettingsXML})
    get_filename_component(name_ ${file_} NAME)
    set(xml_file_ ${CMAKE_CURRENT_BINARY_DIR}/${name_})
    configure_file(${file_} ${xml_file_} @ONLY)
endforeach()

# Create a single combined oskar.xml file called 'oskar_all.xml'
# found in the settings/xml folder of the build directory.
process_import_nodes(${CMAKE_CURRENT_BINARY_DIR}/oskar.xml)

