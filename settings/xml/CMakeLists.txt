#
# src/settings/xml/CMakeLists.txt
#

FILE(GLOB SettingsXML *.xml)

foreach (file_ ${SettingsXML})
    get_filename_component(name_ ${file_} NAME)
    configure_file(${file_} ${CMAKE_CURRENT_BINARY_DIR}/${name_} @ONLY)
endforeach()

