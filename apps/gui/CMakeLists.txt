#
# src/apps/gui/CMakeLists.txt
#
include_directories(${QT_INCLUDE_DIR})

set(gui_MOC_SRC
    oskar_About.h
    oskar_BinaryLocations.h
    oskar_CudaInfoDisplay.h
    oskar_DocumentationDisplay.h
    oskar_MainWindow.h
    oskar_RunDialog.h
    oskar_RunThread.h
)
set(gui_SRC
    src/oskar_About.cpp
    src/oskar_BinaryLocations.cpp
    src/oskar_CudaInfoDisplay.cpp
    src/oskar_DocumentationDisplay.cpp
    src/oskar_MainWindow.cpp
    src/oskar_RunDialog.cpp
    src/oskar_RunThread.cpp
)

set(gui_SRC "${gui_SRC}" PARENT_SCOPE)
set(gui_MOC_SRC "${gui_MOC_SRC}" PARENT_SCOPE)
