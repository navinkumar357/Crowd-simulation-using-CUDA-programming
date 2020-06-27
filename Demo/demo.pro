TEMPLATE = app
TARGET = Demo\Demo
DEPENDPATH += . release debug
INCLUDEPATH += . ..\Libpedsim\src
DESTDIR = .\

QMAKE_CXXFLAGS += 
LIBS += -llibpedsim Qt5PlatformSupport.lib -L..\\$(Platform)\\$(Configuration)  -LC:\Progra~1\NVIDIA~2\CUDA\v7.5\lib\x64\  -lcudart
#LIBS += -L"C:\\Folder\\Folder2\\LibFolder" -lextlib.lib

DEFINES += NOMINMAX

QT += opengl
QT += widgets

CONFIG += embed_manifest_exe
CONFIG += $$DEBUGRELEASE
CONFIG += console

# Input
HEADERS += src\MainWindow.h src\ParseScenario.h  src\ViewAgent.h src\PedSimulation.h 
SOURCES += src\main.cpp src\MainWindow.cpp src\ParseScenario.cpp src\ViewAgent.cpp src\PedSimulation.cpp