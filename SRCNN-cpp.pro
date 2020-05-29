
#OpenMP on macOS: first install llvm via brew, setup llvm kit & compiler in Qt settings!
macx{
    QMAKE_CC = /usr/local/opt/llvm/bin/clang
    QMAKE_CXX = /usr/local/opt/llvm/bin/clang++
    QMAKE_LINK = /usr/local/opt/llvm/bin/clang++
    QMAKE_CFLAGS += -fopenmp
    QMAKE_CXXFLAGS += -fopenmp
    INCLUDEPATH += -I/usr/local/opt/llvm/include
    LIBS += -L/usr/local/opt/llvm/lib -lomp
}
#OpenMP on Windows
win32: QMAKE_CFLAGS += -O2 -fopenmp -msse4.1 -mssse3 -msse3 -msse2 -msse -D_FILE_OFFSET_BITS=64 -std=c99
win32: QMAKE_CXXFLAGS += -fopenmp
win32: LIBS += -llibgomp-1

#OpenCV
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
macx: LIBS += -L/usr/local/lib/
macx: INCLUDEPATH += /usr/local/include/opencv4

INCLUDEPATH += ./src/

SOURCES += \
    main.cpp \
    src/gaussian.cpp \
    src/srcnn.cpp

HEADERS += \
    src/gaussian.hpp \
    src/srcnn.hpp
