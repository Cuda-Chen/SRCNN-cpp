
#OpenMP on macOS: first install llvm via brew, setup llvm kit & compiler in Qt settings!
macx{
    QMAKE_CC = /usr/local/opt/llvm/bin/clang
    QMAKE_CXX = /usr/local/opt/llvm/bin/clang++
    QMAKE_LINK = /usr/local/opt/llvm/bin/clang++
    QMAKE_CFLAGS += -fopenmp
    QMAKE_CXXFLAGS += -fopenmp
    INCLUDEPATH += -I/usr/local/opt/llvm/include
    LIBS += -L/usr/local/opt/llvm/lib -lomp
    QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.8
}

INCLUDEPATH += ./src/ \
               /usr/local/include/opencv4

LIBS += -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

SOURCES += \
    main.cpp \
    src/gaussian.cpp \
    src/srcnn.cpp

HEADERS += \
    src/gaussian.hpp \
    src/srcnn.hpp
