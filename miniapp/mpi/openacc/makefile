CXX = CC
CPPFLAGS = -I$(MPICH_DIR)/include
CXXFLAGS = -std=c++11 -O3 -acc -Minfo=accel -Mcuda -ta=tesla:cc60,pinned
LDFLAGS  = -L$(CUDATOOLKIT_HOME)/lib64

SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp   main.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o     main.o      cuda_helpers.o

.SUFFIXES: .cpp

all: main

cuda_helpers.o: cuda_helpers.cpp cuda_helpers.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c cuda_helpers.cpp

stats.o: stats.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c stats.cpp

data.o: data.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c data.cpp

operators.o: operators.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c operators.cpp

linalg.o: linalg.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c linalg.cpp

main.o: main.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c main.cpp

main: $(OBJ) unit_tests.h
	$(CXX) $(CXXFLAGS) $(OBJ) -o main $(LDFLAGS) -lcublas

clean:
	rm -f main
	rm -f *.o
	rm -f *.i
	rm -f output.*
