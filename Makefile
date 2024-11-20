CXXFLAGS=-std=c++17 -Ofast -march=native -funroll-loops -fopenmp $(OTHER_CXXFLAGS)
BINDIR=bin

.PHONY: bin
bin:
	mkdir -p $(BINDIR)

nn: bin src/main.cpp
	echo "Compiling neuronka"
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(BINDIR)/nn

clean:
	rm -rf $(BINDIR)
