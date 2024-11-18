CXXFLAGS=-std=c++17 -pedantic -Wall -Wextra -O3 -march=native -funroll-loops $(OTHER_CXXFLAGS)
BINDIR=bin

.PHONY: bin
bin:
	mkdir -p $(BINDIR)

nn: bin src/main.cpp
	echo "Compiling neuronka"
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(BINDIR)/nn

clean:
	rm -rf $(BINDIR)
