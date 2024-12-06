CXXFLAGS=-std=c++17 -Ofast -march=native -funroll-loops -fopenmp -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-trapping-math -fsingle-precision-constant $(OTHER_CXXFLAGS)
BINDIR=bin
OMP_NUM_THREADS=16

.PHONY: bin clean
bin:
	mkdir -p $(BINDIR)

nn: bin src/main.cpp
	@echo "### Compiling neuronka ###"
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(BINDIR)/nn
run: nn
	@echo "### Training neuronka ###"
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) nice -n 19 ./$(BINDIR)/nn

clean:
	rm -rf $(BINDIR)
