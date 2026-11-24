# Compilação, benchmark e validação do Barenn.
#   make            -> compila a versão CPU (./barenn)
#   make cuda       -> compila a versão GPU (./barenn_cuda); idem bench-cuda e
#                      validate-cuda
#   make bench      -> 10 execuções, resume média/desvio/mediana do treino
#   make validate   -> compara saída e pesos finais com a referência do legado
#   make callgrind  -> perfil por função (valgrind --tool=callgrind)

CXX      ?= g++
# -march=x86-64-v3: AVX2 em qualquer x86-64 recente (não só nesta máquina).
# -ffp-contract=off: proíbe fundir a*b+c em FMA, que arredonda uma vez só e
# mudaria os resultados em relação ao legado.
# -fopenmp: leitura dos arquivos em paralelo.
CXXFLAGS ?= -std=c++17 -O3 -march=x86-64-v3 -ffp-contract=off -fopenmp -g
REPS     ?= 10

# GPU: --fmad=false impede a fusão a*b+c em FMA, para que somas e produtos
# arredondem como na CPU (resultados idênticos bit a bit ao legado).
NVCC      ?= $(or $(shell command -v nvcc),/usr/local/cuda/bin/nvcc)
CUDA_ARCH ?= sm_86
NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH) --fmad=false -lineinfo

all: barenn

cuda: barenn_cuda

barenn: barenn.cpp comum.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

barenn_cuda: barenn_cuda.cu comum.hpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" -lgomp -o $@ $<

bench: barenn
	python3 scripts/bench.py ./barenn -n $(REPS)

validate: barenn
	./scripts/validate.sh ./barenn

bench-cuda: barenn_cuda
	python3 scripts/bench.py ./barenn_cuda -n $(REPS)

validate-cuda: barenn_cuda
	./scripts/validate.sh ./barenn_cuda

callgrind: barenn
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out.barenn ./barenn > /dev/null
	callgrind_annotate --auto=no callgrind.out.barenn | head -40

clean:
	rm -f barenn barenn_cuda callgrind.out.barenn

.PHONY: all cuda bench validate bench-cuda validate-cuda callgrind clean experimento-omp

# Experimento da issue #7: OpenMP dentro de cada amostra (versão escalar v0.1).
experimentos/omp_intra_amostra: experimentos/omp_intra_amostra.cpp
	$(CXX) -std=c++17 -O2 -g -fopenmp -o $@ $<

experimento-omp: experimentos/omp_intra_amostra
	for t in 1 2 4 8; do echo "threads=$$t"; \
	  OMP_NUM_THREADS=$$t python3 scripts/bench.py ./$< -n 5 | tail -2; done
