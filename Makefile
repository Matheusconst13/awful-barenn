# Compilação, benchmark e validação do Barenn.
#   make            -> compila a versão CPU (./barenn)
#   make bench      -> 10 execuções, resume média/desvio/mediana do treino
#   make validate   -> compara saída e pesos finais com a referência do legado
#   make callgrind  -> perfil por função (valgrind --tool=callgrind)

CXX      ?= g++
# -march=x86-64-v3: AVX2 em qualquer x86-64 recente (não só nesta máquina).
# -ffp-contract=off: proíbe fundir a*b+c em FMA, que arredonda uma vez só e
# mudaria os resultados em relação ao legado.
CXXFLAGS ?= -std=c++17 -O3 -march=x86-64-v3 -ffp-contract=off -g
REPS     ?= 10

all: barenn

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

bench: barenn
	python3 scripts/bench.py ./barenn -n $(REPS)

validate: barenn
	./scripts/validate.sh ./barenn

callgrind: barenn
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out.barenn ./barenn > /dev/null
	callgrind_annotate --auto=no callgrind.out.barenn | head -40

clean:
	rm -f barenn callgrind.out.barenn

.PHONY: all bench validate callgrind clean
