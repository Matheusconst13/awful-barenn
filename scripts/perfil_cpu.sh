#!/usr/bin/env bash
# Uso: scripts/perfil_cpu.sh <binario> <diretorio_saida>
#
# Gera os perfis de CPU de uma versão:
#   perf_stat.txt         contadores de hardware (5 repetições)
#   perf_report.txt       funções mais caras (perf record, amostragem por ciclos)
#   callgrind_funcoes.txt instruções e misses simulados de cache por função
#   callgrind_linhas.txt  instruções por linha de barenn.cpp
# Requer perf (kernel.perf_event_paranoid <= 1) e valgrind.
set -eu

binario=$1
saida=$2
mkdir -p "$saida"
temporario=$(mktemp -d)
trap 'rm -rf "$temporario"' EXIT

perf stat -r 5 -o "$saida/perf_stat.txt" \
    -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses \
    -e LLC-loads,LLC-load-misses,branches,branch-misses,task-clock \
    "$binario" > /dev/null

perf record -q -o "$temporario/perf.data" "$binario" > /dev/null 2>&1
perf report -i "$temporario/perf.data" --no-children --stdio --sort symbol \
    2> /dev/null | grep -E '^ +[0-9]' | head -15 | sed 's/ *$//' \
    > "$saida/perf_report.txt"

valgrind --tool=callgrind --cache-sim=yes \
    --callgrind-out-file="$temporario/callgrind.out" "$binario" > /dev/null 2>&1
callgrind_annotate --auto=no --inclusive=no --threshold=99 "$temporario/callgrind.out" \
    | sed "s#$PWD/##g" > "$saida/callgrind_funcoes.txt"
callgrind_annotate --auto=yes --show=Ir,Dr,D1mr "$temporario/callgrind.out" \
    barenn.cpp | sed "s#$PWD/##g" > "$saida/callgrind_linhas.txt"
