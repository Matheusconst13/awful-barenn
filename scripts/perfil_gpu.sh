#!/usr/bin/env bash
# Uso: scripts/perfil_gpu.sh <binario> <diretorio_saida> [regex_kernels_ncu]
#
# Gera os perfis de GPU de uma versão:
#   nsys_resumo.txt  Nsight Systems: tempo das chamadas da API CUDA, dos
#                    kernels e das cópias de memória (PCIe)
#   ncu.txt          Nsight Compute: métricas dos kernels (só como root, pois
#                    os contadores da GPU exigem privilégio; ex.: sudo -E)
# O ncu fixa o clock da GPU no valor base (--clock-control base, padrão), então
# as durações dele são maiores que as do nsys e servem para comparar versões.
set -eu

binario=$1
saida=$2
kernels=${3:-.}
CUDA_BIN=${CUDA_BIN:-/usr/local/cuda/bin}
mkdir -p "$saida"
temporario=$(mktemp -d)
trap 'rm -rf "$temporario"' EXIT

"$CUDA_BIN/nsys" profile -t cuda,osrt --stats=false -f true \
    -o "$temporario/perfil" "$binario" > /dev/null 2>&1
"$CUDA_BIN/nsys" stats -q -f column \
    -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    "$temporario/perfil.nsys-rep" > "$saida/nsys_resumo.txt"

if [ "$(id -u)" = 0 ]; then
    "$CUDA_BIN/ncu" -k "regex:$kernels" -c 12 \
        --section SpeedOfLight --section Occupancy --section LaunchStats \
        --section WarpStateStats --section SchedulerStats \
        --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
        --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum \
        "$binario" > "$saida/ncu.txt" 2>&1
fi
