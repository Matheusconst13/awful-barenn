#!/usr/bin/env python3
"""Roda um binário do Barenn N vezes e resume os tempos.

Uso: scripts/bench.py <binario> [-n 10] [--aquecimento 1] [-- arquivo_treino arquivo_teste]

"treino" é o tempo que o próprio programa mede (laço de épocas);
"total" é o tempo de parede do processo inteiro (leitura, treino e avaliação).
"""
import argparse
import platform
import re
import statistics
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("binario")
parser.add_argument("-n", type=int, default=10, help="repetições medidas")
parser.add_argument("--aquecimento", type=int, default=1,
                    help="execuções descartadas antes da medição")
parser.add_argument("args", nargs="*", help="argumentos repassados ao binário")
opcoes = parser.parse_args()


def executar():
    inicio = time.perf_counter()
    saida = subprocess.run([opcoes.binario, *opcoes.args], check=True,
                           capture_output=True, text=True).stdout
    total = time.perf_counter() - inicio
    treino = float(re.search(r"Tempo de treino ([\d.]+) s", saida).group(1))
    return treino, total


for _ in range(opcoes.aquecimento):
    executar()
amostras = [executar() for _ in range(opcoes.n)]

print(f"{opcoes.binario} | {platform.node()} | n={opcoes.n}")
for nome, indice in (("treino", 0), ("total", 1)):
    valores = [amostra[indice] for amostra in amostras]
    print(f"{nome:6s}: media {statistics.mean(valores):.4f} s"
          f" | desvio {statistics.stdev(valores):.4f} s"
          f" | mediana {statistics.median(valores):.4f} s"
          f" | min {min(valores):.4f} s | max {max(valores):.4f} s")
