#!/usr/bin/env bash
# Uso: scripts/validate.sh <binario>
#
# Valida a corretude de uma versão contra a referência gerada pelo código
# legado (tag v0.0-baseline):
#   1. a saída padrão deve ser idêntica, exceto a linha "Tempo de treino";
#   2. os pesos e vieses finais (BARENN_PESOS) devem ser idênticos bit a bit.
set -euo pipefail

binario=$1
referencia=$(dirname "$0")/../referencia
temporario=$(mktemp -d)
trap 'rm -rf "$temporario"' EXIT

BARENN_PESOS="$temporario/pesos.bin" "$binario" > "$temporario/saida.txt"

if ! diff <(grep -v '^Tempo de treino' "$referencia/saida_legado.txt") \
          <(grep -v '^Tempo de treino' "$temporario/saida.txt"); then
    echo "FALHA: saída diverge da referência" >&2
    exit 1
fi
if ! cmp "$referencia/pesos_legado.bin" "$temporario/pesos.bin"; then
    echo "FALHA: pesos finais diferem da referência" >&2
    exit 1
fi
echo "OK: saída e $(( $(stat -c %s "$temporario/pesos.bin") / 4 )) parâmetros finais idênticos à referência"
