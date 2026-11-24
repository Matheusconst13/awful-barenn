# Barenn: relatório de otimização (CPU e CUDA)

Rede densa 64 → 32 → 16 → 10 treinada com SGD por amostra (51 épocas × 3823 imagens = 194.973 passos sequenciais). **Todas as versões produzem saída e parâmetros finais idênticos bit a bit ao código legado** (`make validate` / `make validate-cuda`).

## Tabela consolidada

Medida em uma única sessão: cada tag compilada a partir de um checkout isolado, `scripts/bench.py -n 10` (1 aquecimento + 10 execuções; média ± desvio). "Treino" é o cronômetro interno do programa (o mesmo do legado). Na GPU ele inclui as cópias host↔device e exclui a criação do contexto CUDA. "Total" é o processo inteiro. Saída bruta em [`perfil/consolidado.txt`](perfil/consolidado.txt).

| Categoria | Versão (tag) | Treino | Speedup treino | Total (processo) | Validação |
|:--|:--|--:|--:|--:|:--:|
| **Baseline legado** | [`v0.0-baseline`](../../tree/v0.0-baseline) | 0,859 s ± 0,009 | 1,0x | 0,891 s | bit a bit |
| CPU | [`v0.1-sem-copias`](../../tree/v0.1-sem-copias) | 0,601 s ± 0,014 | 1,4x | 0,634 s | bit a bit |
| CPU | [`v0.2-simd`](../../tree/v0.2-simd) | 0,0640 s ± 0,0009 | 13,4x | 0,093 s | bit a bit |
| **CPU otimizada** | [`v0.3-multicore`](../../tree/v0.3-multicore) | **0,0637 s ± 0,0007** | **13,5x** | **0,069 s** (mediana) | bit a bit |
| **CUDA inicial** | [`v1.0-cuda`](../../tree/v1.0-cuda) | 5,709 s ± 0,123 | 0,15x | 5,900 s | bit a bit |
| CUDA | [`v1.1-persistente`](../../tree/v1.1-persistente) | 0,331 s ± 0,000 | 2,6x | 0,506 s | bit a bit |
| **CUDA otimizado** | [`v1.2-warp`](../../tree/v1.2-warp) | **0,171 s ± 0,000** | **5,0x** | 0,340 s | bit a bit |

Ambiente: Intel i7-11800H (8C/16T, até 4,6 GHz), 16 GB, NVIDIA GeForce RTX 3060 Laptop (GA106, sm_86, 30 SMs, 6 GB), Ubuntu 22.04 (kernel 6.8), GCC 11.4, CUDA 12.9, driver 580.178. Entrada: `optdigits.tra` / `optdigits.tes`. Flags: CPU `-O3 -march=x86-64-v3 -ffp-contract=off -fopenmp`; GPU `nvcc -O3 -arch=sm_86 --fmad=false`. Na GPU o "total" inclui ~0,15 s de criação do contexto CUDA.

## Índice (issue → PR → tag)

| # | Bloco lógico | Issue (hipótese, medições, discussão) | PR | Tag |
|:--:|:--|:--|:--|:--|
| 1 | Baseline reproduzível, validação bit a bit, perf/callgrind | [#1](../../issues/1) | [#2](../../pull/2) | [`v0.0-baseline`](../../tree/v0.0-baseline) |
| 2 | Sem cópias/alocações e sem o gradiente descartado | [#3](../../issues/3) | [#4](../../pull/4) | [`v0.1-sem-copias`](../../tree/v0.1-sem-copias) |
| 3 | Layout transposto contíguo + SIMD AVX2 sem reordenar somas | [#5](../../issues/5) | [#6](../../pull/6) | [`v0.2-simd`](../../tree/v0.2-simd) |
| 4 | Multicore: leitura paralela (OpenMP), `from_chars`; OpenMP no SGD refutado | [#7](../../issues/7) | [#8](../../pull/8) | [`v0.3-multicore`](../../tree/v0.3-multicore) |
| 5 | Migração funcional para CUDA (porting direto) | [#9](../../issues/9) | [#10](../../pull/10) | [`v1.0-cuda`](../../tree/v1.0-cuda) |
| 6 | CUDA: kernel persistente + memória compartilhada | [#11](../../issues/11) | [#12](../../pull/12) | [`v1.1-persistente`](../../tree/v1.1-persistente) |
| 7 | CUDA: 1 warp, registradores, shuffles, padding, prefetch; streams e vazão (experimentos) | [#13](../../issues/13) | [#14](../../pull/14) | [`v1.2-warp`](../../tree/v1.2-warp) |
| 8 | Este sumário | [#15](../../issues/15) | [#16](../../pull/16) | [`v2.0-relatorio`](../../tree/v2.0-relatorio) |

## Conclusões de engenharia

- **O gargalo do legado era a contagem de instruções, não a memória** (IPC 3,44, 0,04% de miss na L1d). Os ganhos vieram de remover trabalho morto e cópias (#3) e de **vetorizar entre neurônios**, o que preserva a ordem de cada soma e o resultado bit a bit (#5). O treino roda a ~65% do pico AVX2 de um núcleo.
- **O SGD por amostra é 100% serial entre amostras (Amdahl).** OpenMP dentro da amostra deixou o treino 3x mais lento (barreiras de ~0,7 µs para ~0,3 µs de trabalho, #7). O multicore foi aplicado onde há independência: a leitura dos arquivos.
- **Na GPU, o custo dominante é a latência, não a banda nem o cálculo.** O porting direto fazia 2,15 M lançamentos e 392 mil cópias síncronas de 40 bytes (#9). Um kernel persistente (#11) e depois uma warp sem barreiras, com registradores e shared sem conflitos (#13), levaram a GPU de 5,7 s para 0,171 s (**33x**). Streams não ajudam quando o PCIe é 0,2% do tempo.
- **A CPU otimizada vence a GPU em um único treino (0,064 s × 0,171 s):** as duas gastam ~1 400–1 700 ciclos por amostra no caminho crítico, e a CPU roda a 4,4 GHz. A GPU fica ~97% ociosa: com **30 modelos independentes** ela leva o mesmo tempo que com 1, e com 960 chega a 2,5 ms por modelo (~25x a vazão de um núcleo, #13). O datacenter com GPUs compensa para muitos treinos independentes ou para um algoritmo em mini-batch, que exige aprovar a mudança no resultado numérico.

## Como reproduzir

```bash
git checkout v1.2-warp          # ou qualquer tag do índice
make && make validate && make bench
make cuda && make validate-cuda && make bench-cuda
./scripts/perfil_cpu.sh ./barenn perfil/x          # perf stat/record + callgrind
./scripts/perfil_gpu.sh ./barenn_cuda perfil/y     # nsys (e ncu, se root)
make experimento-omp experimento-streams experimento-modelos
```

---

# Enunciado original

## Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
