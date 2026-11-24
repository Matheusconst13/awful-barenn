// Barenn na GPU com kernel persistente de uma única warp. O SGD por amostra
// é uma cadeia de dependências (a amostra n + 1 usa os pesos atualizados pela
// amostra n), então o tempo é dominado pela latência de cada passo, não pela
// vazão. Com 32 threads de uma warp:
//   - cada lane é dona de um neurônio da camada 1 e guarda os 64 pesos dele
//     em registradores;
//   - os valores trocados entre neurônios passam por warp shuffles;
//   - as camadas 2 e 3 ficam na memória compartilhada, com passo ímpar entre
//     linhas para não haver conflito de banco;
//   - a sincronização é só da warp (__syncwarp), sem __syncthreads;
//   - a imagem da próxima amostra é lida antecipadamente (prefetch).
//
// Compilar com --fmad=false: sem a fusão a*b+c em FMA, cada soma e cada
// produto arredondam exatamente como na CPU e os resultados são idênticos
// bit a bit aos do legado.
#include <chrono>

#include <cuda_runtime.h>

#include "comum.hpp"

#define CUDA_CHECK(chamada)                                                  \
    do {                                                                     \
        const cudaError_t erro_cuda = (chamada);                             \
        if (erro_cuda != cudaSuccess) {                                      \
            throw runtime_error(string(#chamada) + ": " +                    \
                                cudaGetErrorString(erro_cuda));              \
        }                                                                    \
    } while (0)

namespace Barenn {

// Dimensões da rede 64 -> 32 -> 16 -> 10.
constexpr int n1 = 32;
constexpr int n2 = 16;
// Parâmetros na memória global, no layout do legado e na ordem do arquivo de
// validação: w1[32][64], b1, w2[16][32], b2, w3[10][16], b3.
struct Parametros {
    static constexpr int w1 = 0;
    static constexpr int b1 = w1 + n1 * entradas;
    static constexpr int w2 = b1 + n1;
    static constexpr int b2 = w2 + n2 * n1;
    static constexpr int w3 = b2 + n2;
    static constexpr int b3 = w3 + classes * n2;
};

constexpr unsigned warp_inteira = 0xffffffffu;
// Passo entre as linhas (uma por entrada) dos pesos transpostos na memória
// compartilhada. Com passo ímpar, as 32 lanes acessando a linha da sua
// entrada caem em 32 bancos distintos.
constexpr int passo2 = n2 + 1;
constexpr int passo3 = classes + 1;

// Treino completo em uma warp. perdas[e * n + a] recebe a perda da amostra a
// na época e; a média por época é feita no host, em double e na mesma ordem
// do legado.
__global__ void __launch_bounds__(32, 1)
treinar_kernel(const float* __restrict__ imagens, const int* __restrict__ rotulos,
               int n, float* parametros, float* perdas, float learning_rate) {
    // w2[j * passo2 + i] liga a entrada j ao neurônio i (idem w3).
    __shared__ float w2[n1 * passo2];
    __shared__ float w3[n2 * passo3];
    const int lane = threadIdx.x;
    // Neurônio calculado pela lane nas camadas 2 e 3. As lanes sem neurônio
    // repetem o de outra lane (mesmo endereço, sem conflito) e seus
    // resultados são ignorados.
    const int i2 = lane % n2;
    const int i3 = lane < classes ? lane : 0;

    float w1[entradas];
#pragma unroll
    for (int j = 0; j < entradas; ++j) {
        w1[j] = parametros[Parametros::w1 + lane * entradas + j];
    }
    float b1 = parametros[Parametros::b1 + lane];
    float b2 = parametros[Parametros::b2 + i2];
    float b3 = parametros[Parametros::b3 + i3];
    for (int k = lane; k < n2 * n1; k += 32) {
        w2[(k % n1) * passo2 + k / n1] = parametros[Parametros::w2 + k];
    }
    for (int k = lane; k < classes * n2; k += 32) {
        w3[(k % n2) * passo3 + k / n2] = parametros[Parametros::w3 + k];
    }
    __syncwarp();

    // Cada lane guarda as entradas lane e lane + 32 da imagem atual.
    float xa = imagens[lane];
    float xb = imagens[32 + lane];
    for (int epoca = 0; epoca < epocas; ++epoca) {
        for (int a = 0; a < n; ++a) {
            // Prefetch: a leitura da próxima imagem fica em voo enquanto a
            // amostra atual é processada.
            const int proxima = a + 1 < n ? a + 1 : 0;
            const float xa_prox = imagens[proxima * entradas + lane];
            const float xb_prox = imagens[proxima * entradas + 32 + lane];
            const int rotulo = rotulos[a];

            // Camada 1: neurônio lane, entradas na ordem do legado.
            float z1 = b1;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                z1 += __shfl_sync(warp_inteira, xa, j) * w1[j];
            }
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                z1 += __shfl_sync(warp_inteira, xb, j) * w1[32 + j];
            }
            const float a1 = z1 > 0.0f ? z1 : 0.0f;

            // Camada 2: neurônio i2.
            float z2 = b2;
#pragma unroll
            for (int j = 0; j < n1; ++j) {
                z2 += __shfl_sync(warp_inteira, a1, j) * w2[j * passo2 + i2];
            }
            const float a2 = z2 > 0.0f ? z2 : 0.0f;

            // Camada 3: neurônio i3, escore linear.
            float z3 = b3;
#pragma unroll
            for (int j = 0; j < n2; ++j) {
                z3 += __shfl_sync(warp_inteira, a2, j) * w3[j * passo3 + i3];
            }

            // Perda e gradiente: o erro da classe i fica na lane i e a soma
            // segue a ordem de Barenn::perda.
            const float erro = z3 - (lane == rotulo ? 1.0f : 0.0f);
            float total = 0;
#pragma unroll
            for (int i = 0; i < classes; ++i) {
                const float e = __shfl_sync(warp_inteira, erro, i);
                total += 0.5f * e * e;
            }
            if (lane == 0) {
                perdas[epoca * n + a] = total;
            }

            // Camada 3: delta é o próprio erro. A lane j (< 16) lê a linha da
            // entrada j: usa o peso antigo no gradiente e então o atualiza.
            float g3 = 0;
#pragma unroll
            for (int i = 0; i < classes; ++i) {
                const float d3 = __shfl_sync(warp_inteira, erro, i);
                float& peso = w3[i2 * passo3 + i];
                g3 += d3 * peso;
                if (lane < n2) {
                    peso -= learning_rate * (d3 * a2);
                }
            }
            if (lane < classes) {
                b3 -= learning_rate * erro;
            }

            // Camada 2: a lane j (< 32) cuida da linha da entrada j.
            const float d2 = g3 * (z2 > 0.0f ? 1.0f : 0.0f);
            float g2 = 0;
#pragma unroll
            for (int i = 0; i < n2; ++i) {
                const float d = __shfl_sync(warp_inteira, d2, i);
                float& peso = w2[lane * passo2 + i];
                g2 += d * peso;
                peso -= learning_rate * (d * a1);
            }
            if (lane < n2) {
                b2 -= learning_rate * d2;
            }

            // Camada 1: cada lane atualiza os pesos do seu neurônio.
            const float d1 = g2 * (z1 > 0.0f ? 1.0f : 0.0f);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                w1[j] -= learning_rate * (d1 * __shfl_sync(warp_inteira, xa, j));
            }
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                w1[32 + j] -= learning_rate * (d1 * __shfl_sync(warp_inteira, xb, j));
            }
            b1 -= learning_rate * d1;

            // Os pesos de w2 e w3 escritos por uma lane são lidos por outras
            // no forward da próxima amostra.
            __syncwarp();
            xa = xa_prox;
            xb = xb_prox;
        }
    }

#pragma unroll
    for (int j = 0; j < entradas; ++j) {
        parametros[Parametros::w1 + lane * entradas + j] = w1[j];
    }
    parametros[Parametros::b1 + lane] = b1;
    if (lane < n2) {
        parametros[Parametros::b2 + lane] = b2;
    }
    if (lane < classes) {
        parametros[Parametros::b3 + lane] = b3;
    }
    for (int k = lane; k < n2 * n1; k += 32) {
        parametros[Parametros::w2 + k] = w2[(k % n1) * passo2 + k / n1];
    }
    for (int k = lane; k < classes * n2; k += 32) {
        parametros[Parametros::w3 + k] = w3[(k % n2) * passo3 + k / n2];
    }
}

// Inferência: as imagens de teste são independentes, uma thread por imagem.
__global__ void inferir_kernel(const float* imagens, int n,
                               const float* parametros, float* escores) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) {
        return;
    }
    const float* x = imagens + a * entradas;
    float h1[n1], h2[n2];
    for (int i = 0; i < n1; ++i) {
        float sum = parametros[Parametros::b1 + i];
        for (int j = 0; j < entradas; ++j) {
            sum += x[j] * parametros[Parametros::w1 + i * entradas + j];
        }
        h1[i] = sum > 0.0f ? sum : 0.0f;
    }
    for (int i = 0; i < n2; ++i) {
        float sum = parametros[Parametros::b2 + i];
        for (int j = 0; j < n1; ++j) {
            sum += h1[j] * parametros[Parametros::w2 + i * n1 + j];
        }
        h2[i] = sum > 0.0f ? sum : 0.0f;
    }
    for (int i = 0; i < classes; ++i) {
        float sum = parametros[Parametros::b3 + i];
        for (int j = 0; j < n2; ++j) {
            sum += h2[j] * parametros[Parametros::w3 + i * n2 + j];
        }
        escores[a * classes + i] = sum;
    }
}

// Buffer na memória global da GPU, liberado ao sair do escopo.
template <typename T>
class BufferGPU {
    T* dados = nullptr;

public:
    explicit BufferGPU(size_t n) { CUDA_CHECK(cudaMalloc(&dados, n * sizeof(T))); }
    BufferGPU(const vector<T>& origem) : BufferGPU(origem.size()) {
        CUDA_CHECK(cudaMemcpy(dados, origem.data(), origem.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    BufferGPU(const BufferGPU&) = delete;
    BufferGPU& operator=(const BufferGPU&) = delete;
    ~BufferGPU() { cudaFree(dados); }

    T* get() const { return dados; }
    void baixar(vector<T>& destino) const {
        CUDA_CHECK(cudaMemcpy(destino.data(), dados, destino.size() * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }
};

// Pesos iniciais (sorteados como no legado) e vieses zero, no layout de
// Parametros.
vector<float> parametros_iniciais() {
    vector<float> parametros;
    for (auto [in, out] : {pair{entradas, n1}, pair{n1, n2}, pair{n2, classes}}) {
        const vector<float> pesos = pesos_iniciais(in, out);
        parametros.insert(parametros.end(), pesos.begin(), pesos.end());
        parametros.insert(parametros.end(), out, 0.0f);
    }
    return parametros;
}

} // namespace Barenn

int main(int argc, char** argv) {
    using namespace Barenn;
    try {
        Conjunto treino, teste;
        if (!carregar(argc, argv, treino, teste)) {
            return 1;
        }
        const int n = static_cast<int>(treino.size());

        // Cria o contexto CUDA fora do cronômetro (custo fixo de ~0,1 s).
        CUDA_CHECK(cudaFree(nullptr));
        vector<float> parametros = parametros_iniciais();
        vector<float> perdas_amostra(static_cast<size_t>(epocas) * n);

        // A medição inclui as cópias para a GPU, o treino de todas as épocas
        // e a cópia das perdas e dos parâmetros finais de volta ao host.
        const auto inicio = chrono::steady_clock::now();
        BufferGPU<float> imagens(treino.imagens);
        BufferGPU<int> rotulos(treino.rotulos);
        BufferGPU<float> parametros_gpu(parametros);
        BufferGPU<float> perdas_gpu(perdas_amostra.size());
        treinar_kernel<<<1, 32>>>(imagens.get(), rotulos.get(), n,
                                              parametros_gpu.get(),
                                              perdas_gpu.get(), taxa);
        CUDA_CHECK(cudaGetLastError());
        perdas_gpu.baixar(perdas_amostra);
        parametros_gpu.baixar(parametros);
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        array<double, epocas> perdas{};
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            for (int a = 0; a < n; ++a) {
                perda_total += perdas_amostra[static_cast<size_t>(epoca) * n + a];
            }
            perdas[epoca] = perda_total / n;
        }

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        imprimir_treino(perdas, treino.size(), tempo.count());
        gravar_parametros(parametros);
        const int n_teste = static_cast<int>(teste.size());
        BufferGPU<float> imagens_teste(teste.imagens);
        BufferGPU<float> escores_gpu(static_cast<size_t>(n_teste) * classes);
        inferir_kernel<<<(n_teste + 127) / 128, 128>>>(
            imagens_teste.get(), n_teste, parametros_gpu.get(), escores_gpu.get());
        CUDA_CHECK(cudaGetLastError());
        vector<float> escores(static_cast<size_t>(n_teste) * classes);
        escores_gpu.baixar(escores);
        relatorio(teste, escores.data(), classes);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
