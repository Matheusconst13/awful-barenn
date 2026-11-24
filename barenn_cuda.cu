// Barenn na GPU com kernel persistente: um único bloco executa todas as
// épocas do treino, com pesos, vieses e ativações na memória compartilhada.
// O SGD por amostra é uma cadeia de dependências (a amostra n + 1 usa os
// pesos atualizados pela amostra n), então o paralelismo explorado é o de
// dentro de cada amostra: neurônios, entradas e pesos.
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
constexpr int threads_treino = 64;
static_assert(threads_treino >= entradas, "uma thread por entrada ao carregar a imagem");

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

// Neurônio i soma o viés aos produtos entre entradas e pesos, na ordem do
// legado. wt é transposto (wt[j * n_out + i]): threads vizinhas leem
// endereços vizinhos da memória compartilhada, sem conflito de banco.
template <int n_in, int n_out, bool final>
__device__ void forward(const float* input, const float* wt, const float* bias,
                        float* z, float* out) {
    const int i = threadIdx.x;
    if (i < n_out) {
        float sum = bias[i];
        for (int j = 0; j < n_in; ++j) {
            sum += input[j] * wt[j * n_out + i];
        }
        z[i] = sum;
        out[i] = final ? sum : (sum > 0.0f ? sum : 0.0f);
    }
}

// Gradiente da entrada j: soma, na ordem dos neurônios, delta vezes peso.
template <int n_in, int n_out>
__device__ void grad_in(const float* delta, const float* wt, float* grad) {
    const int j = threadIdx.x;
    if (j < n_in) {
        float sum = 0;
        for (int i = 0; i < n_out; ++i) {
            sum += delta[i] * wt[j * n_out + i];
        }
        grad[j] = sum;
    }
}

// SGD: as threads dividem os pesos da camada (índice k = j * n_out + i).
template <int n_in, int n_out>
__device__ void atualizar(const float* delta, const float* input, float* wt,
                          float* bias, float learning_rate) {
    for (int k = threadIdx.x; k < n_in * n_out; k += blockDim.x) {
        const int j = k / n_out;
        const int i = k % n_out;
        wt[k] -= learning_rate * (delta[i] * input[j]);
    }
    if (threadIdx.x < n_out) {
        bias[threadIdx.x] -= learning_rate * delta[threadIdx.x];
    }
}

// Delta da camada oculta: o gradiente só passa onde a soma foi positiva.
__device__ float delta_relu(float grad, float z) {
    return grad * (z > 0.0f ? 1.0f : 0.0f);
}

// Copia uma camada entre o layout do legado (global) e o transposto (shared).
template <int n_in, int n_out, bool para_shared>
__device__ void copiar_camada(float* global, float* wt) {
    for (int k = threadIdx.x; k < n_in * n_out; k += blockDim.x) {
        const int i = k / n_in;
        const int j = k % n_in;
        if (para_shared) {
            wt[j * n_out + i] = global[k];
        } else {
            global[k] = wt[j * n_out + i];
        }
    }
}

// Treino completo em um único bloco. perdas[e * n + a] recebe a perda da
// amostra a na época e; a média por época é feita no host, em double e na
// mesma ordem do legado.
__global__ void treinar_kernel(const float* imagens, const int* rotulos, int n,
                               float* parametros, float* perdas,
                               float learning_rate) {
    __shared__ float w1[entradas * n1], b1[n1];
    __shared__ float w2[n1 * n2], b2[n2];
    __shared__ float w3[n2 * classes], b3[classes];
    __shared__ float x[entradas];
    __shared__ float z1[n1], a1[n1], z2[n2], a2[n2], z3[classes], s[classes];
    __shared__ float grad[classes], g3[n2], d2[n2], g2[n1], d1[n1];

    copiar_camada<entradas, n1, true>(parametros + Parametros::w1, w1);
    copiar_camada<n1, n2, true>(parametros + Parametros::w2, w2);
    copiar_camada<n2, classes, true>(parametros + Parametros::w3, w3);
    const int t = threadIdx.x;
    if (t < n1) b1[t] = parametros[Parametros::b1 + t];
    if (t < n2) b2[t] = parametros[Parametros::b2 + t];
    if (t < classes) b3[t] = parametros[Parametros::b3 + t];
    __syncthreads();

    for (int epoca = 0; epoca < epocas; ++epoca) {
        for (int a = 0; a < n; ++a) {
            if (t < entradas) x[t] = imagens[a * entradas + t];
            __syncthreads();
            forward<entradas, n1, false>(x, w1, b1, z1, a1);
            __syncthreads();
            forward<n1, n2, false>(a1, w2, b2, z2, a2);
            __syncthreads();
            forward<n2, classes, true>(a2, w3, b3, z3, s);
            __syncthreads();
            // Perda e gradiente, sequenciais como em Barenn::perda.
            if (t == 0) {
                const int rotulo = rotulos[a];
                float total = 0;
                for (int i = 0; i < classes; ++i) {
                    const float erro = s[i] - (i == rotulo ? 1.0f : 0.0f);
                    total += 0.5f * erro * erro;
                    grad[i] = erro;
                }
                perdas[epoca * n + a] = total;
            }
            __syncthreads();
            // Última camada: delta é o próprio gradiente.
            grad_in<n2, classes>(grad, w3, g3);
            __syncthreads();
            atualizar<n2, classes>(grad, a2, w3, b3, learning_rate);
            if (t < n2) d2[t] = delta_relu(g3[t], z2[t]);
            __syncthreads();
            grad_in<n1, n2>(d2, w2, g2);
            __syncthreads();
            atualizar<n1, n2>(d2, a1, w2, b2, learning_rate);
            if (t < n1) d1[t] = delta_relu(g2[t], z1[t]);
            __syncthreads();
            atualizar<entradas, n1>(d1, x, w1, b1, learning_rate);
            __syncthreads();
        }
    }

    copiar_camada<entradas, n1, false>(parametros + Parametros::w1, w1);
    copiar_camada<n1, n2, false>(parametros + Parametros::w2, w2);
    copiar_camada<n2, classes, false>(parametros + Parametros::w3, w3);
    if (t < n1) parametros[Parametros::b1 + t] = b1[t];
    if (t < n2) parametros[Parametros::b2 + t] = b2[t];
    if (t < classes) parametros[Parametros::b3 + t] = b3[t];
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
        treinar_kernel<<<1, threads_treino>>>(imagens.get(), rotulos.get(), n,
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
