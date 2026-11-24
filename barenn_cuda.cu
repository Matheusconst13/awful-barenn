// Barenn na GPU: porting direto da classe DenseLayer para CUDA. Cada etapa
// de cada camada (forward, delta, gradiente da entrada, atualização) é um
// kernel; a perda continua no host, como no código original.
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

// Um thread por neurônio: soma o viés aos produtos entre entradas e pesos,
// na mesma ordem do legado (j = 0, 1, ..., n_in - 1).
__global__ void forward_kernel(const float* input, const float* weights,
                               const float* bias, float* z, float* out,
                               int n_in, int n_out, bool final) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) {
        return;
    }
    float sum = bias[i];
    for (int j = 0; j < n_in; ++j) {
        sum += input[j] * weights[i * n_in + j];
    }
    z[i] = sum;
    // As camadas ocultas usam ReLU. A última devolve escores lineares.
    out[i] = final ? sum : (sum > 0.0f ? sum : 0.0f);
}

// Delta é o gradiente da perda em relação à soma antes da ativação.
__global__ void delta_kernel(const float* grad_out, const float* z,
                             float* delta, int n_out, bool final) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        delta[i] = final ? grad_out[i]
                         : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
    }
}

// Um thread por entrada. Usa os pesos anteriores à atualização.
__global__ void grad_in_kernel(const float* delta, const float* weights,
                               float* grad_in, int n_in, int n_out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_in) {
        return;
    }
    float sum = 0;
    for (int i = 0; i < n_out; ++i) {
        sum += delta[i] * weights[i * n_in + j];
    }
    grad_in[j] = sum;
}

// Um bloco por neurônio e um thread por entrada (SGD por amostra).
__global__ void update_kernel(const float* delta, const float* input,
                              float* weights, float* bias, int n_in,
                              float learning_rate) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    weights[i * n_in + j] -= learning_rate * (delta[i] * input[j]);
    if (j == 0) {
        bias[i] -= learning_rate * delta[i];
    }
}

// Camada densa com todos os vetores na memória global da GPU.
class DenseLayerGPU {
    int in_features;
    int out_features;
    bool final;
    // weights[i * in_features + j] liga a entrada j ao neurônio i.
    float* weights = nullptr;
    float* bias = nullptr;
    float* last_z = nullptr;
    float* out = nullptr;
    float* delta = nullptr;
    float* grad_in = nullptr;
    const float* last_input = nullptr;

public:
    DenseLayerGPU(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final) {
        const vector<float> iniciais = pesos_iniciais(input_size, output_size);
        CUDA_CHECK(cudaMalloc(&weights, iniciais.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&bias, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&last_z, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&delta, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grad_in, in_features * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(weights, iniciais.data(),
                              iniciais.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(bias, 0, out_features * sizeof(float)));
    }

    DenseLayerGPU(const DenseLayerGPU&) = delete;
    DenseLayerGPU& operator=(const DenseLayerGPU&) = delete;

    ~DenseLayerGPU() {
        for (float* ponteiro : {weights, bias, last_z, out, delta, grad_in}) {
            cudaFree(ponteiro);
        }
    }

    const float* Forward(const float* input) {
        last_input = input;
        forward_kernel<<<1, out_features>>>(input, weights, bias, last_z, out,
                                            in_features, out_features, final);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    // Sem calcular_grad_in, pula o gradiente da entrada: a primeira camada
    // não tem camada anterior para recebê-lo.
    const float* Backward(const float* grad_out, float learning_rate,
                          bool calcular_grad_in = true) {
        delta_kernel<<<1, out_features>>>(grad_out, last_z, delta,
                                          out_features, final);
        CUDA_CHECK(cudaGetLastError());
        if (calcular_grad_in) {
            grad_in_kernel<<<1, in_features>>>(delta, weights, grad_in,
                                               in_features, out_features);
            CUDA_CHECK(cudaGetLastError());
        }
        update_kernel<<<out_features, in_features>>>(
            delta, last_input, weights, bias, in_features, learning_rate);
        CUDA_CHECK(cudaGetLastError());
        return grad_in;
    }

    // Acrescenta pesos e vieses (layout do legado) copiados da GPU.
    void Parametros(vector<float>& destino) const {
        const size_t inicio = destino.size();
        destino.resize(inicio + (in_features + 1) * out_features);
        CUDA_CHECK(cudaMemcpy(&destino[inicio], weights,
                              in_features * out_features * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&destino[inicio + in_features * out_features],
                              bias, out_features * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

// Rede densa com dimensões 64 -> 32 -> 16 -> 10.
struct RedeGPU {
    DenseLayerGPU l1{entradas, 32};
    DenseLayerGPU l2{32, 16};
    DenseLayerGPU l3{16, classes, true};

    const float* Forward(const float* imagem) {
        return l3.Forward(l2.Forward(l1.Forward(imagem)));
    }

    // Propaga o gradiente da última camada até a primeira.
    // Cada Backward também atualiza os parâmetros da sua camada.
    void Backward(const float* gradiente, float learning_rate) {
        l1.Backward(l2.Backward(l3.Backward(gradiente, learning_rate),
                                learning_rate),
                    learning_rate, false);
    }

    vector<float> Parametros() const {
        vector<float> parametros;
        l1.Parametros(parametros);
        l2.Parametros(parametros);
        l3.Parametros(parametros);
        return parametros;
    }
};

// Copia as imagens de um conjunto para a GPU.
float* enviar_imagens(const Conjunto& dados) {
    float* imagens;
    CUDA_CHECK(cudaMalloc(&imagens, dados.imagens.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(imagens, dados.imagens.data(),
                          dados.imagens.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return imagens;
}

} // namespace Barenn

int main(int argc, char** argv) {
    try {
        Barenn::Conjunto treino, teste;
        if (!Barenn::carregar(argc, argv, treino, teste)) {
            return 1;
        }

        // Cria o contexto CUDA fora do cronômetro (custo fixo de ~0,1 s).
        CUDA_CHECK(cudaFree(nullptr));
        Barenn::RedeGPU rede;
        float* gradiente_gpu;
        CUDA_CHECK(cudaMalloc(&gradiente_gpu, Barenn::classes * sizeof(float)));
        array<float, Barenn::classes> escores{};
        array<float, Barenn::classes> gradiente{};
        array<double, Barenn::epocas> perdas{};

        // A medição inclui o envio das imagens, forward, perda e backward de
        // todas as épocas e a cópia dos parâmetros finais de volta ao host.
        const auto inicio = chrono::steady_clock::now();
        float* imagens = Barenn::enviar_imagens(treino);
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t i = 0; i < treino.size(); ++i) {
                const float* escores_gpu =
                    rede.Forward(imagens + i * Barenn::entradas);
                // A perda é calculada no host: escores vão, gradiente volta.
                CUDA_CHECK(cudaMemcpy(escores.data(), escores_gpu,
                                      sizeof(escores), cudaMemcpyDeviceToHost));
                perda_total += Barenn::perda(escores.data(), treino.rotulos[i],
                                             gradiente.data());
                CUDA_CHECK(cudaMemcpy(gradiente_gpu, gradiente.data(),
                                      sizeof(gradiente), cudaMemcpyHostToDevice));
                rede.Backward(gradiente_gpu, Barenn::taxa);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const vector<float> parametros = rede.Parametros();
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
        CUDA_CHECK(cudaFree(imagens));
        CUDA_CHECK(cudaFree(gradiente_gpu));

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        Barenn::imprimir_treino(perdas, treino.size(), tempo.count());
        Barenn::gravar_parametros(parametros);
        float* imagens_teste = Barenn::enviar_imagens(teste);
        vector<float> todos_escores(teste.size() * Barenn::classes);
        for (size_t i = 0; i < teste.size(); ++i) {
            const float* escores_gpu =
                rede.Forward(imagens_teste + i * Barenn::entradas);
            CUDA_CHECK(cudaMemcpy(&todos_escores[i * Barenn::classes],
                                  escores_gpu, Barenn::classes * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(imagens_teste));
        Barenn::relatorio(teste, todos_escores.data(), Barenn::classes);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
