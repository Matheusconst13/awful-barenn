#include <chrono>
#include <cstring>

#include "comum.hpp"

namespace Barenn {

// Vetor de 8 floats (256 bits, um registrador AVX). As operações entre
// vetores são elemento a elemento e arredondam como as escalares (IEEE 754).
typedef float v8f __attribute__((vector_size(32)));
constexpr int largura = 8;

// Camada densa com dimensões conhecidas em tempo de compilação: os laços têm
// contagem fixa e o compilador pode desenrolá-los e manter os acumuladores
// em registradores.
template <int IN, int OUT, bool FINAL = false>
class DenseLayer {
public:
    // Neurônios agrupados em blocos de 8. Se OUT não for múltiplo de 8, os
    // neurônios extras têm pesos e viés zero e seus valores nunca são lidos.
    static constexpr int blocos = (OUT + largura - 1) / largura;
    static constexpr int saidas = blocos * largura;

private:
    // weights[j][b] guarda os pesos da entrada j para os neurônios
    // 8b, ..., 8b + 7. No layout transposto os pesos de uma mesma entrada
    // ficam contíguos para todos os neurônios, então o forward vetoriza entre
    // neurônios sem mudar a ordem em que cada neurônio acumula sua soma.
    v8f weights[IN][blocos] = {};
    v8f bias[blocos] = {};
    // Cópia dos pesos no layout do legado (neurônio i, entrada j), usada só
    // pelo gradiente da entrada, que soma sobre os neurônios: assim ele
    // vetoriza entre entradas. As duas cópias recebem as mesmas operações e
    // permanecem idênticas.
    static constexpr int blocos_in = (IN + largura - 1) / largura;
    static_assert(IN % largura == 0, "IN deve ser múltiplo de 8");
    v8f weights_lin[OUT][blocos_in] = {};
    // Dados do último forward usados para calcular os gradientes. A entrada
    // não é copiada: ela pertence à amostra ou à camada anterior e continua
    // válida até o Backward da mesma amostra.
    const float* last_input = nullptr;
    v8f last_z[blocos];
    v8f out[blocos];
    v8f delta[blocos];
    alignas(32) float grad_in[IN];

    float& peso(int i, int j) { return weights[j][i / largura][i % largura]; }

public:
    DenseLayer() {
        const vector<float> iniciais = pesos_iniciais(IN, OUT);
        for (int i = 0; i < OUT; ++i) {
            for (int j = 0; j < IN; ++j) {
                peso(i, j) = iniciais[i * IN + j];
                weights_lin[i][j / largura][j % largura] = peso(i, j);
            }
        }
    }

    // Devolve as saídas da camada (saidas floats, as OUT primeiras válidas).
    const float* Forward(const float* input) {
        last_input = input;
        // Cada neurônio soma o viés aos produtos entre entradas e pesos, na
        // ordem j = 0, 1, ..., IN - 1, como no legado.
        v8f sum[blocos];
        for (int b = 0; b < blocos; ++b) {
            sum[b] = bias[b];
        }
        for (int j = 0; j < IN; ++j) {
            const float entrada = input[j];
            for (int b = 0; b < blocos; ++b) {
                sum[b] += entrada * weights[j][b];
            }
        }
        for (int b = 0; b < blocos; ++b) {
            last_z[b] = sum[b];
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out[b] = FINAL ? sum[b] : (sum[b] > 0.0f ? sum[b] : v8f{});
        }
        return reinterpret_cast<const float*>(out);
    }

    // grad_out deve ter saidas floats, com zero nas posições extras.
    // Com GRAD_IN = false, pula o gradiente da entrada: a primeira camada
    // não tem camada anterior para recebê-lo.
    template <bool GRAD_IN = true>
    const float* Backward(const float* grad_out, float learning_rate) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int b = 0; b < blocos; ++b) {
            v8f g;
            memcpy(&g, grad_out + b * largura, sizeof(g));
            delta[b] = FINAL ? g
                             : g * (last_z[b] > 0.0f ? v8f{} + 1.0f : v8f{});
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        if constexpr (GRAD_IN) {
            v8f sum[blocos_in] = {};
            for (int i = 0; i < OUT; ++i) {
                const float d = delta[i / largura][i % largura];
                for (int b = 0; b < blocos_in; ++b) {
                    sum[b] += d * weights_lin[i][b];
                }
            }
            memcpy(grad_in, sum, sizeof(grad_in));
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        for (int j = 0; j < IN; ++j) {
            const float entrada = last_input[j];
            for (int b = 0; b < blocos; ++b) {
                weights[j][b] -= learning_rate * (delta[b] * entrada);
            }
        }
        if constexpr (GRAD_IN) {
            v8f entrada[blocos_in];
            memcpy(entrada, last_input, sizeof(entrada));
            for (int i = 0; i < OUT; ++i) {
                const float d = delta[i / largura][i % largura];
                for (int b = 0; b < blocos_in; ++b) {
                    weights_lin[i][b] -= learning_rate * (d * entrada[b]);
                }
            }
        }
        for (int b = 0; b < blocos; ++b) {
            bias[b] -= learning_rate * delta[b];
        }
        return grad_in;
    }

    // Acrescenta pesos (neurônio a neurônio, como no legado) e vieses.
    void Parametros(vector<float>& destino) const {
        for (int i = 0; i < OUT; ++i) {
            for (int j = 0; j < IN; ++j) {
                destino.push_back(weights[j][i / largura][i % largura]);
            }
        }
        for (int i = 0; i < OUT; ++i) {
            destino.push_back(bias[i / largura][i % largura]);
        }
    }
};

// Rede densa com dimensões 64 -> 32 -> 16 -> 10.
struct Rede {
    DenseLayer<entradas, 32> l1;
    DenseLayer<32, 16> l2;
    DenseLayer<16, classes, true> l3;

    const float* Forward(const float* imagem) {
        return l3.Forward(l2.Forward(l1.Forward(imagem)));
    }

    // Propaga o gradiente da última camada até a primeira.
    // Cada Backward também atualiza os parâmetros da sua camada.
    void Backward(const float* gradiente, float taxa) {
        l1.Backward<false>(l2.Backward(l3.Backward(gradiente, taxa), taxa), taxa);
    }

    vector<float> Parametros() const {
        vector<float> parametros;
        l1.Parametros(parametros);
        l2.Parametros(parametros);
        l3.Parametros(parametros);
        return parametros;
    }
};

// Escores de todas as imagens de teste (saidas floats por imagem). Só
// fazemos forward aqui. Os pesos ficam fixos.
vector<float> inferir(Rede& rede, const Conjunto& teste) {
    constexpr int passo = decltype(rede.l3)::saidas;
    vector<float> escores(teste.size() * passo);
    for (size_t i = 0; i < teste.size(); ++i) {
        const float* saida = rede.Forward(teste.imagem(i));
        copy(saida, saida + passo, &escores[i * passo]);
    }
    return escores;
}

} // namespace Barenn

int main(int argc, char** argv) {
    try {
        Barenn::Conjunto treino, teste;
        if (!Barenn::carregar(argc, argv, treino, teste)) {
            return 1;
        }

        Barenn::Rede rede;
        // Escores e gradiente têm as posições extras da última camada (zeradas).
        alignas(32) array<float, decltype(rede.l3)::saidas> gradiente{};
        array<double, Barenn::epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t i = 0; i < treino.size(); ++i) {
                const float* escores = rede.Forward(treino.imagem(i));
                perda_total += Barenn::perda(escores, treino.rotulos[i],
                                             gradiente.data());
                rede.Backward(gradiente.data(), Barenn::taxa);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        Barenn::imprimir_treino(perdas, treino.size(), tempo.count());
        Barenn::gravar_parametros(rede.Parametros());
        const auto escores = Barenn::inferir(rede, teste);
        Barenn::relatorio(teste, escores.data(), decltype(rede.l3)::saidas);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
