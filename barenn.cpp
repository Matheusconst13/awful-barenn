#include <algorithm>
#include <charconv>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

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
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // A ordem de sorteio é a do legado: neurônio a neurônio.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < OUT; ++i) {
            for (int j = 0; j < IN; ++j) {
                peso(i, j) = distribution(gen);
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

    // Grava pesos (neurônio a neurônio, como no legado) e vieses em binário
    // para validar a corretude bit a bit entre versões.
    void Salvar(ostream& saida) const {
        for (int i = 0; i < OUT; ++i) {
            for (int j = 0; j < IN; ++j) {
                const float valor = weights[j][i / largura][i % largura];
                saida.write(reinterpret_cast<const char*>(&valor), sizeof(valor));
            }
        }
        for (int i = 0; i < OUT; ++i) {
            const float valor = bias[i / largura][i % largura];
            saida.write(reinterpret_cast<const char*>(&valor), sizeof(valor));
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

    void Salvar(ostream& saida) const {
        l1.Salvar(saida);
        l2.Salvar(saida);
        l3.Salvar(saida);
    }
};

// Imagens em um único bloco contíguo (amostra i em imagens[i * entradas]),
// percorrido sequencialmente a cada época.
struct Conjunto {
    vector<float> imagens;
    vector<int> rotulos;

    size_t size() const { return rotulos.size(); }
    const float* imagem(size_t i) const { return &imagens[i * entradas]; }
};

// Pula espaços em branco, como o operator>> dos streams.
const char* pular_espacos(const char* p, const char* fim) {
    while (p < fim && isspace(static_cast<unsigned char>(*p))) {
        ++p;
    }
    return p;
}

// Lê um número como o operator>> (espaços antes e sinal '+' opcionais) com
// std::from_chars, que não usa locale nem aloca memória.
template <typename T>
const char* ler_numero(const char* p, const char* fim, T& valor) {
    p = pular_espacos(p, fim);
    if (p < fim && *p == '+') {
        ++p;
    }
    const auto [resto, erro] = from_chars(p, fim, valor);
    return erro == errc{} ? resto : nullptr;
}

// Valida e converte uma linha [p, fim): 64 atributos de 0 a 16, cada um
// seguido por vírgula, e um rótulo inteiro de 0 a 9. Devolve false se a
// linha é inválida.
bool ler_linha(const char* p, const char* fim, float* imagem, int& rotulo) {
    // Cada atributo deve ser seguido por vírgula, inclusive o último.
    for (int i = 0; i < entradas; ++i) {
        float valor;
        if (!(p = ler_numero(p, fim, valor))) {
            return false;
        }
        p = pular_espacos(p, fim);
        if (p == fim || *p++ != ',' ||
            !isfinite(valor) || valor < 0 || valor > 16) {
            return false;
        }
        // Converte os valores de [0, 16] para [0, 1].
        imagem[i] = valor / 16.0f;
    }
    // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
    return (p = ler_numero(p, fim, rotulo)) && rotulo >= 0 &&
           rotulo < classes && pular_espacos(p, fim) == fim;
}

// Lê o arquivo inteiro de uma vez e converte as linhas em paralelo: cada
// linha é independente e escreve na sua própria posição do Conjunto.
Conjunto ler_dados(const string& caminho) {
    ifstream arquivo(caminho, ios::binary);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }
    const string conteudo{istreambuf_iterator<char>(arquivo),
                          istreambuf_iterator<char>()};
    if (arquivo.bad() || conteudo.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }

    // Separa as linhas como o getline: por '\n', sem linha extra no final.
    vector<size_t> inicios;
    for (size_t pos = 0; pos < conteudo.size();) {
        inicios.push_back(pos);
        const size_t fim = conteudo.find('\n', pos);
        pos = fim == string::npos ? conteudo.size() : fim + 1;
    }
    inicios.push_back(conteudo.size() + 1);

    const long linhas = static_cast<long>(inicios.size()) - 1;
    Conjunto dados;
    dados.imagens.resize(linhas * entradas);
    dados.rotulos.resize(linhas);
    // Como no legado, o erro aponta a primeira linha inválida do arquivo.
    long primeira_invalida = linhas;
#pragma omp parallel for schedule(static) reduction(min : primeira_invalida)
    for (long l = 0; l < linhas; ++l) {
        const char* linha = conteudo.data() + inicios[l];
        const char* fim = conteudo.data() + inicios[l + 1] - 1;
        if (!ler_linha(linha, fim, &dados.imagens[l * entradas], dados.rotulos[l])) {
            primeira_invalida = min(primeira_invalida, l);
        }
    }
    if (primeira_invalida < linhas) {
        throw runtime_error(caminho + ":" + to_string(primeira_invalida + 1) +
            " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

void avaliar(Rede& rede, const Conjunto& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const int rotulo = teste.rotulos[i];
        const float* escores = rede.Forward(teste.imagem(i));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores, escores + classes) - escores);
        bool acertou = previsto == rotulo;
        acertos += acertou;
        ++confusoes[rotulo][previsto];
        perda_total += perda(escores, rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }
        // A leitura é a única região paralela. Sem isto, as threads ociosas
        // do OpenMP ficam em espera ativa (spin) durante o treino serial,
        // ocupando núcleos e reduzindo o turbo do núcleo que treina.
        omp_pause_resource_all(omp_pause_soft);

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        Barenn::Rede rede;
        // Escores e gradiente têm as posições extras da última camada (zeradas).
        alignas(32) array<float, decltype(rede.l3)::saidas> gradiente{};
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t i = 0; i < treino.size(); ++i) {
                const float* escores = rede.Forward(treino.imagem(i));
                perda_total += Barenn::perda(escores, treino.rotulos[i],
                                             gradiente.data());
                rede.Backward(gradiente.data(), taxa);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";

        // Com BARENN_PESOS definido, grava os parâmetros finais para validação.
        if (const char* caminho = getenv("BARENN_PESOS")) {
            ofstream pesos(caminho, ios::binary);
            rede.Salvar(pesos);
        }
        Barenn::avaliar(rede, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
