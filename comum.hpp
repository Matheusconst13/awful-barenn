// Partes do Barenn compartilhadas entre a versão CPU (barenn.cpp) e a versão
// GPU (barenn_cuda.cu): leitura dos dados, perda, relatórios e parâmetros.
#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;
constexpr int epocas = 51;
constexpr float taxa = 0.02f;

// Sorteia os pesos iniciais de uma camada no layout do legado
// (pesos[i * in + j] liga a entrada j ao neurônio i). A semente fixa torna a
// inicialização de cada camada reproduzível.
inline vector<float> pesos_iniciais(int in, int out) {
    mt19937 gen(1337);
    uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    vector<float> pesos(static_cast<size_t>(in) * out);
    for (float& peso : pesos) {
        peso = distribution(gen);
    }
    return pesos;
}

// Imagens em um único bloco contíguo (amostra i em imagens[i * entradas]),
// percorrido sequencialmente a cada época.
struct Conjunto {
    vector<float> imagens;
    vector<int> rotulos;

    size_t size() const { return rotulos.size(); }
    const float* imagem(size_t i) const { return &imagens[i * entradas]; }
};

// Pula espaços em branco, como o operator>> dos streams.
inline const char* pular_espacos(const char* p, const char* fim) {
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
inline bool ler_linha(const char* p, const char* fim, float* imagem, int& rotulo) {
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
inline Conjunto ler_dados(const string& caminho) {
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

// Lê os arquivos de treino e de teste (padrão: optdigits.tra e .tes).
// Devolve false, após imprimir o uso, se os argumentos são inválidos.
inline bool carregar(int argc, char** argv, Conjunto& treino, Conjunto& teste) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return false;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";
    treino = ler_dados(arquivo_treino);
    teste = ler_dados(arquivo_teste);
    if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
        throw runtime_error("Use arquivos distintos para treino e teste.");
    }
    // A leitura é a única região paralela. Sem isto, as threads ociosas do
    // OpenMP ficam em espera ativa (spin) durante o treino serial, ocupando
    // núcleos e reduzindo o turbo do núcleo que treina.
    omp_pause_resource_all(omp_pause_soft);
    return true;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
inline float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
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

// Imprime a perda média de uma a cada cinco épocas e o tempo de treino.
inline void imprimir_treino(const array<double, epocas>& perdas, size_t imagens,
                            double tempo) {
    cout << fixed << setprecision(4)
         << "TREINO | " << imagens << " imagens\n";
    for (int epoca = 0; epoca < epocas; epoca += 5) {
        cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
             << " | perda media " << perdas[epoca] << "\n";
    }
    cout << "Tempo de treino " << tempo << " s\n";
}

// Com BARENN_PESOS definido, grava os parâmetros finais (pesos de cada camada
// no layout do legado, seguidos dos vieses) para validação bit a bit.
inline void gravar_parametros(const vector<float>& parametros) {
    if (const char* caminho = getenv("BARENN_PESOS")) {
        ofstream saida(caminho, ios::binary);
        saida.write(reinterpret_cast<const char*>(parametros.data()),
                    parametros.size() * sizeof(float));
    }
}

// Resume a avaliação no conjunto de teste a partir dos escores já
// calculados: escores + i * passo aponta os escores da amostra i.
inline void relatorio(const Conjunto& teste, const float* todos_escores,
                      size_t passo) {
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

    for (size_t i = 0; i < teste.size(); ++i) {
        const int rotulo = teste.rotulos[i];
        const float* escores = todos_escores + i * passo;
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
