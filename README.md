# Implemetação de classificadores

O repositório contém os arquivos fonte referentes à implemetação das arquiteturas de rede neural ResNet-50 e Vision transformers para a classificação de imagens. No contexto do meu projeto de mestrado, o objetivo é classificar a imagem facial de roedores quanto à presença e intensidade dor. A implementação das arquiteturas é feita de de forma acessível e intuitiva, utilizando uma interface gráfica via PyQt5 que facilita a exploração e aplicação dessa tecnologia em diferentes cenários.

A ResNet50 (Residual Network com 50 camadas) é uma das arquiteturas mais populares e influentes no campo de visão computacional. Sua principal inovação é o uso de conexões residuais (skip connections), que permitem que os gradientes fluam com mais facilidade durante o treinamento de redes muito profundas. Esse mecanismo resolve um problema comum em arquiteturas anteriores: a degradação do desempenho à medida que mais camadas eram adicionadas.

O Vision Transformer representa uma mudança de paradigma em visão computacional, pois adapta os mecanismos de atenção originalmente desenvolvidos para processamento de linguagem natural (os Transformers) ao domínio de imagens. Em vez de processar uma imagem por meio de convoluções, o ViT a divide em pequenos blocos (patches), que são tratados como "palavras visuais". Esses blocos são então passados por camadas de autoatenção, que permitem ao modelo aprender relações globais entre diferentes regiões da imagem desde as primeiras etapas do processamento.


## Dados pessoais
**Nome:** Marcio Salmazo Ramos \
**Redes sociais e contato:**

| [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marcio-ramos-b94669235) | [![Instagram](https://img.shields.io/badge/-Instagram-%23E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/marcio.salmazo) | [![Gmail](https://img.shields.io/badge/Gmail-333333?style=for-the-badge&logo=gmail&logoColor=red)](mailto:contato.marcio.salmazo19@gmail.com) | [![GitHub](https://img.shields.io/badge/GitHub-0077B5?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Marcio-Salmazo) |
|---|---|---|---|

## Objetivos da atividade

Este trabalho tem como principal objetivo o desenvolvimento de uma ferramenta automatizada, voltada para a detecção e classificação de padrões faciais que expressam a
presença de dor em animais. A proposta se baseia na Grimmace Scale associada a técnicas de visão computacional e aprendizado de máquina, visando superar as limitações dos
métodos convencionais, que ainda dependem predominantemente da observação humana
manual. A partir dessa premissa, o desenvolvimento desta aplicação se desdobra nos seguintes objetivos específicos:

* Preparar dados para treinamento (Separando os grupos de treinamento e teste);
* Construir, compilar e treinar o modelo referente às arquiteturas citadas (aplicando os parâmetros necessários para seu funcionamento);
* Retornar resultados via logs para o Tensorboard, bem como armazenar o arquivo .w5 contendo os pesos aprendidos para análises de desempenho;

## Interface e Funcionalidades
### 📂 Janela Inicial

A janela inicial do programa é divida em 2 setores: uma área à esquerda dedicada à exibição das mensagens de log (informando sobre o status da operação) e uma área à direita dedicada às funcionalidades do sistema. Em um primeiro momento a área de funcionalidades exige ao usuário a escolha da arquitetura que será utilizada (ViT ou ResNet), por meio de *radiobuttons*. No momento em que o usuário confirma a seleção, são apresentados as seguintes funcionalidades (referentes à arquitetura selecionada):


  - **Selecionar dataset** – Permite selecionar a pasta contendo a base de dados para o treinamento. Importante salientar que o diretório escolhido deve conter subpastas (cada uma representando as diferentes classes). Essa função exige a definição do tamanho de entrada, tamanho dos lotes (batch) e porcentagem de divisão para os dados de validação;
  - **Construir ResNet50** – Constrói e compila a arquiteura da rede, definindo os parâmetros como formato de entrada, modelo base, camadas da rede, funções de ativação, otimizadores, função custo, dentre outros;  
   - **Construir Modelo ViT** – Constrói e compila a arquiteura da rede. Essa função exige a definição de parâmetros específicos à ViT, os quais são detalhados na seção 'Parâmetros exigidos pelo programa' deste mesmo documento  
  - **Iniciar treinamento** – Inicia o treinamento da rede. Para ter início, exige a definição de um nome para o arquivo de log e a quantidade de épocas para o treinamento;  
  - **Abrir TensorBoard** – Inicia o Tensorboard e abre uma página na web para exibição dos arquivos de log. Esta função exige a escolha do diretório que contém os logs (geralmente está em logs/fit na pasta raiz do executável);  
  - **Fechar programa** – encerra a aplicação. 


## Parâmetros exigidos pela ResNet-50
- **Input size** - Tamanho que as imagens devem ser redimensionadas para servir como entrada da rede. O valor inserido definira a altura e largura da imagem;
- **Batch size** - Refere-se ao número de amostras de dados que um modelo de aprendizado de máquina processa em uma única iteração;
- **Split (treino/validação)** - Define a porcentagem de dados destinados para treino e validação. Exemplo: 0.2 -> 20% para validação e 80% para treino;
- **Nome para logs** - Permite a definição do nome do arquivo de logs gerado após o treinamento;
- **Épocas** - Permite definir a quantidade de épocas de treinamento.

## Parâmetros exigidos pela ViT
- **Input size** - Análogo ao requisito exigido pela ResNet50;
- **Batch size** - Análogo ao requisito exigido pela ResNet50;
- **Split (treino/validação)** - ao requisito exigido pela ResNet50;
- **Épocas** - Análogo ao requisito exigido pela ResNet50;

- **Patch size** - O tamanho dos blocos (patches) em que a imagem será dividida. Quanto menor o patch, mais detalhes o modelo enxerga desde o início, mas também aumenta a quantidade de patches a processar (mais custo computacional);
- **Projection Dim** - A dimensão do vetor em que cada patch será representado após a projeção linear (Dimensões maiores permitem mais capacidade de representação, mas também exigem mais memória e poder de processamento);
- **Transform Layers** - Número de blocos de transformers (compostos por atenção + MLP) empilhados no modelo. Quanto mais camadas, mais refinada e abstrata fica a representação;
- **Attention Heads** - Cada camada de atenção pode ter várias "cabeças", que aprendem a focar em diferentes aspectos da imagem ao mesmo tempo;
- **MLP Units** - Número de neurônios nas camadas densas (feed-forward layers) que seguem a parte de atenção em cada bloco do transformador. Normalmente é um valor maior que o 'projection dim'.
- **Nome para logs** - Análogo ao requisito exigido pela ResNet50;

> 🔎 **Observações Importantes**  
> - O valor de 'Split' deve estar em notação de ponto flutuante, estritamente entre 0.0 e 1.0;
> - O aplicativo indica valores 'padrões' caso o usuário não saiba ao certo o valor de alguns parâmetros;
> - O diretório escolhido para o dataset deve conter subpastas (cada uma representando as diferentes classes);  
> - Seguir as versões dos pré-requisitos à risca, uma vez que versões mais novas podem gerar conflitos na IDE.
> - 

## ⚙️ Pré-requisitos e Instalação

- Sistema Operacional: **Windows**;  
- Python **3.9** (recomendado);  
- Tensorflow **2.10.0**;
- Numpy **1.23.5**;
- Scipy **1.13.1**;
- Protobuf **3.20.2**;
- Tensorboard **2.10.1**;
- Pillow (Sem versão específica, pode ser a mais atual);
- CUDA 11.2 (Para uso da GPU);
- CuDNN 8.1 (Para uso da GPU).

---

## Tutorial para instalar GPU para TensorFlow no Windows

1. **Desinstalar pacotes conflitantes (opcional, mas recomendado):** utilizar o comando *pip uninstall tensorflow tensorflow-gpu tensorflow-intel* ou desinstalar manualmente via explorador do windows;
2. **Instalar TensorFlow GPU 2.10:** versão da biblioteca específica para esta implementação;
3. **Baixar & instalar CUDA 11.2:** Download oficial pelo site da NVIDIA *https://developer.nvidia.com/cuda-11.2.0-download-archive*;
4. **Baixar cuDNN 8.1 para CUDA 11.2:** Requer login NVIDIA Developer (gratuito) e pode ser acessado pelo link *https://developer.nvidia.com/rdp/cudnn-archive#a-collapse811-110*. O resultado do download será um arquivo .ZIP;
5. **Copiar conteúdo das pastas baixadas (cuDNN 8.1):** É necessário extrair o conteúdo compactado e mover o conteúdo das pastas, seguindo o seguinte esquema:

conteúdo da pasta **bin**  → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\
conteúdo da pasta **lib**  → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64\
conteúdo da pasta **include** → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include

6. **Adicionar ao PATH:** Adicionar as seguintes entradas ao PATH do Windows e reiniciar o computador:

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

7. **Testar o download:** Abrir o console python e inserir os seguintes comandos:

import tensorflow as tf\
print(tf.config.list_physical_devices('GPU'))

Se o resultado for algo como *[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]*, indica que o processo foi realizado com sucesso.

---

## Estrutura de base de dados padrão para utilização

Para que o programa reconheça devidamente a base de dados, é necessário seguir a seguinte estrutura:

├── Diretório contendo a base de dados (Arquivos .png)/\
├── Classe 1 (Diretótio)/\
│   └── Conjunto de imagens (Arquivos .png)/\
├── Classe 2 (Diretótio)/\
│   └── Conjunto de imagens (Arquivos .png)/\
│   ...\
├── Classe N (Diretótio)/\
│   └── Conjunto de imagens (Arquivos .png)/\


---

## ▶️ Modo de Uso

1. Abrir o diretório em uma IDE python de sua preferência e criar um novo ambiente virtual (recomendo o PyCharm Community Edition);
2. Instalar os pacotes requeridos pela aplicação (Ver seção de pré-requisitos);
3. Executar o arquivo main.py;
4. Carregar um diretório contendo uma base de dados e definir os parâmetros exigidos. Levar em consideração a estrutura de arquivos exigida (Ver seção anterior); 
5. Construir a estrutura da arquitetura
6. Iniciar o treinamento, definindo os parâmetros exigidos
7. Aguardar até o encerramento do treino para obter o arquivo de pesos e logs

---

## Bugs conhecidos

* Ao finalizar o treinamento utilizando a arquitetura, o programa trava e se encerra sozinho, acredito que seja algo relacionado ao *callback*, contudo não consegui corrigir. É importante destacar que mesmo com esse erro, os pesos e logs são registrados normalmente.

---

## ⚠️ Erros Comuns

| Erro | Causa provável | Solução |
|------|----------------|---------|
| ❌ Erro ao abrir base de dados | Estrutura de arquivos inválida | Verifique se o diretório contém as subpastas como categorias |
| ❌ Aplicativo não abre | Python ou dependências ausentes | Reinstale dependências |
| ❌ Travamento ou fechamento inesperado | Instabilidade de código | Contatar desenvolvedor |

---

## 🆕 Atualizações / Changelog

- **v0.5.0**
  - Versão inicial, contemplando a unificação de ambas as arquiteturas em um único programa
  
---

## 👨‍💻 Autores / Contribuidores

- Maurício Cunha Escarpinati (Orientador - UFU) 
- Daniel Duarte Abdala (Co-orientador - UFU)  

