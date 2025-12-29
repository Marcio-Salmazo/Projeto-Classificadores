# Implemetação de classificadores

O repositório contém os arquivos fonte referentes à implemetação das arquiteturas de rede neural ResNet-50 e 
Vision transformers para a classificação de imagens. No contexto do meu projeto de mestrado, o objetivo é classificar 
a imagem facial de roedores quanto à presença e intensidade dor. A implementação das arquiteturas é feita de de forma 
acessível e intuitiva, utilizando uma interface gráfica via PyQt5 que facilita a exploração e aplicação dessa 
tecnologia em diferentes cenários.

A ResNet50 (Residual Network com 50 camadas) é uma das arquiteturas mais populares e influentes no campo de visão 
computacional. Sua principal inovação é o uso de conexões residuais (skip connections), que permitem que os gradientes 
fluam com mais facilidade durante o treinamento de redes muito profundas. Esse mecanismo resolve um problema comum 
em arquiteturas anteriores: a degradação do desempenho à medida que mais camadas eram adicionadas.

O Vision Transformer representa uma mudança de paradigma em visão computacional, pois adapta os mecanismos de atenção 
originalmente desenvolvidos para processamento de linguagem natural (os Transformers) ao domínio de imagens. 
Em vez de processar uma imagem por meio de convoluções, o ViT a divide em pequenos blocos (patches), que são tratados 
como "palavras visuais". Esses blocos são então passados por camadas de autoatenção, que permitem ao modelo aprender 
relações globais entre diferentes regiões da imagem desde as primeiras etapas do processamento.

## Dados pessoais
**Nome:** Marcio Salmazo Ramos \
**Redes sociais e contato:**

| [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marcio-ramos-b94669235) | [![Instagram](https://img.shields.io/badge/-Instagram-%23E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/marcio.salmazo) | [![Gmail](https://img.shields.io/badge/Gmail-333333?style=for-the-badge&logo=gmail&logoColor=red)](mailto:contato.marcio.salmazo19@gmail.com) | [![GitHub](https://img.shields.io/badge/GitHub-0077B5?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Marcio-Salmazo) |
|---|---|---|---|

## Objetivos da atividade

Este trabalho tem como principal objetivo o desenvolvimento de uma ferramenta automatizada, voltada para a detecção e 
classificação de padrões faciais que expressam a presença de dor em animais. A proposta se baseia na Grimmace Scale 
associada a técnicas de visão computacional e aprendizado de máquina, visando superar as limitações dos métodos 
convencionais, que ainda dependem predominantemente da observação humana manual. 
A partir dessa premissa, o desenvolvimento desta aplicação se desdobra nos seguintes objetivos específicos:

* Preparar dados para treinamento (Separando os grupos de treinamento e teste);
* Construir, compilar e treinar o modelo referente às arquiteturas citadas 
  (aplicando os parâmetros necessários para seu funcionamento);
* Retornar resultados via logs, bem como armazenar o arquivo contendo os pesos aprendidos para análises de desempenho;

## 📂 Estrutura de diretórios do projeto

O repositório está organizado de acordo com a seguinte estrutura:

    ├── Documentation/
    ├     └── (...)
    ├── Main_Project/
    ├     └── (...)
    ├── Main_Validation/
    ├     └── (...) 
    ├── Old Versions Backups/
    ├     └── (...) 
    ├── Requirements/
    ├     └── (...) 
    ├── .gitignore
    ├
    └──README.md

### Diretórios mais relevantes:

1. **Main_Project** &rarr; Contém os arquivos referentes à implementação do projeto principal
   voltado para a classificação das imagens de camundongos. Além de agregar os diretórios 
   responsáveis por armazenar logs e pesos de treinamento;


2. **Main_Validation** &rarr; Contém os arquivos referentes à implementação do projeto 
   voltado para a replicação de estudos envolvendo a ResNet e a ViT. A ideia é construir a 
   arquitetura fiel à um estudo científico consolidado para garantir a confiabilidade de seu
   funcionamento;


3. **Old Versions Backups** &rarr; Contém os arquivos referentes à implementação antiga do projeto,
   a qual contava com uma interface gráfica construída por meio da biblioteca PyQt5. A ideia central
   era criar um ambiente mais amigável ao usuário, contudo, incompatibilidades entre bibliotecas e
   a má integração entre os módulos acabaram por consumir bastante tempo do cronograma do projeto, o que
   levou à decisão de descartar essa implementação em prol de algo mais simples e objetivo, permitindo
   dar foco ao real objetivo do projeto.


4. **Documentation** &rarr; Contém os arquivos de documentação do projeto, o que inclui os artigos científicos 
utilizados como referência para a validação das arquiteturas de rede, bem como os arquivos .docx referentes
à documentação dos experimentos conduzidos, reunindo em um único local todo o material necessário para consulta e 
reprodutibilidade dos testes. 

   * **Experimentos conduzidos.docx:** Documentação dos exeprimentos conduzidos, por ordem de realização
   * **Protocolo para garantia de confiabilidade.docx:** Definição do protocolo de validação, em conformidade com os artigo

## 📂 Estrutura da base de dados para utilização do projeto principal

Para que o código referente ao projeto principal reconheça devidamente a base de dados, é necessário seguir a 
seguinte estrutura:

    ├── Diretório contendo a base de dados (Arquivos .png)/\
    │    │
    │    ├── Classe 1 (Diretótio)/
    │    │   └── Conjunto de imagens (Arquivos .png)/
    │    ├── Classe 2 (Diretótio)/
    │    │   └── Conjunto de imagens (Arquivos .png)/
    │    │   ...
    │    ├── Classe N (Diretótio)/
    │    │   └── Conjunto de imagens (Arquivos .png)/

## ✔️ Testes de confiabilidade

Com o intuito de garantir que as implementações das arquiteturas ResNet-50 e Vision Transformer (ViT) 
estejam corretas e produzam resultados condizentes com aqueles descritos na literatura científica, 
foi feita a replicação de dois estudos  consolidados na área. 
A obtenção de resultados equivalentes aos reportados nos trabalhos permite validar o comportamento 
das redes, garantindo que suas funcionalidades estejam corretamente implementadas. 
A partir dessa base confiável, torna-se possível introduzir modificações pontuais nas arquiteturas, 
para adaptá-las aos objetivos específicos deste projeto.

**Estrutura atual do diretório Main_Validadtion:**


    ├── RESNET/
    │    └── (Arquivos.py referentes à construção e treinamento da rede ResNet50)
    │
    ├── VISION TRANSFORMER/
    │    ├── Checkpoints/
    │    │      └── (Diretório de logs referente ao treinamento, contendo checkpoints, logs por época e relatório)
    │    └── (Arquivos.py referentes à construção e treinamento da rede ViT)
    │
    ├── Create_TFRecords.py
    │
    └── Process_ImageNet.py

* **OBSERVAÇÃO:** Os arquivos *Create_TFRecords.py* e *Process_ImageNet.py* não são exclusivos para o processo de 
validação das arquiteturas, são scripts auxiliares para tratar a base de dados da ImageNet no formato de TFRecords 


* **OBSERVAÇÃO:** O TFRecords é um formato de arquivo binário simples e eficiente, projetado especificamente pela Google para armazenar 
sequências de dados no TensorFlow Permite o pré-carregamento e o streaming eficiente de grandes volumes de dados, 
o que acelera o treinamento de modelos de aprendizado de máquina, especialmente em pipelines de dados complexos ou 
distribuídos. Adicionalmente, pode armazenar uma variedade de tipos de dados (inteiros, floats, strings, imagens) 
e estruturas de dados complexas, serializando-os em um formato de buffer de protocolo 

### 1. Validação para a ViT (Vision Transformer):

* ***Artigo de referência utilizado:*** Dosovitskiy et al., 2021 – “An Image is Worth 16x16 words: Transformers for Image Recognition at Scale”
* ***Dataset utilizado:*** ImageNet (ILSVRC2012)
* ***Acesso:*** https://arxiv.org/abs/2010.11929

#### Definições exigidas em código:
Uma vez que a validação não faz parte do escopo do projeto (servindo apenas como um teste preliminar para grantir a 
confiabilidade das implementações), não foi feita uma interface para definir caminhos e flags. Elas 
devem sere especificadas diretamente no código.

* **Flags de execução** - Definem o comportamento da execução e estão localizadas no arquivo ***ViT_MainTeste.py:***

        RUN_PRETRAIN - Executa (ou não) um pre-treino da rede com a base de 
                       dados selecionada (normalmente definda como FALSE caso 
                       sejam carregados pesos externos)
        RUN_FINETUNE - Executa (ou não) o treino de refinamento, com a 
                       rede pré-treinada
        RUN_EVALUATE - Aplica o script de validação da rede, retornando 
                       as métricas de treinamento.

* **Caminhos (paths) exigidos**:

  * **Script Vit_MainTest.py:**
            
        TF_ENV_PYTHON = Caminho do executável python.exe
                        referente ao ambiente virtual .tf_venv;
        TFRECORD_SCRIPT = Caminho do script responsável por criar os
                          TFRecords;
        TFRECORD_DIR = Diretório onde serão criados: /train/*.tfrecord 
                       e /validation/*.tfrecord
        OUTPUT_DIR = Diretório de checkpoints do ViT
  
  * **Script VisionTransformer_trainer.py:**

        pretrained_path = Define o arquivo de pesos pré-treinados que 
                          devem ser carregados ao modelo criado

  * **Script Create_TFRecords.py:**

        TRAIN_TAR = Caminho do arquivo .tar para treino proveniente 
                    da ImageNet
        VAL_TAR = Caminho do arquivo .tar para validação proveniente 
                  da ImageNet
        TFRECORD_DIR = Diretório onde serão criados: /train/*.tfrecord 
                       e /validation/*.tfrecord
        OUTPUT_DIR = Diretório de checkpoints do ViT
        VAL_ANNOTATIONS = Caminho do arquivo onde serão armazenados as notas 
                          de validação
        DELETE_TARS_AFTER_TFRECORDS = Define se os arquivos .tar originais 
                                      da Imagenet devem ser excluídos após 
                                      a criação dos TFRecords (Booleano)

### 2. Validação para a ResNet-50:

* ***Artigo de referência utilizado:*** He et al., 2015 – “Deep Residual Learning for Image Recognition” 
* ***Dataset utilizado:*** ImageNet (ILSVRC2012)
* ***Acesso:*** https://arxiv.org/abs/1512.03385

#### Definições exigidas em código:

* AINDA NÃO DEFINIDO

---

## 🪟 Funcionalidades do ANTIGO PROJETO

### Localização: Old Versions Backups/Old Project Scripts (OG Networks)
* **OBSERVAÇÃO I:** Para que essa versão opere normalmente, é interessante inserir o conteúdo do diretório na pasta raiz do
  projeto (onde ficam alocados os ambientes virtuais). Isso serve apenas para garantir que nenhum caminho fixo definido
  em código seja quebrado durante a execução do programa. O mesmo deve ser feito com o diretório 'Figures'.


* **OBSERVAÇÃO II:** A antiga versão do projeto utiliza uma estrutura de rede diferente daquelas utilizadas pelo projeto 
  principal (localizado em Main_Project). Este é um dos motivos pelo seu eventual abandono, uma vez que foi optado pela
  utilização de implementações já consolidadas pela literatura científica.


* **OBSERVAÇÃO III:** O diretório *Old Project Scripts (ViT+Interface)* refere-se à uma versão mais "atualizada" do 
  projeto contido no diretório *Old Project Scripts (OG Networks)*, contudo ela também é mais problemática. A ideia era
  integrar a arquitetura ViT préviamente validada ao projeto com interface gráfica, contudo, problemas com 
  com incompatibilidade de bibliotecas, complexidade pelo uso de diferentes ambientes virtuais e instabilidade geral
  levaram ao abandono dessa tentativa de integração.


* **OBSERVAÇÃO IV:** Caso for utilizar os scripts do projeto antigo é necessário atualizar alguns 
  caminhos fixos definidos. Como por exemplo: O caminho do ambiente virtual .vit_venv no arquivo Interface.py do 
  diretório Old Main Project Scripts (ViT+Interface)".

### ✯ Janela Inicial - Old Project Scripts (OG Networks)

A janela inicial do programa é divida em 2 setores: uma área à esquerda dedicada à exibição das mensagens de log 
(informando sobre o status da operação) e uma área à direita dedicada às funcionalidades do sistema. 
Em um primeiro momento a área de funcionalidades exige ao usuário a escolha da arquitetura que será 
utilizada (ViT ou ResNet), por meio de *radiobuttons*. No momento em que o usuário confirma a seleção, 
são apresentados as seguintes funcionalidades (referentes à arquitetura selecionada):

  - **Selecionar dataset** – Permite selecionar a pasta contendo a base de dados para o treinamento. 
    Importante salientar que o diretório escolhido deve conter subpastas (cada uma representando as diferentes classes). 
    Essa função exige a definição do tamanho de entrada, tamanho dos lotes (batch) e porcentagem de divisão para os 
    dados de validação;
---
  - **Construir ResNet50** – Constrói e compila a arquiteura da rede, definindo os parâmetros como formato de entrada, 
    modelo base, camadas da rede, funções de ativação, otimizadores, função custo, dentre outros;  
---
   - **Construir Modelo ViT** – Constrói e compila a arquiteura da rede. Essa função exige a definição de parâmetros 
     específicos à ViT, os quais são detalhados na seção 'Parâmetros exigidos pelo programa' deste mesmo documento;
---
  - **Iniciar treinamento** – Inicia o treinamento da rede. Para ter início, exige a definição de um nome para o 
    arquivo de log e a quantidade de épocas para o treinamento;  
---
  - **Abrir TensorBoard** – Inicia o Tensorboard e abre uma página na web para exibição dos arquivos de log. 
    Esta função exige a escolha do diretório que contém os logs 
    (geralmente está em logs/fit na pasta raiz do executável);  
---
  - **Fechar programa** – encerra a aplicação.

## ⚙️ Parâmetros exigidos pela ResNet-50
- **Input size** - Tamanho que as imagens devem ser redimensionadas para servir como entrada da rede. O valor inserido 
    definira a altura e largura da imagem;


- **Batch size** - Refere-se ao número de amostras de dados que um modelo de aprendizado de máquina processa 
    em uma única iteração;


- **Split (treino/validação)** - Define a porcentagem de dados destinados para treino e validação. 
    Exemplo: 0.2 -> 20% para validação e 80% para treino;


- **Nome para logs** - Permite a definição do nome do arquivo de logs gerado após o treinamento;


- **Épocas** - Permite definir a quantidade de épocas de treinamento.

## ⚙️ Parâmetros exigidos pela ViT
- **Input size** - Análogo ao requisito exigido pela ResNet50;


- **Batch size** - Análogo ao requisito exigido pela ResNet50;


- **Split (treino/validação)** - Análogo ao requisito exigido pela ResNet50;


- **Épocas** - Análogo ao requisito exigido pela ResNet50;


- **Patch size** - O tamanho dos blocos (patches) em que a imagem será dividida. Quanto menor o patch, mais detalhes 
o modelo enxerga desde o início, mas também aumenta a quantidade de patches a processar (mais custo computacional);


- **Projection Dim** - A dimensão do vetor em que cada patch será representado após a projeção linear 
(Dimensões maiores permitem mais capacidade de representação, mas também exigem mais memória e poder de processamento);


- **Transform Layers** - Número de blocos de transformers (compostos por atenção + MLP) empilhados no modelo. 
Quanto mais camadas, mais refinada e abstrata fica a representação;


- **Attention Heads** - Cada camada de atenção pode ter várias "cabeças", que aprendem a focar em diferentes 
aspectos da imagem ao mesmo tempo;


- **MLP Units** - Número de neurônios nas camadas densas (feed-forward layers) que seguem a parte de atenção em 
cada bloco do transformador. Normalmente é um valor maior que o 'projection dim'.


- **Nome para logs** - Análogo ao requisito exigido pela ResNet50;

> 🔎 **Observações Importantes**  
> - O valor de 'Split' deve estar em notação de ponto flutuante, estritamente entre 0.0 e 1.0;
> - O aplicativo indica valores 'padrões' caso o usuário não saiba ao certo o valor de alguns parâmetros;
> - O diretório escolhido para o dataset deve conter subpastas (cada uma representando as diferentes classes);  
> - Seguir as versões dos pré-requisitos à risca, uma vez que versões mais novas podem gerar conflitos na IDE.

---

## 🔧 Pré-requisitos e Instalação de ambientes virtuais
Os requisitos específicos de cada projeto devem ser instalados em ambientes virtuais separdos, de acordo 
com o que foi definido no tópico abaixo 'Tutorial para criação dos ambientes virtuais e instalação de dependências'

### Requisitos gerais:

- Sistema Operacional: **Windows**;  
- Python **3.9.13** (específico);  

### Requisitos para o ambiente .gen_venv (Destinado aos projetos antigos):

* Os pacotes devem ser inseridos no ambiente virtual destinado à execução geral do antigo projeto, incluindo a
condução de treinamentos que utilizam especificamente a arquitetura RESNET-50 e a INTERFACE em PYQT5.


* Os requisitos a seguir estão alocados em General_Requirements.txt, no diretório 
'./Old Versions Backups/Old Requirements/' e podem ser instalados via pip.


- tensorflow **2.10.0**
- numpy **1.23.5**
- scipy **1.13.1**
- tensorboard **2.10.1**
- Pillow
- scikit-learn **1.6.1**
- openpyxl
- PyQt5 **5.15.11**
- pandas **2.3.3**
- tensorflow-datasets **4.7.0**
- protobuf **3.19.0**
- matplotlib
- seaborn
- tqdm
- ipykernel

### Requisitos para o ambiente .tf_venv (Destinado exclusivamente aos projetos de Validação):

* Os pacotes devem ser inseridos no ambiente virtuaL destinado exclusivamente para execução do tensorflow
voltado para a criação dos TFRecords, caso os aquivos .TAR da imagenet sejam utilizados em um processo de
pré-treino ou fine-tunning


* Os requisitos a seguir estão alocados em TFRecords_Requirements.txt, no diretório 
'./Requirements/' e podem ser instalados via pip.

- tensorflow **2.10.0**
- numpy **1.23.5**
- tensorflow-datasets **4.7.0**
- absl-py **0.12.0 (Ou superior)**
- tqdm
- pillow
- scipy

### Requisitos para o ambiente .vit_venv:

* Os pacotes devem ser inseridos no ambiente virtual destinado à condução de treinamentos
que utilizam especificamente a arquitetura VISION TRANSFORMERS, uma vez que, no projeto de validação
é utilizado uma versão do jax/flax que é incompatível com a versão do Tensorflow exigida por outros módulos


* Os requisitos a seguir estão alocados em ViT_Requirements.txt, no diretório 
'./Requirements/' e podem ser instalados via pip.

- numpy **1.26 (Ou superior)**

- jax[cuda11_pip] **0.4.23**
--find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

- jaxlib **0.4.20 (Ou superior)**
- flax **0.8.3**
- optax **0.2.4**
- chex **0.1.87 (Ou superior)**
- orbax-checkpoint **0.3.5**
- einops **0.3.0 (Ou superior)**
- absl-py **0.12.0 (Ou superior)**
- ml-collections **0.1.0 (Ou superior)**
- clu **0.0.3 (Ou superior)**
- tensorflow-datasets **4.7.0 (Ou superior)**
- tensorflow entre **2.13 e 2.16**
- tqdm
- scipy

### Requisitos para o ambiente .cnn_venv (Destinado exclusivamente ao projeto de Validação da ViT):

* Os pacotes devem ser inseridos no ambiente virtual destinado à condução de treinamentos
que utilizam especificamente a arquitetura RESNET-50.


* Os requisitos a seguir estão alocados em ResNet_Requirements.txt, no diretório 
'./Requirements/' e podem ser instalados via pip.

AINDA NÃO DEFINIDOS

---

### OBSERVAÇÃO IMPORTANTE: 
* Os requisitos destinados ao ambiente .vit_venv também servem para o projeto principal, sendo o ambiente virtual
utilizado para sua execução, contudo, os requisitos para este projeto em específico estão localizados em 
'.../Requirements/ViT_Requirements.txt' exclusivamente por uma questão de organização.

---
### Tutorial para criação dos ambientes virtuais e instalação de dependências:

- **Ambiente virtual .gen_venv (Válido apenas para implementações antigas):** 

      > cd Projeto-Classificadores
      > python -m venv .gen_venv
      > .\.gen_venv\Scripts\activate
      > pip install -r .\Old Versions Backups\Old Requirements\General_Requirements.txt

- **Ambiente virtual .vit_venv:**

      > cd Projeto-Classificadores
      > python -m venv .vit_venv
      > .\.vit_venv\Scripts\activate
      > pip install -r .\Requirements\ViT_Requirements.txt

- **Ambiente virtual .tf_venv:**

      > cd Projeto-Classificadores
      > python -m venv .tf_venv
      > .\.tf_venv\Scripts\activate
      > pip install -r .\Requirements\TFRecords_requirements.txt

- **Ambiente virtual .cnn_venv:**

      > cd Projeto-Classificadores
      > python -m venv .cnn_venv
      > .\.cnn_venv\Scripts\activate
      > pip install -r .\Requirements\ResNet_Requirements.txt

- **OBSERVAÇÃO 1:** Sempre lembrar de fechar o ambiente virtual, caso precise utilizar o outro.


- **OBSERVAÇÃO 2:** Seguir o tutorial sobre a instalação da GPU para o Tensorflow ANTES de 
criar o ambiente virtual e instalar as dependências.

---

## 🛠️ Tutorial para instalar GPU para TensorFlow no Windows:
### Esse tutorial garante o uso da GPU durante o treinamento para o projeto principal, bem como a validação da ResNet-50

1. **Desinstalar pacotes conflitantes (opcional, mas recomendado):** utilizar o comando *pip uninstall tensorflow 
tensorflow-gpu tensorflow-intel* ou desinstalar manualmente via explorador do windows;


2. **Instalar TensorFlow GPU 2.10:** versão da biblioteca específica para esta implementação;


3. **Baixar & instalar CUDA 11.2:** Download oficial pelo site da NVIDIA 
*https://developer.nvidia.com/cuda-11.2.0-download-archive*;


4. **Baixar cuDNN 8.1 para CUDA 11.2:** Requer login NVIDIA Developer (gratuito) e pode ser acessado pelo link 
*https://developer.nvidia.com/rdp/cudnn-archive#a-collapse811-110*. O resultado do download será um arquivo .ZIP;


5. **Copiar conteúdo das pastas baixadas (cuDNN 8.1):** É necessário extrair o conteúdo compactado e mover o 
conteúdo das pastas, seguindo o seguinte esquema:

conteúdo da pasta **bin**  → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\

conteúdo da pasta **lib**  → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64\

conteúdo da pasta **include** → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include

6. **Adicionar ao PATH:** Adicionar as seguintes entradas ao PATH do Windows e reiniciar o computador:

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

7. **Testar o download:** Abrir o console python e inserir os seguintes comandos:

import tensorflow as tf\
print(tf.config.list_physical_devices('GPU'))

Se o resultado for algo como *[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]*, indica que o 
processo foi realizado com sucesso.

## 🛠️ Instalação da GPU para os arquivos de validação da arquitetura ViT:
### Esse tutorial garante o uso da GPU durante o treinamento para o projeto principal, bem como a validação da ResNet-50

A partir das versões 0.4.x, o JAX não fornece mais wheels pré-compilados para CUDA 11.2.
Os wheels atuais são distribuídos apenas para: CUDA 11.8 e CUDA 12.x. Por isso torna-se necessário
fazer a instalação de uma versão mais recente (sem alterar o funcionamento do CUDA 11.2).

* **Este é o motivo pelo qual se faz necessário a definição de dois ambientes virtuais distintos. A implementação do ViT com o JAX exige dependências que podem quebrar a implementação anterior.**

Após seguir o tutorial anterior para instalar GPU para TensorFlow, é necessário seguir as seguintes etapas:

1. **Confirmar sua placa NVIDIA e drivers:**

Abra o terminal e digite:

        nvidia-smi
Se aparecer algo como:

        Driver Version: 535.xx
        CUDA Version: 12.2
Está tudo perfeito (JAX usará o driver, não o toolkit).

2. **Baixar o CUDA Toolkit 11.8 (site oficial NVIDIA):** Download oficial pelo site da NVIDIA 
https://developer.nvidia.com/cuda-11-8-0-download-archive


3. **Instalar CUDA 11.8 sem afetar o CUDA 11.2:** 
      
       1) Execute o instalador como administrador.
       2) Selecione Custom (Advanced)
       3) Desmarque estes itens:
           - "Driver" (não reinstalar driver NVIDIA)
           - "Nsight" (não precisa)
           - "CUDA documentation" (opcional)  
    
4. **Não modifique** os caminhos nas variáveis PATH do sistema, nem as instalações de CUDA anteriores
OBS: O instalador colocará o CUDA 11.8 em C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\ 


5. **Verificar se CUDA 11.8 foi instalado corretamente:** 

       Abra o terminal e digite o comando:
           dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
        
       Se aparecer nvcc.exe → deu certo.

6. **Não adicione o CUDA 11.8 ao PATH:** O JAX usa as DLLs diretamente da pasta do CUDA Toolkit, com isso Mantemos o ambiente do TensorFlow totalmente intacto.


7. **Instalar os requisitos referentes à validação da arquitetura ViT**: vit_requirements.txt


8. **Testar o download:** Abrir o console python e inserir os seguintes comandos:

        python -c "import jax; print(jax.devices())"

Se o resultado for algo como [gpu(id=0)], indica que o processo foi realizado com sucesso.

---

## ⚖️ Downlaod dos pesos pré-treinados

Os pesos utilizados para inicializar a arquitetura ViT foram obtidos por meio do repositório
oficial de implementação desta arquitura citado no artigo 
“An Image is Worth 16x16 words: Transformers for Image Recognition at Scale”.
Podendo ser diretamente acessado pelo link: https://console.cloud.google.com/storage/browser/vit_models/imagenet21k 

## ▶️ Modo de uso para o projeto principal

1. Abrir o projeto em uma IDE python de sua preferência e criar o ambiente virtual seguindo as orientações
   fornecidas préviamente (recomendo o PyCharm Community Edition);


2. Instalar os pacotes requeridos pela aplicação (Ver seção de pré-requisitos);


3. Definir os hiperparâmetros exigidos nos arquivos ResNet_Main.py ou ViT_Main.py (De acordo com a arquitetura
selecionada);


4. Para iniciar um treinamento utilizando a arquitetura ResNet, executar o arquivo ResNet_Main.py (Localizado no 
diretório '.\Main_Project');


5. Para iniciar um treinamento utilizando a arquitetura ViT, executar o arquivo ViT_Main.py (Localizado no 
diretório '.\Main_Project');


6. Selecionar o Dataset e o arquivos de pesos, conforme for solicitado pela aplicação;


7. Aguardar o término do treinamento e avaliar os arquivos gerados com as métricas de treinamento coletadas.

---

## ▶️ Modo de uso para a validação das arquiteturas

1. Abrir o projeto em uma IDE python de sua preferência e criar o ambiente virtual seguindo as orientações
   fornecidas préviamente (recomendo o PyCharm Community Edition);


2. Instalar os pacotes requeridos pela aplicação (Ver seção de pré-requisitos);


3. Realizar o download da base de dados ImageNet (Utilizada pela literatura base);


4. Atualizar em código os caminhos exigidos (Para o dataset e para o armazenamento dos TFRecords);


5. Caso queira aplicar a validação da ResNet-50, executar o arquivo 'ResNet50_MainTest.py';


6. Caso queira aplicar a validação da ViT, executar o arquivo 'ViT_MainTest.py';


7. Aguardar até o encerramento do treino para obter o arquivo de pesos e logs.

---

## 🕷️ Bugs conhecidos

* N/A

---

## ⚠️ Erros Comuns

| Erro                                  | Causa provável                            | Solução                                                                     |
|---------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| ❌ Erro ao abrir base de dados         | Estrutura de arquivos inválida            | Verifique se o diretório contém as subpastas como categorias                |
| ❌ Aplicativo não abre                 | Python ou dependências ausentes           | Reinstale dependências                                                      |
| ❌ Travamento ou fechamento inesperado | Instabilidade de código                   | Contatar desenvolvedor                                                      |
| ❌ Erro de dependências                | Versões incorretas ou Falha na instalação | Garntir a instalação das requisitos de acordo com os tutoriais apresentados |

---

## 🆕 Atualizações / Changelog

- **v0.1.0**
  - Implementação inicial das arquiteturas, não seguindo nenhum trabalho correlato;
  - Unificação de ambas as arquiteturas em um único programa com interface gráfica
  utilizando a biblioteca PyQt5.

- **v0.2.0**
  - Correção de bugs para o treinamento da ViT (Que fechava sozinho após o treino);
  - Aumento da robustez dos modelos e otimização do fluxo de treinamento.

- **v0.3.0**
  - Re-estruturação da organização dos diretórios;
  - Utilização de novas implementações para a criação da rede, seguindo estudos já consolidados;
  - Implemetação de protocolos para garantir a confiabilidade na construção das redes;
  - Maior grau de documentação do código (via comentários) e documentos separados para 
  explicação de implementações e resultados.

- **v0.4.0**
  - Exclusão da interface gráfica em razão de incompatibilidade de pacotes;
  - Foco na implementação, validação e experimentação utilizando a arquitetura ViT;
  - Nova re-estruturação da organização dos diretórios;
  - Organização das implementações anteriores em diretórios de Backup;

---

## 👨‍💻 Autores / Contribuidores

- Marcio Salmazo Ramos (Desenvolvedor)
- Maurício Cunha Escarpinati (Orientador - UFU) 
- Daniel Duarte Abdala (Co-orientador - UFU)  

