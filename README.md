# Classificadores neurais - Visão Geral:

Este repositório contém a implementação de arquiteturas de Deep Learning voltadas para classificação 
de imagens faciais de roedores, com o objetivo de detectar e estimar a intensidade de dor (seguindo
os critérios estabelecidos pela *Mouse Grimmace Scale*). O projeto está inserido no contexto do meu 
trabalho de mestrado e busca automatizar a análise laboratorial, reduzindo a dependência de 
avaliação humana. A proposta utiliza duas abordagens principais:

* **Arquiteturas da famiília ResNet (CNN)** &rarr; aprendizado baseado em convoluções

  * As redes neurais residuais (Residual Neural Networks), popularizadas pela arquitetura ResNet, 
  representam um avanço no campo do aprendizado profundo dentro da área de Visão Computacional. 
  Foi desenvolvida para enfrentar o problema da degradação do desempenho em redes muito profundas, 
  por meio de “atalhos” (skip connections), que permitem que a informação contorne uma ou mais camadas, 
  facilitando o aprendizado de funções residuais em vez de transformações completas.


* **Arquitetura Vision Transformer (ViT)** &rarr; aprendizado baseado em atenção

  * A ViT representa uma mudança de paradigma em visão computacional por adaptar os mecanismos de atenção 
  originalmente desenvolvidos para processamento de linguagem natural ao domínio de imagens. 
  Em vez de processar uma imagem por meio de convoluções, o ViT a divide em pequenos blocos (patches), 
  que operam como "palavras visuais". Esses blocos são então passados por camadas de autoatenção, 
  que permitem ao modelo aprender relações globais entre diferentes regiões da imagem desde as 
  primeiras etapas do processamento.

## 📂 Estrutura de diretórios do projeto

O repositório está organizado de acordo com a seguinte estrutura:

    ├── Documents and Backups/
    ├     └── (...)
    ├── Project_CNN/
    ├     └── (...)
    ├── Project_VIT/
    ├     └── (...)
    ├── Requirements/
    ├     └── (...) 
    ├── .gitignore
    ├
    └──README.md

### Detalhamento dos diretórios:

1. **Documents and Backups** &rarr; Contém os arquivos de documentação do projeto, o que inclui os 
artigos científicos utilizados como referência para a validação das arquiteturas de rede, bem como 
os arquivos referentes à documentação dos experimentos conduzidos, reunindo em um único local todo 
o material necessário para consulta e reprodutibilidade dos testes. 
<br>
<br>
    Adicionalmente, o diretório contém os arquivos referentes à implementação antiga do projeto,
a qual contava com uma interface gráfica construída por meio da biblioteca PyQt5. A ideia central
era criar um ambiente mais amigável ao usuário, contudo, incompatibilidades entre bibliotecas e
a má integração entre os módulos acabaram por consumir bastante tempo do cronograma do projeto, o que
levou à decisão de descartar essa implementação em prol de algo mais simples e objetivo, permitindo
dar foco ao real objetivo do projeto.


2. **Project_CNN** &rarr; Contém os arquivos referentes à implementação da arquitetura 
convolucional resideual, além de agregar os diretórios responsáveis por armazenar logs e 
checkpoints do treinamento. O diretório contém sub-pastas dedicadas à organização das 
implementações voltadas para a validação da rede construída (buscando seguir os mesmos 
parâmetros do artigo de referência) e as implementações voltadas para o objetivo central 
do projeto (Treinamento utilizando uma base de dados com expressões faciais de camundongos).


3. **Project_VIT** &rarr; Contém os arquivos referentes à implementação da arquitetura 
baseada em transformers, além de agregar os diretórios responsáveis por armazenar logs e 
checkpoints do treinamento. O diretório também contém sub-pastas dedicadas à organização das 
implementações voltadas para a validação da rede construída e as implementações voltadas para 
o objetivo central do projeto.


4. **Requirements** &rarr; Contém os arquivos .txt referentes à definição dos requisitos
necessários para cada um dos ambientes virtuais que devem ser criados durante à execução 
dos scripts.

## 📂 Estrutura da base de dados para utilização do projeto principal

Para que o código referente ao projeto principal reconheça devidamente a base de dados, ]
é necessário seguir a seguinte estrutura:

    ├── Diretório contendo a base de dados (Arquivos .png)/
    │    │
    │    ├── Classe 1 (Diretótio)/
    │    │   └── Conjunto de imagens (Arquivos .png)/
    │    ├── Classe 2 (Diretótio)/
    │    │   └── Conjunto de imagens (Arquivos .png)/
    │    │   ...
    │    ├── Classe N (Diretótio)/
    │    │   └── Conjunto de imagens (Arquivos .png)/

Importante destacar que uma das etapas dos scripts principais é a re-organização desta estrutura, de modo que seja
definida uma divisão da base em conjunto de treino e validaçã, de acordo com a porcentagem definida em 
código pelo usuário. Após a execução do Script, o diretório utilizado para o treinamento terá a seguinte estrutura:

    └──Diretório reorganizado (Arquivos .png)/
        │
        ├── train (Diretótio)/
        │    ├── Classe 1 (Diretótio)/
        │    │   └── Conjunto de imagens (Arquivos .png)/
        │    ├── Classe 2 (Diretótio)/
        │    │   └── Conjunto de imagens (Arquivos .png)/
        │    │   ...
        │    └── Classe N (Diretótio)/
        │        └── Conjunto de imagens (Arquivos .png)/
        │   
        └── val (Diretótio)/
             ├── Classe 1 (Diretótio)/
             │   └── Conjunto de imagens (Arquivos .png)/
             ├── Classe 2 (Diretótio)/
             │   └── Conjunto de imagens (Arquivos .png)/
             │   ...
             └── Classe N (Diretótio)/
                 └── Conjunto de imagens (Arquivos .png)/

* Observação: Caso a estrutura já esteja organizada com os devidos diretórios de treino
e validação, o script de organização não é executado e o treino prossegue normalmente.

## ✔️ Testes de confiabilidade

Com o intuito de garantir que as implementações das arquiteturas ResNet e Vision Transformer 
estejam corretas e produzam resultados condizentes com aqueles descritos na literatura científica, 
foi feita a replicação de dois estudos  consolidados na área. A obtenção de resultados equivalentes 
aos reportados nos trabalhos permite validar o comportamento das redes, garantindo que sua construção 
esteja corretamente implementadas. A partir dessa base confiável, torna-se possível introduzir 
modificações pontuais nas arquiteturas, para adaptá-las aos objetivos específicos deste projeto.


* **OBSERVAÇÃO:** Os arquivos *Create_TFRecords.py* e *Process_ImageNet.py* são scripts auxiliares 
utilizados para tratar a base de dados da ImageNet no formato de TFRecords, o qual se caracteriza como
um formato de arquivo binário simples e eficiente, projetado especificamente pela Google para armazenar 
sequências de dados no TensorFlow, permitindo o pré-carregamento e o streaming eficiente de grandes 
volumes de dados, o que acelera o treinamento de modelos de aprendizado de máquina.

### 1. Configurações para a validação da ViT:

* ***Artigo de referência utilizado:*** Dosovitskiy et al., 2021 – “An Image is Worth 16x16 words: 
Transformers for Image Recognition at Scale”
* ***Dataset utilizado:*** ImageNet (ILSVRC2012)
* ***Acesso:*** https://arxiv.org/abs/2010.11929

#### Definições exigidas em código:

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

### 2. Configurações para a validação da ResNet:

* ***Artigo de referência utilizado:*** He et al., 2015 – “Deep Residual Learning for Image Recognition” 
* ***Dataset utilizado:*** ImageNet (ILSVRC2012)
* ***Acesso:*** https://arxiv.org/abs/1512.03385

#### Definições exigidas em código:

* **Parâmetros gerais** - Definem os parâmetros de carregamento de dados e de treinamento. 
Estão localizadas no arquivo ***ResNet_Main.py:***

        - IMAGE_SIZE = Define a dimensão da imagem.
        - NUM_CLASSES = Definição da quantidade de classes para classificação.
        - TRAIN_SIZE = Amostras para treinamento (O valor presente no código é padrão da Imagenet).
        - VAL_SIZE = Amostras para treinamento (O valor presente no código é padrão da Imagenet).
        - BATCH_SIZE = Tamanho dos batches de treinamento.
        - EPOCHS = Quantidade de épocas de treinamento.
        - INITIAL_LR = Learning rate inicial (Ajustado conforme as métricas retornadas).
        - MOMENTUM = Momentum de treinamento.
        - WEIGHT_DECAY = Decaimento dos pesos.
        - LOG_DIR = Nome do diretório onde serão armazenados os arquivos de log.
        - CHECKPOINT = Caminho do diretório onde serão armazenados os checkpoints.

* **Caminhos (paths) exigidos**:


    - TF_ENV_PYTHON = Caminho do ambiente virtual contendo o tensorflow para criação dos TFRecords 
      (Sem entrar em conflitos com o Jax)
    - TFRECORD_SCRIPT = Caminho do script responsável por gerenciar os TFRecords
    - TFRECORD_DIR = Diretório onde serão criados: /train/*.tfrecord e /validation/*.tfrecord
    - OUTPUT_DIR = Diretório de checkpoints do ViT
    - CHECKPOINT_PATH = Caminhos para o checkpoint


## ⚙️ Detalhamento dos parâmetros exigidos pelas redes

### 1. ResNet

- **Input size** - Tamanho que as imagens devem ser redimensionadas para servir como entrada da rede. O valor inserido 
    definira a altura e largura da imagem;

- **Batch size** - Refere-se ao número de amostras de dados que um modelo de aprendizado de máquina processa 
    em uma única iteração;

- **Split (treino/validação)** - Define a porcentagem de dados destinados para treino e validação. 
    Exemplo: 0.2 -> 20% para validação e 80% para treino;

- **Nome para logs** - Permite a definição do nome do arquivo de logs gerado após o treinamento;

- **Épocas** - Permite definir a quantidade de épocas de treinamento.

### 2. Vision Transformers

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
> - Seguir as versões dos pré-requisitos à risca, uma vez que versões mais novas podem gerar conflitos na IDE.


## 🛠️ Tutorial para configurar o uso da GPU para o TensorFlow durante o treinamento com a ResNet:

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

        >> import tensorflow as tf\
        >> print(tf.config.list_physical_devices('GPU'))

Se o resultado for algo como *[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]*, indica que o 
processo foi realizado com sucesso.


## 🛠️ Tutorial para configurar o uso da GPU para o TensorFlow durante o treinamento com a ViT:

A partir das versões 0.4.x, o JAX não fornece mais wheels pré-compilados para CUDA 11.2.
Os wheels atuais são distribuídos apenas para: CUDA 11.8 e CUDA 12.x. Por isso torna-se necessário
fazer a instalação de uma versão mais recente (sem alterar o funcionamento do CUDA 11.2). 

* Após seguir o tutorial anterior para instalar GPU para TensorFlow, é necessário seguir as seguintes etapas:

1. **Confirmar A placa NVIDIA e drivers:**

   * Abra o terminal e digite:

           nvidia-smi
   * Se aparecer algo como:

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


6. **Não adicione o CUDA 11.8 ao PATH:** O JAX usa as DLLs diretamente da pasta do CUDA Toolkit, 
com isso Mantemos o ambiente do TensorFlow totalmente intacto.

## ▶️ Tutorial para a Execução do projeto principal com a arquitetura ResNet

1. Criar um ambiente virtual python (Especificamente na versão 3.9.13) e instalar dependências. 
No diretório raíz do projeto (*\Projeto-Classificadores*) abrir o CMD e inserir os seguintes comandos:

        >> python -m venv .cnn_venv
        >> .\.cnn_venv\Scripts\activate
        >> pip install -r .\Requirements\CNN_Requirements.txt

2. Abrir o diretório *\Projeto-Classificadores\Project_CNN\Main* em alguma IDE python 
de preferência do usuário (Recomendo o uso do PyCharm). No caso da IDE PyCharm, o usuário deve:

        >> Main Menu -> Settings -> Project -> Python Interpreter
        >> Add Interpreter -> Add Local Interpreter
        >> Select Existing -> Selecionar o ambiente virtual criado na primeira etapa 

3. Configurar os parâmetros do arquivo ResNet_Main.py localizado em *Project_CNN/Main/ResNet_Main.py*. 
O usuário pode escolher configurações mais adequadas ao teste a ser conduzido.


4. Executar o arquivo ResNet_Main.py diretamente da IDE. O script vai solicitar ao usuário para selecionar
a base de dados que deve ser utilizada no treinamento, a qual deve estar organizada de acordo com o
que foi descrito na terceira seção deste documento. 


5. Ao final do treinamento serão gerados arquivos de LOG e Checkpopints nos caminhos definidos
pelo usuário no Script principal. Cabe ao usuário organizá-los da forma que julgar ser mais
conveniente

## ▶️ Tutorial para a Execução do projeto principal com a arquitetura Vision Transformer

Durante o desenvolvimento, foi identificado um problema crítico, no qual o TensorFlow e JAX possuem dependências incompatíveis (numpy, ml-dtypes, CUDA).
Por isso, a solução adotada foi a separação completa do pipeline de execução
em dois ambientes virtuais distintos:

* Ambiente 1 (TensorFlow + Numpy) → Utilizado para o preprocessamento das imagens e carregamento dos dados de entrada.
* Ambiente 2 (JAX) → Treinamento efeitivo do modelo, com utilização da GPU

***Observação:*** Foi necessário o uso do WSL (Windows Subsystem for Linux) para garantir a compatibilidade do CUDA + JAX. Caso contrário, a utilização
da GPU era prejudicada, impactando diretamente o treino 

1. **Instalação do WSL:**
   
    - Abrir o CMD no modo administrador e executar o comando:
      
          >> wsl --install

    - Verificar a instalação com o comando:
  
          >> wsl

    - Atualizar o sistema Linux:
          
          >> sudo apt update && sudo apt upgrade -y

2. **Configuração da GPU**
   
      - Inserir o comando dentro do terminal WSL:
          
            >> nvidia-smi 
      
      - Caso a GPU e o CUDA sejam exibidos, o processo deu certo. Vale ressaltar que é necessário a instalação dos Drivers NVIDIA e do CUDA compatível (conforme apresentado nas seções anteriores) no próprio Windows.

3. **Instalação do Python - Versão 3.10**

      - Dentro do terminal do WSL, executar os seguintes comandos em ordem:

            >> sudo apt install software-properties-common -y
            >> sudo add-apt-repository ppa:deadsnakes/ppa -y
            >> sudo apt update
            >> sudo apt install python3.10 python3.10-venv python3.10-dev -y
            >> python3.10 --version

4. **Copiar arquivos para o ambiente WSL**
   
      - Os arquivos python para a execução da ViT, bem como o dataset precisam ser copiados para o ambiente linux do WSL, para isso, recomenda-se a cópia de todo o diretório 'Project_VIT'. Este processo pode ser feito por meio do comando:

            >> mkdir vit_project
            >> cp -r /mnt/c/Users/'SeuUsuario'/.../Projeto-Classificadores/Project_VIT/* .

5. **Criação dos ambientes virtuais**
   
      - Os ambientes virtuais devem ser criados na pasta raíz para onde foram extraídos os conteúdos de *Project_VIT*
  
      - Criação do ambiente com TensorFlow para a gestão do carregamento de dados, bem como o processamento das imagens de entrada:
   
            >> python3.10 -m venv vit_tf_env
            >> source vit_tf_env/bin/activate
            >> pip install --upgrade pip
            >> pip install tensorflow==2.15.0 numpy==1.26.4 tensorflow-datasets

      - Criação do ambiente com JAX para a gestão do treinamento com a GPU:
      - ***Observação:*** Alguns do pacotes exigidos nesta etapa podem gerar conflitos de dependências, por isso, foi adotado a utilização de um arquivo de Constraints, para forçar o versionamento correto de alguns pacotes. O arquivo de Constraints está localizado na raiz do diretório *Project_VIT*, denominado 'ViT_Constraints.txt'.
   
            >> pip install "jax[cuda12_pip]==0.4.28" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
            >> pip install flax==0.8.3 optax==0.2.2 chex==0.1.86 orbax-checkpoint -c constraints.txt
            >> pip install numpy scipy einops absl-py ml-collections clu tqdm -c constraints.txt
            

      - ***Observação 2:*** Para testar se a operação deu certo, verificar:

            -----------------------------------------------------
            >> pip list | grep jax
            -----------------------------------------------------
            O resultado deve ser algo como:
            -> jax     0.4.28
            -> jaxlib  0.4.28+cuda

            -----------------------------------------------------
            >> python -c "import jax; print(jax.devices())"
            -----------------------------------------------------
            O resultado deve ser algo como:
            -> [GpuDevice(id=0)]

5. **Configuração da IDE VsCode para a execução**
      
      - É necessário ter o VsCode instalado no Windows
      - É necessário fazer o download da extensão WSL (da própria Microsoft)
      - Ainda no terminal do WSL, na pasta raiz do projeto (Project_VIT), inserir o seguinte comando:

            >> code .
      
      - O VsCode será aberto automaticamente no local do projeto. Para confirmar se ele está devidamente conectado ao WSL basta verificar no canto inferior esquerdo a presença de 'WSL: Ubuntu'
      - Selecionar o Interpretador python, apontando para os ambientes virtuais criados. Para isso, basta executar:
            
            >> No proprio VsCode:
            >> Ctrl + Shift + P → Python: Select Interpreter
            >> Escolher o caminho: /home/'usuario'/'venv'/bin/python
            
      - ***Observação 2:*** Modificar 'usuario' e 'venv' de acordo com o que foi definido pelo usuário  

6. **Execução do Script para o processamento dos dados**
   
      - As imagens do dataset e as labels, são préviamente transformadas para o formato do Numpy, a fim de que a etapa de seu processamento seja totalmente separada da etapa de treinamento (Evitando conflito de bibliotecas), por isso, inicialmente o script *ViT_DataLoader.py* deve ser excutado.
  
        1. Selecionar no VsCode o ambiente virtual *'vit_tf_env'*, o qual contém o Tensorflow para o carregamento dos dados.
        2. Executar separadamente o Script *ViT_DataLoader.py*
   
      - Serão criados arquivos .npy responsáveis por armazenar os dados das imagems (x_train, x_val) e suas respectivas labels (y_train, y_val), as quais serão carregadas automaticamente pelo Script de treinamento.
      - ***Observação:*** Os arquivos gerados devem ficar alocados no mesmo diretório que o Script *ViT_Main.py*

7. **Execução do Script para o treinamento**

     - Selecionar no VsCode o ambiente virtual *'vit_jax_env'*, o qual contém o pipeline com JAX + CUDA para o treinamento via GPU.
     - Executar o Script *ViT_Main.py* diretamente do VsCode.
  
     - ***Observação:*** O usuário pode alterar algumas configurações referentes à hiperparâmetros no Script *ViT_Main.py* antes de sua execução, conforme julgar necessário
  
     - ***Observação 2:*** Após o término do treinamento, os logs e checkpoints serão armazenados no diretório 'Results'


- **OBSERVAÇÃO 1:** Sempre lembrar de fechar o ambiente virtual, caso precise utilizar o outro, o que pode ser feito pelo comando 'deactivate' no terminal do WSL, ou pelo próprio VsCode.

- **OBSERVAÇÃO 2:** Seguir o tutorial sobre a instalação da GPU para o Tensorflow ANTES de criar o ambiente virtual e instalar as dependências. De modo geral, recomenda-se seguir o passo-a-passo NA ORDEM DE APRESENTAÇÃO descrito neste documento. 

## ⚖️ Downlaod dos pesos pré-treinados

Os pesos utilizados para inicializar a arquitetura ViT foram obtidos por meio do repositório
oficial de implementação desta arquitura citado no artigo 
“An Image is Worth 16x16 words: Transformers for Image Recognition at Scale”.
Podendo ser diretamente acessado pelo link: https://console.cloud.google.com/storage/browser/vit_models/imagenet21k 

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
  - Organização das implementações anteriores em diretórios de Backup.

- **v0.5.0**
  - Correção do carregamento do dataset específico para este projeto;
  - Foco na implementação, validação e experimentação utilizando a arquitetura ResNet-50;
  - Adequação da arquitetura ResNet-50 ao projeto principal;
  - Re-organização do diretório de documentação;
  - Inclusão do dataset utilizado no mestrado ao repositório.

- **v0.6.0**
  - Reorganização completa dos diretórios do projeto, separando o conteúdo destinado à ViT e à ResNet;
  - Utilização de ambientes separados para a execução da ViT, visando evitar problemas de incompatibilidade de pacotes
  - Utilização do WSL para evitar incompatibilidade (ViT)
---

## Dados pessoais
**Nome:** Marcio Salmazo Ramos \
**Redes sociais e contato:**

| [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marcio-ramos-b94669235) | [![Instagram](https://img.shields.io/badge/-Instagram-%23E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/marcio.salmazo) | [![Gmail](https://img.shields.io/badge/Gmail-333333?style=for-the-badge&logo=gmail&logoColor=red)](mailto:contato.marcio.salmazo19@gmail.com) | [![GitHub](https://img.shields.io/badge/GitHub-0077B5?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Marcio-Salmazo) |
|---|---|---|---|

## 👨‍💻 Autores / Contribuidores

- Marcio Salmazo Ramos (Desenvolvedor - Aluno de mestrado)
- Maurício Cunha Escarpinati (Orientador - UFU) 
- Daniel Duarte Abdala (Co-orientador - UFU)  

