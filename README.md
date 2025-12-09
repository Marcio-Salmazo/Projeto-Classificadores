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

## 📂 Estrutura de diretórios do projeto

O repositório está organizado de acordo com a seguinte estrutura:

    ├── .idea/
    ├     └── (...) 
    ├── Main Project/
    ├     └── (...)
    ├── Network_Validation/
    ├     └── (...) 
    ├── __pycache__/
    ├     └── (...) 
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    └── vit_requirements.txt

### Diretórios mais relevantes:

1. **Main Project** &rarr; Contém os arquivos referentes à implementação do projeto principal
   voltado para a classificação das imagens de camundongos. Além de agregar os diretórios 
   responsáveis por armazenar logs e pesos de treinamento;
2. **Network Validation** &rarr; Contém os arquivos referentes à implementação do projeto 
   voltado para a replicação de estudos envolvendo a ResNet e a Vit. A ideia é construir a 
   arquitetura fiel à um estudo científico consolidado para garantir a confiabilidade de seu
   funcionamento.

## 📂 Estrutura da base de dados para utilização do projeto principal

Para que o programa reconheça devidamente a base de dados, é necessário seguir a seguinte estrutura:

    ├── Diretório contendo a base de dados (Arquivos .png)/\
    ├── Classe 1 (Diretótio)/\
    │   └── Conjunto de imagens (Arquivos .png)/\
    ├── Classe 2 (Diretótio)/\
    │   └── Conjunto de imagens (Arquivos .png)/\
    │   ...\
    ├── Classe N (Diretótio)/\
    │   └── Conjunto de imagens (Arquivos .png)/\

## ✔️ Testes de confiabilidade

Com o intuito de garantir que as implementações das arquiteturas ResNet-50 e Vision Transformer (ViT) 
estejam corretas e produzam resultados condizentes com aqueles descritos na literatura científica, 
foi feita a replicação de dois estudos  consolidados na área. 
A obtenção de resultados equivalentes aos reportados nos trabalhos permite validar o comportamento 
das redes, garantindo que suas funcionalidades estejam corretamente implementadas. 
A partir dessa base confiável, torna-se possível introduzir modificações pontuais nas arquiteturas, 
para adaptá-las aos objetivos específicos deste projeto.

Os arquivos referentes aos experimentos de validação conduzidos para a arquitetura ResNet-50 
encontram-se organizados no diretório ***Network_Validation***. Esse diretório contém tanto as 
implementações dos algoritmos descritos nos estudos analisados, quanto os artigos científicos 
utilizados como referência, reunindo em um único local todo o material necessário para consulta e 
reprodutibilidade dos testes.

### 1. Validação para a ViT (Vision Transformer):

* ***Artigo de referência utilizado:*** Dosovitskiy et al., 2021 – “An Image is Worth 16x16 words: Transformers for Image Recognition at Scale”
* ***Dataset utilizado:*** ImageNet (ILSVRC2012)
* ***Acesso:*** https://arxiv.org/abs/2010.11929

### 2. Validação para a ResNet-50:

* ***Artigo de referência utilizado:*** He et al., 2015 – “Deep Residual Learning for Image Recognition” 
* ***Dataset utilizado:*** ImageNet (ILSVRC2012)
* ***Acesso:*** https://arxiv.org/abs/1512.03385

---

## 🪟 Interface e Funcionalidades do projeto principal
### Janela Inicial

A janela inicial do programa é divida em 2 setores: uma área à esquerda dedicada à exibição das mensagens de log (informando sobre o status da operação) e uma área à direita dedicada às funcionalidades do sistema. Em um primeiro momento a área de funcionalidades exige ao usuário a escolha da arquitetura que será utilizada (ViT ou ResNet), por meio de *radiobuttons*. No momento em que o usuário confirma a seleção, são apresentados as seguintes funcionalidades (referentes à arquitetura selecionada):


  - **Selecionar dataset** – Permite selecionar a pasta contendo a base de dados para o treinamento. Importante salientar que o diretório escolhido deve conter subpastas (cada uma representando as diferentes classes). Essa função exige a definição do tamanho de entrada, tamanho dos lotes (batch) e porcentagem de divisão para os dados de validação;
  - **Construir ResNet50** – Constrói e compila a arquiteura da rede, definindo os parâmetros como formato de entrada, modelo base, camadas da rede, funções de ativação, otimizadores, função custo, dentre outros;  
   - **Construir Modelo ViT** – Constrói e compila a arquiteura da rede. Essa função exige a definição de parâmetros específicos à ViT, os quais são detalhados na seção 'Parâmetros exigidos pelo programa' deste mesmo documento  
  - **Iniciar treinamento** – Inicia o treinamento da rede. Para ter início, exige a definição de um nome para o arquivo de log e a quantidade de épocas para o treinamento;  
  - **Abrir TensorBoard** – Inicia o Tensorboard e abre uma página na web para exibição dos arquivos de log. Esta função exige a escolha do diretório que contém os logs (geralmente está em logs/fit na pasta raiz do executável);  
  - **Fechar programa** – encerra a aplicação.

## ⚙️ Parâmetros exigidos pela ResNet-50
- **Input size** - Tamanho que as imagens devem ser redimensionadas para servir como entrada da rede. O valor inserido definira a altura e largura da imagem;
- **Batch size** - Refere-se ao número de amostras de dados que um modelo de aprendizado de máquina processa em uma única iteração;
- **Split (treino/validação)** - Define a porcentagem de dados destinados para treino e validação. Exemplo: 0.2 -> 20% para validação e 80% para treino;
- **Nome para logs** - Permite a definição do nome do arquivo de logs gerado após o treinamento;
- **Épocas** - Permite definir a quantidade de épocas de treinamento.

## ⚙️ Parâmetros exigidos pela ViT
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

## 🔧 Pré-requisitos e Instalação

### Requisitos gerais:

- Sistema Operacional: **Windows**;  
- Python **3.9.13** (específico);  

### Requisitos para o projeto principal:

- Tensorflow **2.10.0**;
- Numpy **1.23.5**;
- Scipy **1.13.1**;
- Protobuf **3.20.2**;
- Tensorboard **2.10.1**;
- Pillow (Sem versão específica, pode ser a mais atual);
- CUDA 11.2 (Para uso da GPU);
- CuDNN 8.1 (Para uso da GPU).
- scikit-learn **1.6.1**
- openpyxl
- PyQt5 **5.15.11**
- pandas **2.3.3**
- tensorflow-datasets **4.7.0**

### Requisitos para as validações da arquitetura:

- tensorflow **2.10.0**
- numpy **1.23.5**
- tensorflow-datasets **4.7.0**
- ml-collections **0.1.0** (ou maior)
- tensorflow-probability  **0.11.1** (ou maior)

- absl-py **0.12.0** (ou maior)
- aqtp diferente da versão **0.1.1** 
- chex **0.0.7** (ou maior)
- clu **0.0.3** (ou maior)
- einops **0.3.0** (ou maior)
- flax **0.6.4** (ou maior)

- jax[cuda11_pip]>=0.4.2
-  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

---
**OBSERVAÇÃO**: Os requisitos específicos podem ser instalados via terminal, porém
é importante destacar que dois ambientes virtuais distintos sejam criados. Um destinado 
especificamente para o projeto principal e para a validação da arquitetura ResNet e outro 
destinado especificamente para os códigos de validação da arquitetura ViT.
---
### Tutorial para criação dos ambientes virtuais e instalação de dependências:

- Para o projeto principal e a validação da arquitetura ResNet-50:

      > cd Projeto-Classificadores
      > python -m venv .venv
      > .\.venv\Scripts\activate
      > pip install -r requirements.txt

- Para a validação da arquitetura ViT:

      > cd Projeto-Classificadores
      > python -m venv .vit_venv
      > .\.vit_venv\Scripts\activate
      > pip install -r vit_requirements.txt
- **OBSERVAÇÃO 1:** Sempre lembrar de fechar o ambiente virtual, caso precise utilizar o outro.

- **OBSERVAÇÃO 2:** Seguir o tutorial sobre a instalação da GPU para o Tensorflow ANTES de 
criar o ambiente virtual e instalar as dependências.

---

## 🛠️ Tutorial para instalar GPU para TensorFlow no Windows:
### Esse tutorial garante o uso da GPU durante o treinamento para o projeto principal, bem como a validação da ResNet-50

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

2. **Baixar o CUDA Toolkit 11.8 (site oficial NVIDIA):** Download oficial pelo site da NVIDIA https://developer.nvidia.com/cuda-11-8-0-download-archive
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


## ▶️ Modo de uso para o projeto principal

1. Abrir o projeto em uma IDE python de sua preferência e criar o ambiente virtual seguindo as orientações
   fornecidas préviamente (recomendo o PyCharm Community Edition);
2. Instalar os pacotes requeridos pela aplicação (Ver seção de pré-requisitos);
3. Executar o arquivo main.py;
4. Carregar um diretório contendo uma base de dados e definir os parâmetros exigidos. Levar em consideração a estrutura de arquivos exigida (Ver seção anterior); 
5. Construir a estrutura da arquitetura;
6. Iniciar o treinamento, definindo os parâmetros exigidos;
7. Aguardar até o encerramento do treino para obter o arquivo de pesos e logs.

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

- **v0.5.0**
  - Versão inicial, contemplando a unificação de ambas as arquiteturas em um único programa

- **v0.8.0**
  - Correção de bugs para o treinamento da ViT (Que fechava sozinho após o treino);
  - Aumento da robustez dos modelos e otimização do fluxo de treinamento;

- **v0.9.0**
  - Re-estruturação dos diretórios
  - Implemetação de algoritmos e protocolos para garantir a confiabilidade das redes (utilizando a literatura científica como base);
 - Maior grau de documentação do código (via comentários) e documentos separados para explicação de implementações e resultados.
---

## 👨‍💻 Autores / Contribuidores

- Marcio Salmazo Ramos (Desenvolvedor)
- Maurício Cunha Escarpinati (Orientador - UFU) 
- Daniel Duarte Abdala (Co-orientador - UFU)  

