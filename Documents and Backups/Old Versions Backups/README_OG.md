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

---

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