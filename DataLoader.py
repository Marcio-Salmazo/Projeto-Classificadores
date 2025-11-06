import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:

    # Construtor da classe, aqui serão definidos
    # o tamanho padrão das imagens e o batch size
    # para a definição dos grupos de validação e
    # treinamento

    def __init__(self, img_size, batch_size):
        self.img_size = img_size
        self.batch_size = batch_size
        # self.train_generator = None
        # self.validation_generator = None

    # A função abaixo é responsável por gerar a tela
    # do explorer que permite ao usuário selecionar
    # um diretório no sistema. Tal seleção retorna
    # o caminho do diretório

    # FUTURAMENTE JOGAR ESSA FUNÇÃO PARA O MODEL

    def load_path(self):
        root = tk.Tk()  # Classe do tkinter
        root.withdraw()  # Oculta a janela principal

        path = filedialog.askdirectory(title="Selecionar pasta com dataset")

        return path

    # A função a baixo é responsável por gerenciar os dados
    # presentes no diretório selecionado. Aqui as imagens são
    # rescalonadas e separadas entre treino e validação
    # RE-ESCREVER
    # Importante salientar que cada sub-pasta dentro do diretorio
    # representas as diferentes classes do dataset

    def process_data(self, path):
        # O imageDataGenerator uma ferramenta do TensorFlow / Keras
        # que gera lotes(batches) de imagens de forma eficiente,
        # aplicando pré - processamento e (opcionalmente) data
        # augmentation — tudo isso em tempo real, enquanto o modelo treina.
        #
        # É uma ferramenta muito útil quando trabalhamos com muitos
        # arquivos de imagem e é desejado:
        #
        #    a - Evitar carregar tudo na memória.
        #    b - Automatizar o carregamento, normalização e divisão treino / validação.
        #    c - Aplicar transformações como rotação, zoom, flips, etc.
        #
        # obs: O rescale=1./255 faz uma normalização, convertendo os valores para o intervalo [0, 1].
        # obs2: O validation_split define a divisão dos dados entre validação e treinamento e só
        #       dEssa divisão só funciona se você usar depois subset='training' e subset='validation'
        #       ao chamar flow_from_directory().

        # train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
        train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

        # Gera um batch de imagens, a partir do ImageDataGenerator

        train_generator = train_datagen.flow_from_directory(
            path,  # Define o caminho do diretório
            target_size=(self.img_size, self.img_size),  # Redimensiona as imagens para 224x224
            batch_size=self.batch_size,  # Define a quantidade de imagens carregadas por vez
            class_mode='categorical',  # Indica que é um problema de classificação multi-classe (e.g., softmax)
            subset='training'  # Especifica que o subset carregado pertence à fatia de 80% destinados ao treino
        )

        # Segue a mesma lógica do bloco de código anterior

        validation_generator = train_datagen.flow_from_directory(
            path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'  # Especifica que o subset carregado pertence à fatia de 20% destinados à validação
        )

        return train_generator, validation_generator
