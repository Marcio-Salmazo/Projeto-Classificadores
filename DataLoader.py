import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:

    def __init__(self, img_size=224, batch_size=16):

        """
            Construtor da classe, aqui serão definidos
            o tamanho padrão das imagens e o batch size
            para a definição dos grupos de validação e
            treinamento. Adicionalmente são definidos
            os layers para augmentation.
        """

        self.img_size = img_size
        self.batch_size = batch_size

        # Augmentations
        self.augment = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomBrightness(0.25),
            tf.keras.layers.RandomContrast(0.25),
            tf.keras.layers.GaussianNoise(0.05),
            tf.keras.layers.RandomCrop(img_size, img_size),
        ])

    def load_path(self):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        return filedialog.askdirectory(title="Selecionar pasta com dataset")

    """
        A função a baixo é responsável por gerenciar os dados
        presentes no diretório selecionado. Aqui as imagens são
        submetidas a um pré-processamento e separadas em grupos
        destinados ao treino e à validação
    
        Importante destacar que cada sub-pasta dentro do diretorio
        representas uma classe distinta.
    """
    def process_data(self, path):

        """
            O imageDataGenerator uma ferramenta do TensorFlow / Keras
            que gera lotes(batches) de imagens de forma eficiente,
            aplicando pré - processamento e (opcionalmente) data
            augmentation — tudo isso em tempo real, enquanto o modelo treina.

            É uma ferramenta muito útil quando trabalhamos com muitos
            arquivos de imagem e é desejado:

                a - Evitar carregar tudo na memória.
                b - Automatizar o carregamento, normalização e divisão treino / validação.
                c - Aplicar transformações como rotação, zoom, flips, etc.

            obs: O rescale = 1./255 faz uma normalização, convertendo os valores para o intervalo [0, 1].
            obs2: O validation_split define a divisão dos dados entre validação e treinamento.
                  Essa divisão só funciona se você usar depois subset='training' e subset='validation'
                  ao chamar flow_from_directory().
        """
        train_gen_raw = ImageDataGenerator(
            rescale=1 / 255.0,
            validation_split=0.2
        )

        val_gen_raw = ImageDataGenerator(
            rescale=1 / 255.0,
            validation_split=0.2
        )

        # Gera um batch de imagens, a partir do ImageDataGenerator
        train_raw = train_gen_raw.flow_from_directory(
            path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training"
        )

        # Gera um batch de imagens, a partir do ImageDataGenerator
        val_raw = val_gen_raw.flow_from_directory(
            path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation"
        )

        # Aplicando augmentation com tf.data

        '''
            O AUTOTUNE seria como dizer ao TensorFlow:
            “Otimize automaticamente o número de threads e o paralelismo para acelerar o carregamento dos dados.”
        '''
        AUTOTUNE = tf.data.AUTOTUNE

        '''
            Convertendo o generator do Keras para um Dataset
            train_raw é um generator do ImageDataGenerator. A operação abaixo converte o generator em um objeto 
            Dataset, que é:

                * Otimizado em C++
                * Paralelizável
                * Compatível com .map(), .batch(), .prefetch()
                * Muito mais rápido
                
            No 'output_signature' O TensorFlow precisa saber como será cada item produzido pelo generator, 
            por isso, é necessário informar:

                * Para as imagens (X):
                    shape: (None, img_size, img_size, 3)
                    dtype: float32
                    O None no início significa: o batch ainda não está batido — vem imagem por imagem após unbatch.
                
                * Para os rótulos (y):
                    shape: (None, num_classes)
                    dtype: float32
                    Isso corresponde ao one-hot encoding.
                    
            O 'flow_from_directory' entrega batches. Mas buscamos aplicar augmentação imagem por imagem. Então:
            
                .unbatch() divide cada batch em imagens individuais, de modo que o .map() consiga 
                aplicar augmentação uma imagem por vez.
            
            Como analogia, podemos imaginar um pacote com 16 fotos:
            Se o objetivo é aplicar filtros individualmente, torna-se necessário 
            abrir o pacote e pegar uma foto por vez.
            
            Em .map(...) para cada imagem x aplica-se o pipeline de augmentação Keras (self.augment)
            e retorna a imagem transformada, bem como seu rótulo y.

                * num_parallel_calls = AUTOTUNE é usado para que o TensorFlow aplique a augmentação 
                em múltiplos núcleos de CPU automaticamente.
                
            Depois de transformar as imagens individualmente, o programa remonta o batch original.
            por meio de .batch(self.batch_size)
            
            OBSERVAÇÃO: .prefetch(AUTOTUNE) permite que o TensorFlow processe o batch N+1 enquanto 
            o modelo treina no batch N
        '''
        train_ds = tf.data.Dataset.from_generator(
            lambda: train_raw,
            output_signature=(
                tf.TensorSpec(shape=(None, self.img_size, self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, train_raw.num_classes), dtype=tf.float32),
            )
        ).unbatch().map(
            lambda x, y: (self.augment(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        ).batch(self.batch_size).prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: val_raw,
            output_signature=(
                tf.TensorSpec(shape=(None, self.img_size, self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, val_raw.num_classes), dtype=tf.float32),
            )
        ).unbatch().batch(self.batch_size).prefetch(AUTOTUNE)

        return train_ds, val_ds


'''
IMPLEMENTAÇÃO ANTIGA

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
'''
