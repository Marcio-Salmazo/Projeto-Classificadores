import os
import sys
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Model:

    @staticmethod
    def resource_path(relative_path):
        """ Retorna o caminho absoluto para o arquivo, compatível com PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)

    @staticmethod
    def open_directory():
        """
            O tkinter é utilizado para exibir janela do explorer a fim de selecionar a pasta contendo o Dataset.
                * root = tk.Tk() - instância do tkinter
                * root.withdraw() -  Oculta a janela principal (para exibir apenas o pop-up)
                * filedialog.askdirectory(title="") - Abre a janela de seleção de pastas e retorna o caminho escolhido
        """
        root = tk.Tk()
        root.withdraw()

        path = filedialog.askdirectory(title="Selecione a pasta desejada")
        # Se o usuário cancelar ou fechar a janela, path será ""
        if not path:
            return None

        return path

    @staticmethod
    def load_data(dataset_path, img_size=(128, 128), batch_size=32, val_split=0.3):
        """
        O ImageDataGenerator é uma classe do Keras (tensorflow.keras.preprocessing.image) que facilita o
        pré-processamento de imagens para redes neurais. Ele permite carregar imagens de um diretório e aplicar
        transformações como normalização, rotação, espelhamento, aumento de dados (data augmentation), entre outras

        obs: A divisão entre treino e validação não é aleatória por padrão no ImageDataGenerator quando
        usamos o parâmetro validation_split. A separação é feita de forma ordenada, baseada na ordem
        dos arquivos dentro das pastas.
        """

        # Instância de uma objeto do ImageDataGenerator, definindo como parâmetros operações para o pré processamento
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,  # Normalização do valor dos pixels das imagens (Faixa de 0 à 1)
            validation_split=val_split  # Define a divisão dos dados (imagens )entre treino e validação
        )

        """
        O flow_from_directory do ImageDataGenerator é utilizado para carregar automaticamente as imagens de um 
        diretório organizado em subpastas, onde cada subpasta representa uma classe. Além disso, são aplicadas 
        as transformações definidas no momento da instância (como normalização) e as prepara em batches para 
        serem usadas no treinamento da CNN.
        """
        # Define a geração do grupo de treinamento, utilizando a instância do ImageData generator
        train_generator = datagen.flow_from_directory(

            dataset_path,  # Indica o caminho do diretório onde estão armazenadas as imagens
            target_size=img_size,  # Redimensiona o tamanho das imagens que serão carregadas na CNN
            batch_size=batch_size,  # Define o número de imagens por lote que será carregado em cada iteração
            class_mode='categorical',  # Define as classes. 'categorical' indica que as saídas serão vetores one-hot.
            subset='training'  # Aqui indicamos se queremos carregar os dados definidos em validation_split
            # obs: Os dados definidos em validation_split podem ser descritos po 'training' ou 'validation'
        )

        # Define a geração do grupo de treinamento, utilizando a instância do ImageData generator
        val_generator = datagen.flow_from_directory(

            dataset_path,  # Indica o caminho do diretório onde estão armazenadas as imagens
            target_size=img_size,  # Redimensiona o tamanho das imagens que serão carregadas na CNN
            batch_size=batch_size,  # Define o número de imagens por lote que será carregado em cada iteração
            class_mode='categorical',  # Define as classes. 'categorical' indica que as saídas serão vetores one-hot.
            subset='validation'  # Aqui indicamos se queremos carregar os dados definidos em validation_split
        )

        log_training_samples = (f"Foram encontradas {train_generator.samples} imagens "
                                f"pertencentes a {train_generator.num_classes} classes distintas para o treinamento")
        log_validation_samples = (f"Foram encontradas {val_generator.samples} imagens "
                                  f"pertencentes a {val_generator.num_classes} classes distintas para a validação")
        log_indexes = f"Classes identificadas: {train_generator.class_indices}"

        return (train_generator,
                val_generator,
                log_training_samples,
                log_validation_samples,
                log_indexes,
                train_generator.num_classes)

    @staticmethod
    def log_directory_manager(logName):
        # Criar o diretório "logs/fit/" caso não exista
        log_dir = "logs/fit/"
        os.makedirs(log_dir, exist_ok=True)  # Garante que o diretório existe

        # Criar um subdiretório único para cada execução
        run_id = logName + '_run_' + str(len(os.listdir(log_dir)) + 1)
        full_log_path = os.path.join(log_dir, run_id)

        return full_log_path
