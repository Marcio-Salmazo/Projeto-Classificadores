from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton,
    QLabel, QGroupBox, QApplication, QTextEdit, QDialog, QMessageBox
)

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap

import ApplicationModel
import ModelCreator
import TrainerThread
from ApplicationModel import Model
from Parameters import NetworkLogName, VitParameters, InputDataParameters
import subprocess
import webbrowser
import time

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class Interface(QWidget):
    log_signal = pyqtSignal(str)  # sinal para enviar mensagens de log ao PyQt

    def __init__(self):

        super().__init__()

        self.model = Model()
        self.setWindowTitle("Interface de treinamento unificado")
        self.selected_model = None
        self.setWindowIcon(QIcon(self.model.resource_path("figures/figNN")))

        # Layout principal (horizontal), responsável por separar a área que vai mostrar o log de treinamento
        # da seção responsável por conter as funções do programa e configurações da rede
        main_layout = QHBoxLayout(self)

        # Define a área onde o status de treinamento será exibido
        # O QTextEditÉ uma área de texto multilinha que permite a visualização de várias
        # linhas de texto, rolagem automática e até textos formatados (negrito, cores).
        # Ao longo do treinamento da rede, novas mensagens são inseridas aqui.
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)  # somente leitura
        self.log_area.setPlaceholderText("Status do treinamento aparecerá aqui...")  # Texto de placeholder
        main_layout.addWidget(self.log_area, stretch=3)  # Adiciona o widget no layout principal

        # ---- Layout vertical para as funções ----
        functions_layout = QVBoxLayout()

        # ---- Seleção de Arquitetura ----
        self.radio_vit = QRadioButton("Vision Transformer (ViT)")
        self.radio_resnet = QRadioButton("Res_Net50 (CNN)")

        self.confirm_btn = QPushButton("Selecionar Arquitetura")
        self.confirm_btn.clicked.connect(self.confirm_model_choice)

        self.reset_btn = QPushButton("Resetar Aplicação")
        self.reset_btn.clicked.connect(self.reset_app)

        self.btn_exit = QPushButton("Sair")
        self.btn_exit.clicked.connect(self.exit_program)

        functions_layout.addWidget(QLabel("Escolha a arquitetura:"))
        functions_layout.addWidget(self.radio_vit)
        functions_layout.addWidget(self.radio_resnet)
        functions_layout.addWidget(self.confirm_btn)
        functions_layout.addWidget(self.reset_btn)
        functions_layout.addWidget(self.btn_exit)
        functions_layout.addStretch()  # empurra os botões para cima

        # --------------------------------------------------------------------------------------------------------------
        # ---- Área ViT ----
        self.vit_group = QGroupBox("Opções para ViT")
        self.vit_group.setVisible(False)

        vit_layout = QVBoxLayout()
        self.btn_vit_build = QPushButton("Construir Modelo ViT")
        vit_layout.addWidget(self.btn_vit_build)
        self.btn_vit_build.clicked.connect(self.build_vit)

        (self.common_layout_vit,
         self.btn_select_folder_vit,
         self.btn_train_vit,
         self.btn_tensorboard_vit,
         self.btn_metrics_vit) = self.create_common_buttons_layout()

        vit_layout.addLayout(self.common_layout_vit)
        self.vit_group.setLayout(vit_layout)

        self.btn_select_folder_vit.clicked.connect(self.select_data)
        self.btn_train_vit.clicked.connect(self.train_network)
        self.btn_tensorboard_vit.clicked.connect(self.open_logs)
        self.btn_metrics_vit.clicked.connect(lambda: self.confusion_metrics('vit'))

        # Definindo o status inicial dos botões
        self.btn_select_folder_vit.setEnabled(True)
        self.btn_vit_build.setEnabled(False)
        self.btn_train_vit.setEnabled(False)
        self.btn_metrics_vit.setEnabled(True)
        self.btn_tensorboard_vit.setEnabled(True)

        # --------------------------------------------------------------------------------------------------------------
        # ---- Área ResNet ----
        self.resnet_group = QGroupBox("Opções para ResNet")
        self.resnet_group.setVisible(False)

        resnet_layout = QVBoxLayout()
        self.btn_resnet_build = QPushButton("Construir Modelo ResNet")
        resnet_layout.addWidget(self.btn_resnet_build)
        self.btn_resnet_build.clicked.connect(self.build_resnet)

        (self.common_layout_resnet,
         self.btn_select_folder_resnet,
         self.btn_train_resnet,
         self.btn_tensorboard_resnet,
         self.btn_metrics_resnet) = self.create_common_buttons_layout()

        resnet_layout.addLayout(self.common_layout_resnet)
        self.resnet_group.setLayout(resnet_layout)

        self.btn_select_folder_resnet.clicked.connect(self.select_data)
        self.btn_train_resnet.clicked.connect(self.train_network)
        self.btn_tensorboard_resnet.clicked.connect(self.open_logs)
        self.btn_metrics_resnet.clicked.connect(lambda: self.confusion_metrics('resnet'))

        # Definindo o status inicial dos botões
        self.btn_select_folder_resnet.setEnabled(True)
        self.btn_resnet_build.setEnabled(False)
        self.btn_train_resnet.setEnabled(False)
        self.btn_metrics_resnet.setEnabled(True)
        self.btn_tensorboard_resnet.setEnabled(True)

        # --------------------------------------------------------------------------------------------------------------

        # ---- Inserção de label para inserir a logo da UFU ----
        self.logo_label = QLabel()
        pixmap = QPixmap(self.model.resource_path("figures/fig_ufu.png"))
        pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)  # Centraliza a imagem
        self.logo_label.setContentsMargins(0, 30, 0, 0)  # Padding para espaçar a exibição da imagem
        functions_layout.addWidget(self.logo_label)

        # ---- Inserção de label para definir a versão do software ----
        # Seguindo o padrão de Versionamento Semântico -> MAJOR.MINOR.PATCH-SUFIX
        self.version_label = QLabel("Ver. 0.5.0", self)
        self.version_label.setAlignment(Qt.AlignCenter)
        functions_layout.addWidget(self.version_label)

        # ---- Layout Final ----
        main_layout.addLayout(functions_layout)
        main_layout.addWidget(self.vit_group)
        main_layout.addWidget(self.resnet_group)

        self.setLayout(main_layout)

        # --------------------------------------------------------------------------------------------------------------

        # ---- Atributos ----
        # Atributos referentes ao carregamento dos dados
        self.image_generator_input_size = None
        self.image_generator_batch_size = None
        self.image_generator_split = None
        self.train_data = None
        self.val_data = None

        # Atributos compartilhados entre as redes
        self.network_input_size = None
        self.dataset_classes = None
        self.trainer_thread = None
        self.fileName_weights = None

        # Atributos específicos dos parâmetros da ViT
        self.vit_model = None
        self.vit = None
        self.patch_size = None
        self.projection_dim = None
        self.transformer_layers = None
        self.num_heads = None
        self.mlp_units = None

        # Atributos específicos dos parâmetros da ResNet
        self.resnet = None

        # Atributos específicos para a predição de resultados e obtenção de métricas
        self.y_pred = None
        self.y_pred_classes = None

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ---- Funções ----

    def create_common_buttons_layout(self):

        layout = QVBoxLayout()

        btn_select_folder = QPushButton("Selecionar Dataset")
        btn_train = QPushButton("Iniciar Treinamento")
        btn_tensorboard = QPushButton("Abrir Tensorboard")
        btn_metrics = QPushButton("Métricas de confusão")

        layout.addWidget(btn_select_folder)
        layout.addWidget(btn_train)
        layout.addWidget(btn_tensorboard)
        layout.addWidget(btn_metrics)
        layout.addStretch()

        # retorna também os botões para controlar estado depois
        return layout, btn_select_folder, btn_train, btn_tensorboard, btn_metrics

    def confirm_model_choice(self):
        if self.radio_vit.isChecked():
            self.selected_model = "vit"
            self.vit_group.setVisible(True)
            self.resnet_group.setVisible(False)
            self.reset_btn.setVisible(True)

        elif self.radio_resnet.isChecked():
            self.selected_model = "resnet"
            self.resnet_group.setVisible(True)
            self.vit_group.setVisible(False)
            self.reset_btn.setVisible(True)

        else:
            return  # nenhum selecionado

        # desabilitar mudança
        self.radio_vit.setEnabled(False)
        self.radio_resnet.setEnabled(False)
        self.confirm_btn.setEnabled(False)
        self.reset_btn.setVisible(True)

    def reset_app(self):
        # limpar seleção de modelo
        self.selected_model = None
        self.log_area.clear()

        # resetar radio buttons
        self.radio_vit.setEnabled(True)
        self.radio_resnet.setEnabled(True)
        self.radio_vit.setChecked(False)
        self.radio_resnet.setChecked(False)
        self.confirm_btn.setEnabled(True)

        # resetar visibilidade dos botões
        self.btn_select_folder_vit.setEnabled(True)
        self.btn_vit_build.setEnabled(False)
        self.btn_train_vit.setEnabled(False)
        self.btn_metrics_vit.setEnabled(True)
        self.btn_tensorboard_vit.setEnabled(True)

        self.btn_select_folder_resnet.setEnabled(True)
        self.btn_resnet_build.setEnabled(False)
        self.btn_train_resnet.setEnabled(False)
        self.btn_metrics_resnet.setEnabled(True)
        self.btn_tensorboard_resnet.setEnabled(True)

        # esconder menus
        self.vit_group.setVisible(False)
        self.resnet_group.setVisible(False)

        # Atributos referentes ao carregamento dos dados
        self.image_generator_input_size = None
        self.image_generator_batch_size = None
        self.image_generator_split = None
        self.train_data = None
        self.val_data = None

        # Atributos compartilhados entre as redes
        self.network_input_size = None
        self.dataset_classes = None
        self.trainer_thread = None
        self.fileName_weights = None

        # Atributos específicos dos parâmetros da ViT
        self.vit_model = None
        self.vit = None
        self.patch_size = None
        self.projection_dim = None
        self.transformer_layers = None
        self.num_heads = None
        self.mlp_units = None

        # Atributos específicos dos parâmetros da ResNet
        self.resnet = None

    def add_log_message(self, msg: str):
        self.log_area.append(msg)

    def select_data(self):

        path = self.model.open_directory()
        if path is None:
            QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        dialog = InputDataParameters()
        if dialog.exec_() == QDialog.Accepted:
            self.image_generator_input_size = dialog.input_size
            self.image_generator_batch_size = dialog.batch_size
            self.image_generator_split = dialog.split

        else:
            QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        self.add_log_message(f'Input Size escolhido: {self.image_generator_input_size}')
        self.add_log_message(f'Batch Size escolhido: {self.image_generator_batch_size}')
        self.add_log_message(f'Taxa de divisão escolhida: {self.image_generator_split}')
        self.add_log_message('--------------------------------------------------------')

        model = ApplicationModel.Model()
        image_size = (self.image_generator_input_size, self.image_generator_input_size)
        print('chegou aqui', image_size)

        (self.train_data, self.val_data, log_training_samples, log_validation_samples,
         log_indexes, self.dataset_classes) = (model.load_data(path, image_size,
                                                               self.image_generator_batch_size,
                                                               self.image_generator_split))

        self.add_log_message(log_training_samples)
        self.add_log_message(log_validation_samples)
        self.add_log_message(log_indexes)
        self.add_log_message('--------------------------------------------------------')

        # Definindo o status do botão de construção da rede
        self.btn_resnet_build.setEnabled(True)
        self.btn_vit_build.setEnabled(True)

    def build_resnet(self):

        if self.image_generator_input_size is None or self.dataset_classes is None:
            QMessageBox.warning(self, "Erro",
                                "Input Size ou quantidade de classes não foram definidas, recarregue o dataset")
            return

        # O input size da rede tem o mesmo formato dos dados gerados, com 3 camadas (normal de imagens sem tratamento)
        cnn_input_size = (self.image_generator_input_size, self.image_generator_input_size, 3)
        cnn_model = ModelCreator.Resnet50(cnn_input_size, self.dataset_classes)
        self.resnet = cnn_model.resnet_classifier()

        if self.resnet:
            self.add_log_message(f'Rede construída: {self.resnet}')
            self.add_log_message(f'Rede compilada com sucesso!')
            self.add_log_message(f'Quantidade de classes encontradas: {self.dataset_classes}')
        self.add_log_message('--------------------------------------------------------')

        # Definindo o status do botão de construção da rede
        self.btn_train_resnet.setEnabled(True)

    def build_vit(self):

        if self.image_generator_input_size is None or self.dataset_classes is None:
            QMessageBox.warning(self, "Erro",
                                "Input Size ou quantidade de classes não foram definidas, recarregue o dataset")
            return

        dialog = VitParameters()
        if dialog.exec_() == QDialog.Accepted:
            self.patch_size = dialog.patch_size
            self.projection_dim = dialog.projection_dim
            self.transformer_layers = dialog.transformer_layers
            self.num_heads = dialog.num_heads
            self.mlp_units = dialog.mlp_units
        else:
            QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        self.add_log_message(f'Tamanho dos patches: {self.patch_size}')
        self.add_log_message(f'Tamanho do vetor para projeção: {self.projection_dim}')
        self.add_log_message(f'Camadas de transformer: {self.transformer_layers}')
        self.add_log_message(f'Numero de cabeças de atenção: {self.num_heads}')
        self.add_log_message(f'Unidades do multilayer perceptron: {self.mlp_units}')
        self.add_log_message('--------------------------------------------------------')

        # O input size da rede tem o mesmo formato dos dados gerados, com 3 camadas (normal de imagens sem tratamento)
        vit_input_size = (self.image_generator_input_size, self.image_generator_input_size, 3)
        num_patches = (self.image_generator_input_size // self.patch_size) ** 2

        print("o código chega aqui")
        print("vit_input_size", vit_input_size)
        print("num_patches", num_patches)

        self.vit = ModelCreator.VisionTransformer(vit_input_size,
                                                  self.patch_size,
                                                  num_patches,
                                                  self.projection_dim,
                                                  self.transformer_layers,
                                                  self.num_heads,
                                                  self.mlp_units,
                                                  self.dataset_classes)

        self.vit_model = self.vit.vit_classifier()

        if self.vit_model:
            self.add_log_message(f'Rede construída: {self.vit_model}')
            self.add_log_message(f'Rede compilada com sucesso!')
            self.add_log_message(f'Quantidade de classes encontradas: {self.dataset_classes}')
        self.add_log_message('--------------------------------------------------------')

        # Definindo o status do botão de construção da rede
        self.btn_train_vit.setEnabled(True)

    def train_network(self):
        if hasattr(self, "trainer_thread") and self.trainer_thread.isRunning():
            QMessageBox.warning(self, "Treinamento em andamento",
                                "Aguarde o treinamento terminar antes de iniciar outro.")
            return

        if self.train_data is None or self.val_data is None:
            QMessageBox.warning(self, "Erro", "Dataset inválido para iniciar treinamento")
            return

        dialog = NetworkLogName()
        if dialog.exec_() == QDialog.Accepted:
            fileName = dialog.log_name
            self.fileName_weights = fileName
            epochs = dialog.epochs
        else:
            QMessageBox.warning(self, "Erro de valor", "Seleção de nome dos logs cancelada pelo usuário.")
            return  # encerra a função sem travar

        if self.resnet is not None:
            # cria a thread de treinamento
            self.trainer_thread = TrainerThread.ResNet_TrainerThread(self.resnet, self.train_data, self.val_data,
                                                                     epochs, fileName)
            self.trainer_thread.log_signal.connect(self.add_log_message)  # conecta o log ao QTextEdit
            self.trainer_thread.training_finished.connect(self.save_weights)  # conecta flag
            self.trainer_thread.start()

        elif self.vit_model is not None:
            # cria a thread de treinamento
            self.trainer_thread = TrainerThread.ViT_TrainerThread(self.vit_model, self.train_data, self.val_data,
                                                                  epochs, fileName)
            self.trainer_thread.log_signal.connect(self.add_log_message)  # conecta o log ao QTextEdit
            self.trainer_thread.training_finished.connect(self.save_weights)  # conecta flag
            self.trainer_thread.start()

    def save_weights(self, success: bool):
        if success:
            self.add_log_message("Treinamento concluído. Salvando pesos:")

            if self.resnet is not None:
                self.resnet.save(f"{self.fileName_weights}_weights.h5")
                self.add_log_message(f"Pesos de treinamento salvos como {self.fileName_weights}_weights.h5")
                self.trainer_thread = None

            elif self.vit_model is not None:
                self.vit_model.save(f"{self.fileName_weights}_weights.h5")
                self.add_log_message(f"Pesos de treinamento salvos como {self.fileName_weights}_weights.h5")
                self.trainer_thread = None
        else:
            QMessageBox.warning(self, "Erro de valor", "Treinamento não foi iniciado ou concluído")
            return  # encerra a função sem travar

    def open_logs(self):

        # CONSERTAR A INSTÂNCIA DO TENSORBOARD
        log_path = self.model.open_directory()
        if not log_path:
            QMessageBox.warning(self, "Erro", "Caminho para o diretório não foi definido")
            return

        try:
            process = subprocess.Popen(
                ["tensorboard", f"--logdir=\"{log_path}\"", "--port=6006"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(3)  # dá tempo do servidor iniciar
            webbrowser.open("http://localhost:6006")
            self.add_log_message(f'Tensorboard inicializado no caminho: {self.log_path}')
            self.add_log_message('--------------------------------------------------------')

        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao inicializar o tensorboard. Tente novamente')

    def confusion_metrics(self, model_name):

        # Desabilitar demais funções, exigindo um Reset após a utilização
        # Fiz isso para evitar conflitos ao construir modelos e na seleção
        # do dataset, para obter a validação.
        self.btn_select_folder_vit.setEnabled(False)
        self.btn_vit_build.setEnabled(False)
        self.btn_train_vit.setEnabled(False)
        self.btn_select_folder_resnet.setEnabled(False)
        self.btn_resnet_build.setEnabled(False)
        self.btn_train_resnet.setEnabled(False)

        # Resetando qualquer rede préviamente contruída
        self.vit_model = None
        self.resnet = None

        # Resetando qualquer valor prévio de y_pred e y_pred_classes
        self.y_pred = None
        self.y_pred_classes = None

        # Recriando val_generator exatamente como no treinamento, é importante ressaltar que o ImageDataGenerator
        # com validation_split gera automaticamente uma divisão consistente entre treino e validação,
        # sem precisar salvar nada manualmente.
        QMessageBox.warning(self, "Atenção", "É necessário selecionar o dataset com os mesmos parâmetros defindos em"
                                             " seu treinamento (de acordo com os pesos que serão selecionados), "
                                             "principalmente o split_val!")
        self.select_data()

        # Obter labels verdadeiros
        y_true = self.val_data.classes
        # Nome das classes
        class_names = list(self.val_data.class_indices.keys())

        # A função select_data() ativa os botões de construção do modelo, portanto precisamos desativa-los novamente
        self.btn_resnet_build.setEnabled(True)
        self.btn_vit_build.setEnabled(True)

        # Indicar o caminho dos pesos e do dataset
        QMessageBox.warning(self, "Atenção", "Selecionar os pesos para carregar no modelo. É importante destacar que "
                                             "os pesos selecionados devem ser pertencentes à arquitetura escolha pelo "
                                             "radio button.")
        weights_path = self.model.open_directory()

        if model_name == 'vit':
            self.build_vit()
            if self.vit is not None:
                self.vit_model.load_weights(weights_path)

                # Obter previsões
                self.y_pred = self.vit_model.predict(self.val_data)
                self.y_pred_classes = np.argmax(self.y_pred, axis=1)

            else:
                QMessageBox.warning(self, "Erro", "Rede não construída (Definida como NONE)")
                return

        elif model_name == 'resnet':
            self.build_resnet()
            if self.resnet is not None:
                self.resnet.load_weights(weights_path)

                # Obter previsões
                self.y_pred = self.resnet.predict(self.val_data)
                self.y_pred_classes = np.argmax(self.y_pred, axis=1)

            else:
                QMessageBox.warning(self, "Erro", "Rede não construída (Definida como NONE)")
                return

        # Matriz de confusão
        cm = confusion_matrix(y_true, self.y_pred_classes)
        print("\nMatriz de Confusão:")
        print(cm)

        # Report por classe (TP, FP, TN, FN estão implícitos)
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, self.y_pred_classes, target_names=class_names))

        return cm

    def exit_program(self):

        self.close()  # Fecha a janela principal
        QApplication.quit()  # Finaliza o loop da aplicação corretamente
