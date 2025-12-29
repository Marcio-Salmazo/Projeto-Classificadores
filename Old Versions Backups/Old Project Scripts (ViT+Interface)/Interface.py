from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton,
    QLabel, QGroupBox, QApplication, QTextEdit, QDialog, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap

from Parameters import VitParameters, InputDataParameters
import Utils
import json
import os
import subprocess

class Interface(QWidget):
    log_signal = pyqtSignal(str)  # sinal para enviar mensagens de log ao PyQt

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Interface de treinamento unificado")
        self.selected_model = None
        self.setWindowIcon(QIcon(Utils.resource_path("Figures\\figNN.png")))

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

        # --------------------------------------------------------------------------------------------------------------
        #                             Área de layout inicial para a seleção da arquitetura
        # --------------------------------------------------------------------------------------------------------------
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
        #                                   Área do Layout dedicado para a ViT
        # --------------------------------------------------------------------------------------------------------------
        self.vit_group = QGroupBox("Opções para ViT")
        self.vit_group.setVisible(False)

        vit_layout = QVBoxLayout()

        (self.common_layout_vit,
         self.btn_select_folder_vit,
         self.btn_train_vit) = self.create_common_buttons_layout()

        vit_layout.addLayout(self.common_layout_vit)
        self.vit_group.setLayout(vit_layout)

        self.btn_select_folder_vit.clicked.connect(self.select_data)
        self.btn_train_vit.clicked.connect(self.build_and_run_vit)

        # Definindo o status inicial dos botões
        self.btn_select_folder_vit.setEnabled(True)
        self.btn_train_vit.setEnabled(False)

        # --------------------------------------------------------------------------------------------------------------
        #                                   Área do Layout dedicado para a ResNet
        # --------------------------------------------------------------------------------------------------------------
        self.resnet_group = QGroupBox("Opções para ResNet")
        self.resnet_group.setVisible(False)

        resnet_layout = QVBoxLayout()

        (self.common_layout_resnet,
         self.btn_select_folder_resnet,
         self.btn_train_resnet) = self.create_common_buttons_layout()

        resnet_layout.addLayout(self.common_layout_resnet)
        self.resnet_group.setLayout(resnet_layout)

        self.btn_select_folder_resnet.clicked.connect(self.select_data)
        self.btn_train_resnet.clicked.connect(self.build_and_run_resnet)

        # Definindo o status inicial dos botões
        self.btn_select_folder_resnet.setEnabled(True)

        # --------------------------------------------------------------------------------------------------------------
        #                                  Área do Layout final (integração das demais)
        # --------------------------------------------------------------------------------------------------------------

        # ---- Inserção de label para inserir a logo da UFU ----
        self.logo_label = QLabel()
        pixmap = QPixmap(Utils.resource_path("Figures\\fig_ufu.png"))
        pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)  # Centraliza a imagem
        self.logo_label.setContentsMargins(0, 30, 0, 0)  # Padding para espaçar a exibição da imagem
        functions_layout.addWidget(self.logo_label)

        # ---- Inserção de label para definir a versão do software ----
        # Seguindo o padrão de Versionamento Semântico -> MAJOR.MINOR.PATCH-SUFIX
        self.version_label = QLabel("Ver. 0.9.0", self)
        self.version_label.setAlignment(Qt.AlignCenter)
        functions_layout.addWidget(self.version_label)

        # ---- Layout Final ----
        main_layout.addLayout(functions_layout)
        main_layout.addWidget(self.vit_group)
        main_layout.addWidget(self.resnet_group)

        self.setLayout(main_layout)

        # --------------------------------------------------------------------------------------------------------------
        #                                             Área de atributos
        # --------------------------------------------------------------------------------------------------------------

        # Atributos referentes ao carregamento dos dados
        self.image_input_size = None
        self.image_batch_size = None
        self.dataset_split = None
        self.dataset_path = None

        self.train_data = None
        self.val_data = None
        self.steps_train = None
        self.steps_val = None

        # Atributos compartilhados entre as redes
        self.network_input_size = None
        self.dataset_classes = None
        self.trainer_thread = None
        self.fileName_weights = None

        # Atributos específicos dos parâmetros da ViT
        self.vit_model = None
        self.patch_size = None
        self.hidden_size = None
        self.transformer_layers = None
        self.num_heads = None
        self.mlp_units = None
        self.batch_size_vit = None
        self.total_steps = None
        self.warmup_steps = None
        self.base_lr = None
        self.mode = None
        self.weights_path = None

        # Atributos específicos dos parâmetros da ResNet

        # Atributos específicos para a predição de resultados e obtenção de métricas
        self.y_pred = None
        self.y_pred_classes = None

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                   Funções
    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    # ======================================================================================================================
    # FUNÇÃO AUXILIAR PARA CRIAR LAYOUTS DE BOTÕES COMUNS

    @staticmethod
    def create_common_buttons_layout():

        layout = QVBoxLayout()

        btn_select_folder = QPushButton("Selecionar Dataset")
        btn_train = QPushButton("Construir rede e iniciar Treinamento")

        layout.addWidget(btn_select_folder)
        layout.addWidget(btn_train)
        layout.addStretch()

        # retorna também os botões para controlar estado depois
        return layout, btn_select_folder, btn_train

    # ======================================================================================================================
    # FUNÇÃO AUXILIAR PARA GERENCIAR OS MENUS CONFORME ESCOLHA DO USUÁRIO

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

    # ======================================================================================================================
    # FUNÇÃO QUE RESETA TUDO QUE FOI FEITO PELO USUÁRIO

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
        self.btn_train_vit.setEnabled(False)
        self.btn_select_folder_resnet.setEnabled(True)
        self.btn_train_resnet.setEnabled(False)

        # esconder menus
        self.vit_group.setVisible(False)
        self.resnet_group.setVisible(False)

        # Atributos referentes ao carregamento dos dados
        self.image_input_size = None
        self.image_batch_size = None
        self.dataset_split = None
        self.dataset_path = None

        self.train_data = None
        self.val_data = None
        self.steps_train = None
        self.steps_val = None

        # Atributos compartilhados entre as redes
        self.network_input_size = None
        self.dataset_classes = None
        self.trainer_thread = None
        self.fileName_weights = None

        # Atributos específicos dos parâmetros da ViT
        self.vit_model = None
        self.patch_size = None
        self.hidden_size = None
        self.transformer_layers = None
        self.num_heads = None
        self.mlp_units = None
        self.batch_size_vit = None
        self.total_steps = None
        self.warmup_steps = None
        self.base_lr = None
        self.mode = None
        self.weights_path = None

        # Atributos específicos dos parâmetros da ResNet

        # Atributos específicos para a predição de resultados e obtenção de métricas
        self.y_pred = None
        self.y_pred_classes = None

    # ======================================================================================================================
    # FUNÇÃO AUXILIAR PARA INSERIR MENSAGENS DE LOG À INTERFACE

    def add_log_message(self, msg: str):
        self.log_area.append(msg)

    # ======================================================================================================================
    # FUNÇÃO AUXILIAR QUE ATUA PARA SELECIONAR A BASE DE DADOS, CRIANDO UM JSON DE CONFIGURAÇÕES PARA A ARQUITETURA
    # A UTILIZAÇÃO DE UM ENTRY-POINT SE FAZ NECESSÁRIA EM RAZÃO DOS DIFERENTES AMBIENTES VIRTUAIS CRIADOS

    def select_data(self):
        path = Utils.open_directory()
        if not path:
            QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        self.dataset_path = path
        dialog = InputDataParameters()
        if dialog.exec_() == QDialog.Accepted:

            self.image_input_size = dialog.input_size
            self.image_batch_size = dialog.batch_size
            self.dataset_split = dialog.split
        else:
            QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        self.add_log_message(f"CAMINHO DO DATASET E CONFIGURAÇÕES FORAM DEFINIDOS")
        self.add_log_message('---------------------------------------------------------------')
        self.add_log_message(f"O CARREGAMENTO DA BASE SERÁ FEITO AO INICIAR O TREINAMENTO")
        self.add_log_message("---------------------------------------------------------------")
        self.add_log_message(f"Input Size: {self.image_input_size}")
        self.add_log_message(f"Batch Size: {self.image_batch_size}")
        self.add_log_message(f"Divisão da base para teste: {self.dataset_split}")
        self.add_log_message("---------------------------------------------------------------")

        self.btn_train_vit.setEnabled(True)

    # ======================================================================================================================
    # FUNÇÃO DE CRIAÇÃO E TREINAMENTO DA REDE, AOS MOLDES DA ARQUITETURA RESNET-50

    # A RESNET SERÁ INTEGRADA APÓS O PROCESSO DE VALIDAÇÃO
    def build_and_run_resnet(self):
        return

    # ======================================================================================================================
    # FUNÇÃO DE CRIAÇÃO E TREINAMENTO DA REDE, AOS MOLDES DA ARQUITETURA VISION TRANSFORMER

    def build_and_run_vit(self):

        dialog = VitParameters()
        if dialog.exec_() == QDialog.Accepted:
            self.patch_size = dialog.patch_size
            self.hidden_size = dialog.hidden_size
            self.transformer_layers = dialog.transformer_layers
            self.num_heads = dialog.num_heads
            self.mlp_units = dialog.mlp_units
            # self.batch_size_vit = dialog.batch_size_vit
            self.total_steps = dialog.total_steps
            self.warmup_steps = dialog.warmup_steps
            self.base_lr = dialog.base_lr
            self.mode = dialog.mode

        else:
            QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        self.add_log_message(f'CONFIGURAÇÕES ESCOLHIDAS PARA TREINAMENTO:')
        self.add_log_message(' ')
        self.add_log_message(f'Tamanho dos patches: {self.patch_size}')
        self.add_log_message(f'Tamanho do vetor para projeção: {self.hidden_size}')
        self.add_log_message(f'Camadas de transformer: {self.transformer_layers}')
        self.add_log_message(f'Numero de cabeças de atenção: {self.num_heads}')
        self.add_log_message(f'Tamanho do Batch: {self.batch_size_vit}')
        self.add_log_message(f'Steps totais: {self.total_steps}')
        self.add_log_message(f'Steps de aquecimento: {self.warmup_steps}')
        self.add_log_message(f'Learning Rate: {self.base_lr}')
        self.add_log_message(f'Modo de treinamento: {self.mode}')
        self.add_log_message('---------------------------------------------------------------')

        if self.mode == 'finetune':
            QMessageBox.warning(self, "Alerta", "Selecionar o arquivo de pesos à serem carregados")
            self.weights_path = Utils.open_weight_file()
            if self.weights_path is None:
                QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
                return  # encerra a função sem travar

        config = {
            "dataset_path": self.dataset_path,
            "input_size": self.image_input_size,
            "image_batch_size": self.image_batch_size,
            "split": self.dataset_split,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "transformer_layers": self.transformer_layers,
            "num_heads": self.num_heads,
            "mlp_units": self.mlp_units,
            # "batch_size": self.batch_size_vit,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "mode": self.mode,
            "weights": self.weights_path
        }

        config_dir = Utils.resource_path("MainProject\\VisionTransformers")
        config_path = f"{config_dir}\\vit_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        self.add_log_message("CONFIGURAÇÃO SALVA")
        self.add_log_message("---------------------------------------------------------------")
        self.add_log_message("INICIANDO ETAPAS DE TREINAMENTO")
        self.add_log_message("---------------------------------------------------------------")

        self.run_vit_subprocess(config_path)

    # ======================================================================================================================
    # FUNÇÃO RESPONSÁVEL PELA GESTÃO DO SUB-PROCESSO DE TREINAMENTO ESPECÍFICO DA ViT

    def run_vit_subprocess(self, config_path):

        python_exec = os.path.abspath(".vit_venv/Scripts/python.exe")
        script_path = os.path.abspath("MainProject/VisionTransformers/ViT_EntryPoint.py")

        ''' 
        self.add_log_message("Iniciando subprocesso de treinamento ViT...")
        self.add_log_message(f"Python: {python_exec}")
        self.add_log_message(f"Script: {script_path}")
        self.add_log_message("---------------------------------------------------------------")
        '''

        print("Iniciando subprocesso de treinamento ViT...")
        print(f"Python: {python_exec}")
        print(f"Script: {script_path}")
        print("---------------------------------------------------------------")

        process = subprocess.Popen(
            [python_exec, script_path, config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )

        """
        # Leitura segura do stdout (não bloqueante para a UI)
        while True:
            line = process.stdout.readline()
            if not line:
                break

            self.add_log_message(line.rstrip())

        process.stdout.close()
        return_code = process.wait()

        self.add_log_message("---------------------------------------------------------------")
        self.add_log_message(f"Processo finalizado com código: {return_code}")
        """

    # ======================================================================================================================
    # FUNÇÃO RESPONSÁVEL POR ENCERRAR A EXECUÇÃO DO PROGRAMA

    def exit_program(self):

        self.close()  # Fecha a janela principal
        QApplication.quit()  # Finaliza o loop da aplicação corretamente
