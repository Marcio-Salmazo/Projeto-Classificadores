from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox

class InputDataParameters(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        # Cria uma nova janela para definição de parâmetros do tratamento de
        # dados pelo Image Data Generator
        self.setWindowTitle("Configurar Parâmetros de Dataset")

        self.input_size = None
        self.batch_size = None
        self.split = None

        # Layout principal, organizando verticalmente os widgets na janela
        layout = QVBoxLayout()

        # Organização dos elementos para tornar possível a entrada de valores
        # para a variável Input_Size
        h_layout = QHBoxLayout() # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel("Input Size:")) # Adição do widget label ao layout horizontal
        self.input_size_edit = QLineEdit() # Area de entrada de valores
        self.input_size_edit.setText("224")
        self.input_size_edit.setPlaceholderText("Ex: 224") # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.input_size_edit) # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout) # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Batch Size (Análogo ao bloco de código referente ao Input_Size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_edit = QLineEdit()
        self.batch_size_edit.setText("32")
        self.batch_size_edit.setPlaceholderText("Ex: 32")
        h_layout.addWidget(self.batch_size_edit)
        layout.addLayout(h_layout)

        # Split (treino/validação) (Análogo ao bloco de código referente ao Input_Size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Split (treino/validação):"))
        self.split_edit = QLineEdit()
        self.split_edit.setText("0.2")
        self.split_edit.setPlaceholderText("Ex: 0.2 (Validação)")
        h_layout.addWidget(self.split_edit)
        layout.addLayout(h_layout)

        # Botões OK e Cancel (Análogo à organização do bloco de código referente ao Input_Size)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancelar")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Conectar os botões à funcionalidades do sistema
        ok_button.clicked.connect(self.accept_data)
        cancel_button.clicked.connect(self.reject)

    def accept_data(self):
        try:
            self.input_size = int(self.input_size_edit.text())
            self.batch_size = int(self.batch_size_edit.text())
            self.split = float(self.split_edit.text())

            # valida o intervalo do split
            if not (0 < self.split < 1):
                QMessageBox.warning(self, "Erro de valor", "O split deve ser um número entre 0 e 1 (ex: 0.2).")
                return

            self.accept()  # fecha o dialog com resultado "aceito"
        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao definir parâmetros, possivelmente algum valor foi inserido '
                                                  'incorretamente. Tente novamente')

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class NetworkLogName(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Cria uma nova janela para definição de parâmetros do tratamento de
        # dados pelo Image Data Generator
        self.setWindowTitle("Nome de arquivo de LOG (Tensorboard)")

        self.log_name = None
        self.epochs = None

        # Layout principal, organizando verticalmente os widgets na janela
        layout = QVBoxLayout()

        # Organização dos elementos para tornar possível a entrada de valores
        h_layout = QHBoxLayout() # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel("Escolha um nome para o arquivo de LOG (Tensorboard):")) # Adição do widget label ao layout horizontal
        self.name_edit = QLineEdit() # Area de entrada de valores
        self.name_edit.setPlaceholderText("Ex: name_100epochs_split0.3") # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.name_edit) # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout) # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Organização dos elementos para tornar possível a entrada de valores
        h_layout = QHBoxLayout()  # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel(
            "Escolha a quantidade de épocas de treinamento:"))  # Adição do widget label ao layout horizontal
        self.epochs_edit = QLineEdit()  # Area de entrada de valores
        self.epochs_edit.setPlaceholderText('250')  # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.epochs_edit)  # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout)  # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Botões OK e Cancel (Análogo à organização do bloco de código anterior)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancelar")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Conectar os botões à funcionalidades do sistema
        ok_button.clicked.connect(self.accept_data)
        cancel_button.clicked.connect(self.reject)

    def accept_data(self):
        try:
            self.log_name = str(self.name_edit.text())
            self.epochs = int(self.epochs_edit.text())

            # valida se o valor das épocas é inteiro positivo
            if not self.epochs > 0:
                QMessageBox.warning(self, "Erro", "O valor da quantidade de épocas deve ser positivo e inteiro")
                return

            self.accept()  # fecha o dialog com resultado "aceito"
        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao definir parâmetros, possivelmente algum valor foi inserido '
                                                  'incorretamente. Tente novamente')

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class VitParameters(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Cria uma nova janela para definição de parâmetros do tratamento de
        # dados pelo Image Data Generator
        self.setWindowTitle("Configurar Parâmetros da vision transformer")

        # Valores padrões
        self.patch_size = (16, 16)
        self.hidden_size = 768
        self.transformer_layers = 12
        self.num_heads = 12
        self.mlp_units = 3072
        self.batch_size_vit = 256
        self.total_steps = 100000
        self.warmup_steps = 10000
        self.base_lr = 2e-4
        self.mode = "finetune"

        # Layout principal, organizando verticalmente os widgets na janela
        layout = QVBoxLayout()

        # Organização dos elementos para tornar possível a entrada de valores
        # para a variável Patch_size
        h_layout = QHBoxLayout() # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel("Patch size:")) # Adição do widget label ao layout horizontal
        self.patch_size_edit = QLineEdit() # Area de entrada de valores
        self.patch_size_edit.setText("16")
        self.patch_size_edit.setPlaceholderText("Ex: 16") # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.patch_size_edit) # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout) # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Hidden Size (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Hidden Size:"))
        self.hidden_size_edit = QLineEdit()
        self.hidden_size_edit.setText("768")
        self.hidden_size_edit.setPlaceholderText("Default: 768 (Normal)/192 (Tiny)")
        h_layout.addWidget(self.hidden_size_edit)
        layout.addLayout(h_layout)

        # Transform Layers (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Transform Layers (Depth)"))
        self.transformer_layers_edit = QLineEdit()
        self.transformer_layers_edit.setText("12")
        self.transformer_layers_edit.setPlaceholderText("Default: 12 (Normal)/12 (Tiny)")
        h_layout.addWidget(self.transformer_layers_edit)
        layout.addLayout(h_layout)

        # Attention Heads (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Attention Heads"))
        self.num_heads_edit = QLineEdit()
        self.num_heads_edit.setText("12")
        self.num_heads_edit.setPlaceholderText("Default: 12 (Normal)/3 (Tiny)")
        h_layout.addWidget(self.num_heads_edit)
        layout.addLayout(h_layout)

        # MLP Units (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("MLP Units"))
        self.mlp_units_edit = QLineEdit()
        self.mlp_units_edit.setText("3072")
        self.mlp_units_edit.setPlaceholderText("Default: 3072 (Normal)/768 (Tiny)")
        h_layout.addWidget(self.mlp_units_edit)
        layout.addLayout(h_layout)

        # Batch Size (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Batch Size"))
        self.batch_size_edit = QLineEdit()
        self.batch_size_edit.setText("256")
        self.batch_size_edit.setPlaceholderText("Default: 256")
        h_layout.addWidget(self.batch_size_edit)
        layout.addLayout(h_layout)

        # Total Steps (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Total Steps"))
        self.total_steps_edit = QLineEdit()
        self.total_steps_edit.setText("20000")
        self.total_steps_edit.setPlaceholderText("Default: 20000")
        h_layout.addWidget(self.total_steps_edit)
        layout.addLayout(h_layout)

        # Warmup Steps (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Warmup Steps (pre-train)"))
        self.warmup_steps_edit = QLineEdit()
        self.warmup_steps_edit.setText("10000")
        self.warmup_steps_edit.setPlaceholderText("Default: 10000")
        h_layout.addWidget(self.warmup_steps_edit)
        layout.addLayout(h_layout)

        # Base LR (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Base Learning Rate"))
        self.base_lr_edit = QLineEdit()
        self.base_lr_edit.setText("2e-4")
        self.base_lr_edit.setPlaceholderText("Default: 2e-4")
        h_layout.addWidget(self.base_lr_edit)
        layout.addLayout(h_layout)

        # Mode (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Mode"))
        self.mode_edit = QLineEdit()
        self.mode_edit.setText("finetune")
        self.mode_edit.setPlaceholderText("Default: finetune")
        h_layout.addWidget(self.mode_edit)
        layout.addLayout(h_layout)

        # Botões OK e Cancel (Análogo à organização do bloco de código referente ao Patch_size)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancelar")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Conectar os botões à funcionalidades do sistema
        ok_button.clicked.connect(self.accept_data)
        cancel_button.clicked.connect(self.reject)

    def accept_data(self):
        try:

            self.patch_size = int(self.patch_size_edit.text())
            self.hidden_size = int(self.hidden_size_edit.text())
            self.transformer_layers = int(self.transformer_layers_edit.text())
            self.num_heads = int(self.num_heads_edit.text())
            self.mlp_units = int(self.mlp_units_edit.text())
            self.batch_size_vit = int(self.batch_size_edit.text())
            self.total_steps = int(self.total_steps_edit.text())
            self.warmup_steps = int(self.warmup_steps_edit.text())
            self.base_lr = float(self.base_lr_edit.text())
            self.mode = str(self.mode_edit.text())

            # valida se o valor das épocas é inteiro positivo
            if self.mode not in ['finetune', 'pretrain']:
                QMessageBox.warning(self, "Erro", "O modo de treinamento deve ser "
                                                  "especificamente 'pretrain' ou 'finetune'")
                return

            self.accept()  # fecha o dialog com resultado "aceito"

        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao definir parâmetros, possivelmente algum valor foi inserido '
                                                  'incorretamente. Tente novamente')
