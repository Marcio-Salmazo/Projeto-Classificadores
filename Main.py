import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Escolhe a GPU (0 = principal)
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Evita alocar toda VRAM

import sys
from PyQt5.QtWidgets import QApplication
from Interface import Interface
from tensorflow.keras import mixed_precision

"""
    ==========================
    VERSÃO DO PYTHON:
    
    Python 3.9.13
    
    ==========================
    VERSÕES DAS BIBLIOTECAS:
    
    tensorflow==2.10.0
    numpy==1.23.5
    scipy==1.13.1
    protobuf==3.20.2
    tensorboard==2.10.1
    Pillow
    scikit-learn~=1.6.1
    openpyxl
    PyQt5~=5.15.11
    pandas~=2.3.3
    tensorflow-datasets
    tensorflow-datasets==4.7.0
    protobuf==3.20.0
    
    ==========================
    OBSERVAÇÃO:
    
        CUDA 11.2
        CuDNN 8.1
    ==========================
"""



class Main:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.interface = Interface()

    def run(self):
        self.interface.show()
        self.interface.resize(800, 600)
        sys.exit(self.app.exec())


if __name__ == "__main__":
    # Configura GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU habilitada com memory_growth")
        except RuntimeError as e:
            print(e)
    else:
        print("Nenhuma GPU detectada. Usando CPU.")

    # Mixed precision
    mixed_precision.set_global_policy("float32")
    print("Mixed precision ativado")

    main = Main()
    main.run()
