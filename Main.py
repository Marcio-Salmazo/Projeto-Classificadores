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


# OBSERVAÇÃO:
# Tensorflow 2.10.0
# Python 9
# Numpy 1.23.5
# Scipy 1.13.1
# Protobuf 3.20.2
# Tensorboard 2.10.1
# Pillow
# Scikit-learn

# OBSERVAÇÃO:
# CUDA 11.2
# CuDNN 8.1

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
