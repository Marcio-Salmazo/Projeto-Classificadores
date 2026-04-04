import tensorflow as tf
import os


# ======================================================================================================================
# Learning rate e step decay (≈ epochs 30 e 60)
def _lr_schedule(epoch, initial_lr):
    if epoch < 30:
        return initial_lr
    elif epoch < 60:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01


"""
    O paper não define exatamente quando considerar que ocorreu plateau,
    O repositório oficial em Caffe mostra que plateaus ocorrem APROXIMADAMENTE 
    em pontos que correspondem às epochs 30 e 60, Frameworks modernos adotaram isso como “regra”.
    Para uma implementação mais moderna é possível substituir pela função:

    def lr_scheduler(self):

       # Learning rate decay por plateau — fiel ao artigo:
       # - Começa em 0.1
       # - Divide por 10 quando a métrica de avaliação (acc) estaciona

    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor="acc",  # métrica usada no artigo
        factor=0.1,  # divide LR por 10
        patience=3,  # aguarda 3 épocas sem melhora
        mode="max",
        verbose=1,
        min_lr=1e-5  # LR mínimo (opcional)
    )
"""


# ======================================================================================================================
# Criação efetiva dos Callbacks
def _create_callbacks(log_dir, checkpoint_path):
    return [
        tf.keras.callbacks.LearningRateScheduler(
            _lr_schedule,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        ),

        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(log_dir, "training_metrics.csv"),
            separator=",",
            append=False
        ),
    ]


# ======================================================================================================================
# Função de treinamento
def train(model, train_ds, val_ds, batch_size, epochs, train_size, val_size,
          initial_lr=0.1, momentum=0.9, log_dir="logs", checkpoint_path="checkpoints/resnet50_best.h5"):
    """
        Compila o modelo conforme os parâmetros definidos no artigo
        - Loss: SparseCategoricalCrossentropy
             * É usada quando há um problema em que cada exemplo pertence a
               exatamente uma de várias classes possíveis
             * O termo "Sparse" (esparso) refere-se especificamente ao formato
               em que os rótulos de treinamento são fornecidos. Em vez de usar a codificação
               one-hot, os rótulos são fornecidos como inteiros únicos (ImageNet usa labels inteiros).
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=initial_lr,
        momentum=momentum
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Número fixo de iterações (paper)
    steps_per_epoch = train_size // batch_size
    validation_steps = val_size // batch_size

    callbacks = _create_callbacks(log_dir, checkpoint_path)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f">> ÉPOCAS DE TREINAMENTO: {epochs}")
    print(f">> STEPS POR ÉPOCA: {steps_per_epoch}")
    print(f">> QUANTIDADE TOTAL DE STEPS: {epochs * steps_per_epoch}\n")
    print("==================================================================")

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    print(">> TREINAMENTO FINALIZADO COM SUCESSO")
    print(">> MELHOR MODELO FOI SALVO EM: ", callbacks[1].filepath)
    print("==================================================================")

    return history
