import tensorflow as tf


def compile_model(model):
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.045,
        rho=0.9,
        momentum=0.9
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, train_ds, val_ds, epochs=50):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.98,
        patience=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr_scheduler]
    )

    return history
