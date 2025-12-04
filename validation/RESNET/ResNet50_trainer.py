import tensorflow as tf
from resnet50_paper import build_resnet50, get_imagenet_tfds

batch_size = 256
train_ds, val_ds = get_imagenet_tfds(batch_size=batch_size, img_size=224, use_gcs=True)

model = build_resnet50(input_shape=(224,224,3), num_classes=1000)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')])

# Learning rate schedule (exemplo: reduzir por 10x a lr nos epochs 30 e 60 em 90 epochs)
def step_decay(epoch):
    if epoch < 30:
        return 0.1
    elif epoch < 60:
        return 0.01
    else:
        return 0.001

lr_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("resnet50_best.h5", save_best_only=True, monitor='val_top5', mode='max')

model.fit(train_ds,
          epochs=90,
          validation_data=val_ds,
          callbacks=[lr_callback, checkpoint_cb])