
# ------------------- Example: build and show summary -------------------
if __name__ == "__main__":
    model = build_resnet50(input_shape=(224,224,3), num_classes=1000, include_top=True)
    model.summary(line_length=120)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')])
    print("\nModel compiled. Ready to train. To load ImageNet via TFDS use get_imagenet_tfds(batch_size=..., img_size=...).")

def build_resnet50(input_shape=(224, 224, 3), num_classes=1000, include_top=True, weight_decay=DEFAULT_WEIGHT_DECAY):
    """
    Helper to instantiate and build a ResNet50 model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    model = ResNet50(num_classes=num_classes, include_top=include_top, weight_decay=weight_decay)
    outputs = model(inputs, training=False)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet50_paper')
    return model
