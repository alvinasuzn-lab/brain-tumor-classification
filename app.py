
def generate_gradcam(model, image_batch, class_index):
    resnet = model.get_layer("resnet50")

    # Find the last convolutional layer in ResNet50.
    last_conv = next(
        (
            layer
            for layer in reversed(resnet.layers)
            if isinstance(layer, tf.keras.layers.Conv2D)
        ),
        None,
    )

    if last_conv is None:
        raise RuntimeError("No convolutional layer found in ResNet50.")

    # Return convolutional activations and ResNet features together.
    feature_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=[last_conv.output, resnet.output],
    )

    # Preserve the original trained classification head.
    resnet_index = model.layers.index(resnet)
    head_layers = model.layers[resnet_index + 1:]

    image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_features, resnet_features = feature_model(
            image_tensor,
            training=False,
        )

        tape.watch(conv_features)

        output = resnet_features
        for layer in head_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                output = layer(output, training=False)
            else:
                output = layer(output)

        # Use the selected class score.
        score = output[:, class_index]

    gradients = tape.gradient(score, conv_features)

    if gradients is None:
        raise RuntimeError(
            "Gradients are unavailable. The model graph needs inspection."
        )

    # Grad-CAM channel weights.
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)

    heatmap = tf.reduce_sum(weights * conv_features, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)

    # Normalize without artificially boosting weak activations.
    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (maximum + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    # Colour map: blue = lower, red = higher relative activation.
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )

    original = np.uint8(np.clip(image_batch[0], 0, 255))

    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_rgb,
        0.40,
        0,
    )

    return heatmap_rgb, overlay
