Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Hallplats/*'))
for image in Data_dir[:10]:
    image_size = 128
    
    # Load pre-trained Keras model and the image to classify
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dim = (128,128)
    img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_tensor = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor/255
    
    #image = np.random.random((image_size, image_size, 3))
    #img_tensor = preprocessing.image.img_to_array(image)
    #img_tensor = np.expand_dims(img_tensor, axis=0)
    #img_tensor = preprocess_input(img_tensor)
    
    conv_layer = three_layer_cnn.layers[-10]
    heatmap_model = models.Model([three_layer_cnn.inputs], [conv_layer.output, three_layer_cnn.output])
    print(heatmap_model)
    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
   # heatmap = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = reshape(heatmap, (60,60), 3)
    heatmap = np.maximum(heatmap, 0)

    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    
    heatmap_img = cv2.applyColorMap(img_orig, cv2.COLORMAP_JET)
    plt.matshow(heatmap_img)
    plt.show()
