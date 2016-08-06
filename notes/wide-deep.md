
Machine learning model needs to learn sparse, specific rules from categorical data with a wide set of cross-feature. It also needs to generalize the learnings for further exploring.

Tensorflow TF.Lear API provides functionalities for building both wide and deep model and combined model for building comprehensive models.

## Wide model

The Wide model contains a wide set of sparse column and crossed column to capture and memorize all those sparse, specific rules of the co-occurrence of features.
Wide models with crossed feature columns can memorize sparse interactions between features effectively. 

    item = model(query) with all sparse base columns and crossed column.

    marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),

## Deep model: Neural Network with Embeddings

The Deep model contains continuous columns, embedding vectors from categorical column, and hidden layers.
Deep model generalizes by matching items to queries that are close in embedding space.

For example, query “fried chicken” correlates to “burgers” as well.
    query-item = lower-dimensional dense representations (embedding vectors).

embedding_column converts high-dim categorical columns to low-dim real-valued vector.

    tf.contrib.layers.embedding_column(workclass, dimension=8),

## Combining Wide and Deep models.
  
Two distinct types of query-item relationships in the data.
  1. targeted. query really mean it, need to match text as exactly as possible.
  2. exploratory. “seafood” or “italian food”.

  m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])

  if FLAGS.model_type == "wide":
      m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                            feature_columns=wide_columns)
    elif FLAGS.model_type == "deep":
      m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                         feature_columns=deep_columns,
                                         hidden_units=[100, 50])
    else:
      '''Fully connected hidden layers to find every combination of pair of features.'''
      m = tf.contrib.learn.DNNLinearCombinedClassifier(
          model_dir=model_dir,
          linear_feature_columns=wide_columns,
          dnn_feature_columns=deep_columns,
          dnn_hidden_units=[100, 50])
    return m

  m.fit(input_fn=train_input_fn, steps=200)
  results = m.evaluate(input_fn=eval_input_fn, steps=1)
  for key in sorted(results):
      print "%s: %s" % (key, results[key])



## Columns and FeatureColumn

Sparse features like query="fried chicken" and item="chicken fried rice" are included in both wide and deep models. Prediction errors backpropagated to both models.

  race = tf.contrib.layers.sparse_column_with_keys(
    column_name="race", keys=["Asian-Pac-Islander", "Black", "Other", "White"])
  
  education = tf.contrib.layers.sparse_column_with_hash_bucket(
    "education", hash_bucket_size=1000)

  age = tf.contrib.layers.real_valued_column("age")
  age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  wide_columns = [
    age_buckets, gender,
    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
  ]

Each of the sparse, high-dimensional categorical features are first converted into a low-dimensional and dense real-valued vector, often referred to as an embedding vector.
  
  deep_columns = [
    age,
    tf.contrib.layers.embedding_column(race, dimension=8),
  ]

## Model

placeholder holds values for training input and classifier labels.
  x = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])

variables store model parameters.
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))

variable shape: variable is a multidimensional array.
weight variable is a weight list with each entry repr the weight of one input.

For the first layer, the dimensions are [IMAGE_PIXELS, hidden1_units].
[batch, in_height, in_width, in_channels] => [ht*width*chan, out-chan]

  weights = tf.Variable(tf.truncated_normal(
    [IMAGE_PIXELS, hidden1_units],  // 768 pixels connect to input weight, with n neurons in the layer.
  biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
  
  hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

## ImageNet and Inception with Convolution and Pooling

  Tensorflow provide API for convolution and pooling with tf.nn.conv2d() and tf.nn.max_pool.
  
    tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

    def conv2d(x, W):  # weight as filter
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
  Weight varialbe shape is [5,5,1,32], means 5x5 patch size, 1 input channel, and 32 output channels.
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


## Slim scope, restorable variables, ops, loss.

  weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
  l2_regularizer = lambda t: losses.l2_loss(t, weight=0.0005)
  weights = variables.variable('weights',
                               shape=[100, 100],
                               initializer=weights_initializer,
                               regularizer=l2_regularizer,
                               device='/cpu:0')

  biases = variables.variable('biases',
                              shape=[100],
                              initializer=tf.zeros_initializer,
                              device='/cpu:0')

  conv2d creates a variable called 'weights', representing the convolutional
  kernel/matrix, that is convolved with the input. it returns a tensor representing
  the output of the operation

  Input layer of convolution network has [image_width * image_height] neurons.
  We then slide the local receptive field across the entire input image. 
  Each neuron in hidden layer learn to analyze its particular local receptive field.

  For each local receptive field, there is a different hidden neuron in the first hidden layer.
  Each neuron in the first hidden layer connect to one local receptive region(input neurons inside the local receptive region), and compute a dot product between weights and local receptive field.
  
  The number of neurons in first hidden layer is f(kernel_h, kernel_w, stride), as local receptive field moves stride once.
  For example, for 28×28 input image, and 5×5 local receptive fields, then there will be 
  24×24 (28-5+1) neurons in the first hidden layer. 
  Each hidden neuron 5x5 weights and a bias connected to its local receptive field.
  For each feature map we need 25=5×5 shared weights, plus a single shared bias.
  As a result, the shape of weight is matrix of [n-neurons, [local receptive field]]

  The map from input neuron to hidden layer is called feature map, as all neurons in the first hidden layer detect the same exact feature, just at different location(local receptive field) in the image.

  Each feature map is defined by a set of height x width shared weights. Shared weights of said to define a kernel or filter. kenerl size/filter size is the dim of local receptive field.
  For example, if we have 3 feature maps in first hidden layer, the shape of weights = [3x24x24].
  
  Example of 28x28 image, 20 feature maps, with 5x5 local receptive field. stride 1.
    >>> net = Network([
          ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
            ## output 20 feature maps, each neuro connect to one 5x5 local receptive field
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2)),
          ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), # input is 20 feature map, 12x12
            # output 40 feature maps, still 5x5 on 12x12, each neuron connect to all 20 prev output
            filter_shape=(40, 20, 5, 5),
            poolsize=(2, 2)),
          FullyConnectedLayer(n_in=40*4*4, n_out=100),
          SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    >>> net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)  

  After we define many feature maps, we apply max-pooling to each feature map separately.
  if we apply max 2x2 pooling, the output will from 24x24 neurons to 12x12 neurons for one feature map.

    def conv2d(inputs, num_filters_out, kernel_size, stride=1,...)
      inputs: a tensor of size [batch_size, height, width, channels] or [batch_size, channels].
      kernel_size: defines feature map of height * width shared weights, and a single shared bias. 
        a list of length 2: [kernel_height, kernel_width] of the filters

      num_filters_in = inputs.get_shape()[-1]
      weights_shape = [kernel_h, kernel_w, num_filters_in, num_filters_out]

  For image classification with Inception, conv2d weight is a matrix of 
    [kernel_h * kernel_w * num_filters_in, num_filters_out]

  net = slim.ops.conv2d(input, 32, [3, 3], scope='conv1')

  # Get all model variables from all the layers.
  model_variables = slim.variables.get_variables()

  # Get all model variables from a specific the layer, i.e 'conv1'.
  conv1_variables = slim.variables.get_variables('conv1')

  # Get all variables to restore.
  # (i.e. only those created by 'conv1' and 'conv2')
  variables_to_restore = slim.variables.get_variables_to_restore()

