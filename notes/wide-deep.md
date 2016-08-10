
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

## Build the Graph

3-stage pattern: inference(), loss(), and training().

  inference() - Builds the graph as far as is required for running the network forward to make predictions.
  loss() - Adds to the inference graph the ops required to generate loss.
  training() - Adds to the loss graph the ops required to compute and apply gradients.

placeholder: holds training input and classifier labels.
  x = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])

variables store model parameters.
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))

variable shape is a multidimensional array.
weight variable is a weight list with each entry repr the weight of one input.

For the first layer, the dimensions are [IMAGE_PIXELS, hidden1_units].
[batch, in_height, in_width, in_channels] => [ht*width*chan, out-chan]

  weights = tf.Variable(tf.truncated_normal(
    [IMAGE_PIXELS, hidden1_units],  // 768 pixels connect to input weight, with n neurons in the layer.
  biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
  hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  logits = tf.matmul(hidden2, weights) + biases

Loss: 
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

Training:
  
  tf.scalar_summary(loss.op.name, loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)

  with tf.Graph().as_default():
    sess = tf.Session()
    with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    for step in xrange(FLAGS.max_steps):
      sess.run(train_op)

Feed the Graph
  for each step, the code will generate a feed dictionary for next train batch.
    
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                             FLAGS.fake_data)
  feed_dict = {
      images_placeholder: images_feed,
      labels_placeholder: labels_feed,
  }

  for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train,images_placeholder,labels_placeholder)
    _, loss_value = sess.run([train_op, loss],
                           feed_dict=feed_dict)

  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
  summary_str = sess.run(summary_op, feed_dict=feed_dict)
  summary_writer.add_summary(summary_str, step)

Checkpoint and Restor
  
  saver = tf.train.Saver()
  saver.save(sess, FLAGS.train_dir, global_step=step)
  saver.restore(sess, FLAGS.train_dir)


Example of 28x28 image, 20 feature maps, with 5x5 local receptive field. stride 1.
  image_shape as input to conv layer reduced by pool at each layer.
  weight_shape at each layer is decided by local receptive region and feature maps.
    >>> net = Network([
          ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), // img shape 28x28
            ## create 20 feature maps, each neuro connect to one 5x5 local receptive field
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2)),   # after 2x2 max pool, 28x28 reduced to 12x12
          ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), // img shape of 12x12 with 20 features
            ## create 40 feature maps from prev 20 maps, still 5x5 region over 12x12 img shape
            filter_shape=(40, 20, 5, 5),
            poolsize=(2, 2)),
          FullyConnectedLayer(n_in=40*4*4, n_out=100),
          SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    >>> net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)  

## ImageNet and Inception with Convolution and Pooling

  Tensorflow provide API for convolution and pooling with tf.nn.conv2d() and tf.nn.max_pool.
  
    tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

    def conv2d(x, W):  # weight as filter
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
  1st layer weight shape [5,5,1,32], 5x5 region, 1 input channel, output 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

  2nd layer, 32 feature maps from first layer, stil 5x5 local receptive region, output 64 features.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

#######
## Slim scope, restorable variables, ops, loss.
#######
  Mostly for convolution netowrk, with `conv2d()`.

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
  
  conv2d creates and return a variable called 'weights', representing the convolutional
  kernel/matrix, that is convolved with the input. it returns a tensor representing
  the output of the operation.
  Note `conv2d` returns weights variable with internally re-shape input to filters_out.

  def conv2d(inputs, num_filters_out, kernel_size, stride=1,...)
      inputs: a tensor of shape [batch_size, height, width, channels] or [batch_size, channels].
      kernel_size: defines feature map of height * width shared weights, and a single shared bias. 
        the size of local receptive region. [kernel_height, kernel_width] of the filters

      num_filters_in = inputs.get_shape()[-1]
      weights_shape = [kernel_h, kernel_w, num_filters_in, num_filters_out]

      For image classification with Inception, conv2d weight is a matrix of 
      [kernel_h * kernel_w * num_filters_in, num_filters_out]
      num_filters_out is feature map

  
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

  inception_v3 architecture is weight with 32 feature maps from 3x3 filter/kernel size
  # cascading prev layers output weight as next layers input weight.
  
  with tf.op_scope([inputs], scope, 'inception_v3'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout]:
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):
        # input: 299 x 299 x 3, output: 149x149x3, as stride=2
        end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        # input: 149 x 149 x 32, output: 147 x 147 x 32
        end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # input: 147 x 147 x 32, output: 147 x 147 x 64
        end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2')
        # input: 147 x 147 x 64, output: 73x73x64, stride=2
        end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # input: 73 x 73 x 64, outout: 73x73x80 with 1x1 convolute
        end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3')
        # input: 73 x 73 x 80, output: 71x71x192
        end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4')
        # input 71 x 71 x 192, output: 35x35x192, stride=2
        end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # output: 35 x 35 x 192.
        net = end_points['pool2']
  
## Model parameters checkpoint and restore

  # Get all model variables from all the layers.
  model_variables = slim.variables.get_variables()

  # Get all model variables from a specific the layer, i.e 'conv1'.
  conv1_variables = slim.variables.get_variables('conv1')

  # Get all weights from all the layers.
  weights = slim.variables.get_variables_by_name('weights')

  # Get all bias from all the layers.
  biases = slim.variables.get_variables_by_name('biases')

  # Get all variables to restore.
  variables_to_restore = slim.variables.get_variables_to_restore()



## ML with tf.contrib.learn

