
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



## Columns

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
For the first layer, the dimensions are [IMAGE_PIXELS, hidden1_units] 
  
  weights = tf.Variable(tf.truncated_normal(
    [IMAGE_PIXELS, hidden1_units],  // 768 pixels connect to input weight, with n neurons in the layer.
  biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
  
  hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

Convolution and Pooling

  Tensorflow provide API for convolution and pooling with tf.nn.conv2d() and tf.nn.max_pool.
  
    tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

  Weight varialbe shape is [5,5,1,32], means 5x5 patch size, 1 input channel, and 32 output channels.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
