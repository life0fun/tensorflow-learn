## Machine Learning with tf.contrib.learn API.

A tf.Tensor represents a value produced by an Operation in a tensorflow session.

A Tensor is a symbolic handle to one of the outputs of an Operation. 
It does *NOT* hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow Session.

tf.learn provides a set of APIs for loading dataset and create classifer.

1. training data from load csv data.

    training_set = tf.contrib.learn.datasets.base.load_csv(
      filename=IRIS_TRAINING, target_dtype=np.int)

2. Construct a Deep Neural Network Classifier, 3 layer with 10, 20, 10 units respectively.
    
    classifier = tf.contrib.learn.DNNClassifier(
      hidden_units=[10, 20, 10], n_classes=3)

3. Fit the model with training data
    
    classifier.fit(x=x_train, y=y_train, steps=200)

4. Evaluate

    accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]

5. predict / classify

    new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    y = classifier.predict(new_samples)

## Linear Model

A linear model uses a single weighted sum(features...) to make predication.
Logistic regression plugs weighted sum into logistic fn and output value 0..1
    
    [0..1] <= Logistic(weighted sum)

Build linear model with feature columns and transformations.
A FeatureColumn represents a single feature in your data.

### Continuous columns

with real_valued_column

    age = tf.contrib.layers.real_valued_column("age")

or bucketization
    
    age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

### Categorical features 

Category features in linear model is transformed into a sparse vector. Linear model can handle large sparse vector.

Encoding sparse columns

  education = tf.contrib.layers.sparse_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
  eye_color = tf.contrib.layers.sparse_column_with_keys(
    column_name="eye_color", keys=["blue", "brown", "green"])

### Feature Crosses

  sport_x_city = tf.contrib.layers.crossed_column(
    [sport, city], hash_bucket_size=int(1e4))


### Input Builder function

To pump data into model feature columns, define input builder fn and pass input builder fn to model's `fit` and `evaluate` APIs that initiate training and evaluations.

TF.Learn model, the input data is specified by means of an Input Builder function.

Input builder converts data into Tensor or sparseTensor. It returns a pair of
  
  feature_cols: {feature_column_name: tensor, sparse_feature: sparseTensor}
  label: a tensor containing the label column.

We need two input builder, one for training and one for evaluation.

    import tensorflow as tf

    def input_fn(df):
      # Creates a dictionary mapping from each continuous feature column name (k) to
      # the values of that column stored in a constant Tensor.
      continuous_cols = {k: tf.constant(df[k].values)
                         for k in CONTINUOUS_COLUMNS}
      # Creates a dictionary mapping from each categorical feature column name (k)
      # to the values of that column stored in a tf.SparseTensor.
      categorical_cols = {k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          shape=[df[k].size, 1])
                          for k in CATEGORICAL_COLUMNS}
      # Merges the two dictionaries into one.
      feature_cols = dict(continuous_cols.items() + categorical_cols.items())
      # Converts the label column into a constant Tensor.
      label = tf.constant(df[LABEL_COLUMN].values)
      # Returns the feature columns and the label.
      return feature_cols, label

    def train_input_fn():
      return input_fn(df_train)

    def eval_input_fn():
      return input_fn(df_test)


## Selecting and Engineering Features for the Model

Selecting and crafting the right set of feature is key to learning an effective model.

## Linear Estimator with tf.contrib.learn.LinearClass

build linear model with LinearClassifier by providing a list of feature columns.

    e = tf.contrib.learn.LinearClassifier(feature_columns=[...])
    e.fit(input_fn=input_fn_train, steps=200)
    results = e.evaluate(input_fn=input_fn_test, steps=1)


## Adding Regularization to Prevent Overfitting

  m = tf.contrib.learn.LinearClassifier(feature_columns=[
    gender, native_country, education, occupation, workclass, marital_status, race,
    age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=1.0,
      l2_regularization_strength=1.0),
    model_dir=model_dir)


## Exporter

We can use TensorFlow Serving Exporter module to export the model.
saver is used to serialize graph variable values to the model export so that they can be properly restored later.
sess is the TensorFlow session that holds the trained model you are exporting.
sess.graph.as_graph_def() is the protobuf of the graph. exporter init take it.

    from tensorflow.contrib.session_bundle import exporter
    
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(input_tensor=x, scores_tensor=y)
    model_exporter.init(sess.graph.as_graph_def(),
                        default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)


