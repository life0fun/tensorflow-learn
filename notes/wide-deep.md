
Machine learning model needs to learn sparse, specific rules from categorical data with a wide set of cross-feature. It also needs to generalize the learnings for further exploring.

Tensorflow TF.Lear API provides functionalities for building both wide and deep model and combined model for building comprehensive models.

## Wide model

The Wide model contains a wide set of cross-feature to capture and memorize all those sparse, specific rules of the co-occurrence of [query-item, labels].

    item = model(query) with all sparse base columns and crossed column.

    marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),

## Deep model

The Deep model generalize by matching items to queries that are close in embedding space.
choose continuous columns, embedding vectors/dim for categorical column, and hidden layer size.
For example, query “fried chicken” correlates to “burgers” as well.
  query-item = lower-dimensional dense representations (embedding vectors).

    tf.contrib.layers.embedding_column(workclass, dimension=8),

## Combining Wide and Deep models.
  
two distinct types of query-item relationships in the data.
    1. targeted. query really mean it, need to match text as exactly as possible.
    2. exploratory. “seafood” or “italian food”.

  m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])

## Columns

Sparse features like query="fried chicken" and item="chicken fried rice" are included in both wide and deep models. Prediction errors backpropagated to both models.

Continuous columns, embedding columns for categorical columns, and hidden layers are included in deep model.

## Train