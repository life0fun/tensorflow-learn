## Integrate Inception v3 into Your App.

Tensorflow provide inception model trained with ImageNet data with 1000 classes.
You can extract higher level features from this model which may be reused for other vision tasks.

We can integrate tensorflow pre-trained inception v3 model dirctly into our application. Tensorflow provide python and c++ APIs for your app to use.
    
Python API:

classify_image.py downloads the trained model from tensorflow.org when the program is run for the first time. You'll need about 200M of free space available on your hard disk.

    python tensorflow/models/image/imagenet/classify_image.py --image_file="x.png"

C++ API:

we need to download pre-trained inception v3 model and compile the c++ binary that includes the code to load and run the graph.

    bazel build tensorflow/examples/label_image/data/inception_dec_2015.zip
    bazel-bin/tensorflow/examples/label_image/label_image --image=my_image.png

Integration into your App:

To load pre-trained inception v3 model protobuf file, use ReadBinaryProto().
You can create GraphDefBuilder b object to specify a model to run also.

    load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    session_create_status = (*session)->Create(graph_def);

The inception model expect 299x299 RGB images. We first need to load, resize, and scale the pixel values as desired to get the result the main model expects as its input.

    ReadTensorFromImageFile(image_path, ht, width, input_mean, input_std, &resized_tensors);

    session->Run({{FLAGS_input_layer, resized_tensors[0]}},
                  {FLAGS_output_layer}, {}, &out_tensors)

    *scores = out_tensors[0];
    *indices = out_tensors[1];

Finally, we analyzes the output of the Inception graph to retrieve the highest scores and
their positions in the tensor, which correspond to categories.
    
    GetTopLabels(vector<Tensor>& out_tensors, labels, Tensor* indices, Tensor* scores);



## How to Retrain Inception's final layer for New Categories

First, provide a set of images to teach the model about the new classes you want to recognize.
then build the re-trainer binary and run it. The script loads the pre-trained inception v3 model, remove the old top layer, and trains the new one on the new class image data provided.

    bazel build tensorflow/examples/image_retraining:retrain
    bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/flower_photos

The magic of transfer learning is that lower layers that have been trained to distinguish between some objects can be re-used for many recognition tasks without any alternation.

'Bottleneck' is an informal term we often use for the layer just before the final output layer that actually does the classification. This penultimate layer has been trained to output a set of values that's good enough for the classifier to use to distinguish between all the classes it's been asked to recognize. 

The meaningful summary information needed for recognition is well obtained through many layers before bottleneck and is suitable to perform any final classification.

we cache the bottleneck layer by default into /tmp/bottleneck for re-use during training.

### Using Retrained model

A version of inception v3 with a final layer retrained to your categories is stored at /tmp/output_graph.pb, and a text file contains labels to /tmp/output_labels.txt.

To use it, supply with --graph=/tmp/putput_graph.pb and --labels=/tmp/output_labels.txt
  
  bazel build tensorflow/examples/label_image:label_image
  bazel-bin/tensorflow/examples/label_image/label_image
    --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt
    --output_layer=final_result
    --image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg


### How to Fine-Tune a Pre-Trained Model on a New Task

https://github.com/tensorflow/models/blob/master/inception/README.md#how-to-fine-tune-a-pre-trained-model-on-a-new-task

1. convert img data to the sharded TFRecord format (serialized tf.Example protobuf).

    FLOWERS_DATA_DIR=$HOME/flowers-data
    bazel build inception/download_and_preprocess_flowers
    bazel-bin/inception/download_and_preprocess_flowers "${FLOWERS_DATA_DIR}"

To prepare your custom image data set for transfer learning, use `build_image_data.py` on your custom data set. If your custom data has a different number of examples or classes, you need to change the appropriate values in `imagenet_data.py`.

2. Retrain a pre-trained Model on your own data set.

To fine-tune a pre-trained inception v3 model on your data set, we need

1. change the num of labels in the final classification layer.
2. restore all model params(weight, bias) from the pre-trained inception v3 except for the final classification layer. The final layer gets randomized value initially. Two flags:

    --pretrained_model_checkpoint_path=/path/to/pre-trained/model
    --fine_tune=true // random init the final classification layer


    bazel build inception/flowers_train

    # Path to the downloaded Inception-v3 model.
    MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"

    # Directory where the flowers data resides.
    FLOWERS_DATA_DIR=/tmp/flowers-data/

    # Directory where to save the checkpoint and events files.
    TRAIN_DIR=/tmp/flowers_train/

    # Run the fine-tuning on the flowers data set starting from the pre-trained
    # Imagenet-v3 model.
    bazel-bin/inception/flowers_train \
      --train_dir="${TRAIN_DIR}" \
      --data_dir="${FLOWERS_DATA_DIR}" \
      --pretrained_model_checkpoint_path="${MODEL_PATH}" \
      --fine_tune=True \
      --initial_learning_rate=0.001 \
      --input_queue_memory_factor=1



### Construct new dataset for Retraining

`build_image_data.py` takes a structured directory of images and convert it to a shared TFRecord that can be read by the inception model.
    
    $TRAIN_DIR/dog/image0.jpeg
    $VALIDATION_DIR/cat/image1.jpg

Each sub-directory corresponds to a unique label/class for the image.

    OUTPUT_DIRECTORY=$HOME/my-custom-data/

    # build the preprocessing script.
    bazel build inception/build_image_data

    # convert the data.
    bazel-bin/inception/build_image_data \
      --train_directory="${TRAIN_DIR}" \
      --validation_directory="${VALIDATION_DIR}" \
      --output_directory="${OUTPUT_DIRECTORY}" \
      --labels_file="${LABELS_FILE}" \
      --train_shards=128 \
      --validation_shards=24 \
      --num_threads=8

Output

    $TRAIN_DIR/train-00023-of-00024
    $VALIDATION_DIR/validation-00007-of-00008

Each label corresponds to a number starting from 1. we have 24 shards.


### HyperParameters

Roughly 5-10 hyper-parameters govern the speed at which a network is trained. In addition to --batch_size and --num_gpus, there are several constants defined in inception_train.py which dictate the learning schedule.

    RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
    MOMENTUM = 0.9                     # Momentum in RMSProp.
    RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
    INITIAL_LEARNING_RATE = 0.1        # Initial learning rate.
    NUM_EPOCHS_PER_DECAY = 30.0        # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.16  # Learning rate decay factor.

Higher learning rate, large batch size, etc.


## Inception_v3 and Slim scope, variable, ops, loss

inception_v3 architecture is weight with 32 feature maps from 3x3 filter/kernel size and cascading prev layers output weight as next layers input weight.
  
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
  



