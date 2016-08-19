# Install Tensorflow with virtual env.
  1. isntall pip
    sudo easy_install pip

  2. install virtualenvwrapper with upgrade and ignore-installed six.
    http://stackoverflow.com/questions/33185147/on-os-x-el-capitan-i-can-not-upgrade-a-python-package-dependent-on-the-six-compa

    sudo -H pip install virtualenv --upgrade --ignore-installed six
    sudo pip install virtualenvwrapper --upgrade --ignore-installed six

  3. virtualenv places all isolated venv under WORKON_HOME=$HOME/.venv folder.
  source /usr/local/bin/virtualenvwrapper.sh and add to bashrc.

  4. create venv to TF.
    mkvirtualenv tf
    echo $WORKON_HOME   => /Users/hyan2/.virtualenvs

    deactivate
    workon tf
    lssitepackages

  4. TF binary
  
  If install GPU version of tensorflow version, must install Cuda Toolkit 7.5 and cuDNN v4.

    brew install bazel swig
    brew install coreutils
    brew tap caskroom/cask
    brew cask install cuda
    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
    export PATH="$CUDA_HOME/bin:$PATH"
    
    # with GPU does not work !
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.10.0rc0-py2-none-any.whl

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py2-none-any.whl
    
    pip install --upgrade $TF_BINARY_URL

    lssitepackages  => tensorflow, numpy, protobuf, pbr, etc installed.

  The installed site packages is under venv tf.
    python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'

  5. load tensorflow
    python -c "import tensorflow as tf"

    >>> import tensorflow as tf
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> print(sess.run(hello))
    Hello, TensorFlow!
    >>> a = tf.constant(10)
    >>> b = tf.constant(32)
    >>> print(sess.run(a + b))
    

