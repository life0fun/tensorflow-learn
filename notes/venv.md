# Install Tensorflow with virtual env.
  1. isntall pip
    sudo easy_install pip

  2. install virtualenvwrapper on El Captain.
    http://stackoverflow.com/questions/33185147/on-os-x-el-capitan-i-can-not-upgrade-a-python-package-dependent-on-the-six-compa

    sudo -H pip install virtualenv --upgrade --ignore-installed six
    sudo pip install virtualenvwrapper --upgrade --ignore-installed six

  3. virtualenv places all isolated venv under WORKON_HOME=$HOME/.venv folder.
  source /usr/local/bin/virtualenvwrapper.sh and add to bashrc.
    
  4.


