# Unity-MNIST
A sample project implementing TensorFlowSharp and a trained CNN network to classify MNIST images in the Unity Video Game Engine. TensorFlow models can be applied to Unity for recognition games and simulations. This example takes in sample image textures, evaluates them using a trained model, and displays the classification to the screen.

## Dependencies
* Unity 2017.2 or above
* Unity TensorFlow Plugin ([Download here](https://s3.amazonaws.com/unity-ml-agents/0.3/TFSharpPlugin.unitypackage))
* TensorFlow 1.4
* Keras 2.1.3
* Python packages:
    * argparse
    * matplotlib
    * pillow
    
## Supported devices
 * Linux 64 bit
 * Mac OS X 64 bit
 * Windows 64 bit
 * iOS (Requires additional steps [found here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity.md#ios-additional-instructions-for-building))
 * Android

## Installation
The following packages are required for this project. Specific TensorFlow and Keras versions are required. If you'd like to keep a newer version of these packages, install these packages in a virtual environment. If you don't wish to use a virtual environment, skip the following section.

### VirtualEnv
To create a new virtual environment for TensorFlow 1.4. Note: your python3 library may differ
~~~
cd ~/
virtualenv -p /usr/bin/python3 tensorflow_1.4
~~~

To activate the virtual environment and use for training
~~~
source ~/tensorflow_1.4/bin/activate
~~~

To deactivate this virtual environment, once finished with training
~~~
deactive
~~~

### Automated install
Warning: running the following script will install TensorFlow, Keras, and python packages. No su permissions are necessary
~~~
chmod +x TensorFlow/install.sh
TensorFlow/install.sh
~~~

### Manual install
How to install all of the packages manually (without install script)

#### TensorFlow
Install version 1.4 of TensorFlow (cpu or gpu). Version 1.4 is required for the TFSharp Unity package
~~~
pip3 install tensorflow==1.4
~~~

#### Keras
Install version 2.1.3 of Keras. This version is one compatible with TensorFlow 1.4
~~~
pip3 install keras==2.1.3
~~~

#### Python packages
Install argparse, matplotlib, and pillow
~~~
pip3 install argparse
pip3 install matplotlib
pip3 install pillow
~~~

## Contributing

Please read [CONTRIBUTING.md](https://github.com/jzharris/Unity-MNIST/blob/master/CONTRIBUTING.md) for details on the code of conduct, and the process for submitting pull requests.

## Authors

* **Zach Harris** - *Initial work*

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/jzharris/Unity-MNIST/blob/master/LICENSE) file for details

## Acknowledgments

* Unity Technologies - *ML Agents* - [Repository](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity.md)
