# Unity-MNIST
A sample project implementing TensorFlowSharp and a trained Convolutional Neural Network (CNN) to classify MNIST images in the Unity Video Game Engine. TensorFlow models can be applied to Unity for recognition games and simulations. This example takes in sample image textures, evaluates them using a trained model, and displays the classification to the screen.

<p align="center"> 
   <img src="https://github.com/jzharris/Unity-MNIST/blob/master/Unity-Files/Screenshots/running_game.png">
</p>

## Dependencies
* Unity 2017.2 or above
* TensorFlow 1.4 or above
* Unity TensorFlow Plugin
    * Download [v0.3](https://s3.amazonaws.com/unity-ml-agents/0.3/TFSharpPlugin.unitypackage) if using Tensorflow 1.4.
    * Download [v0.4](https://s3.amazonaws.com/unity-ml-agents/0.4/TFSharpPlugin.unitypackage) or [v0.5](https://s3.amazonaws.com/unity-ml-agents/0.5/TFSharpPlugin.unitypackage) if using a later Tensorflow version.
* Unity .Net 4.6 or above
* Keras 2.1.3
* Python packages:
    * argparse
    * matplotlib
    * pillow
    
## Supported devices
 * Linux 64 bit
 * Mac OS X 64 bit
 * Windows 64 bit
 * Android
 * iOS requires additional steps:
     * iOS support for TFSharp [found here](https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity-(Experimental).md#ios-additional-instructions-for-building).
     * Requires Tensorflow 1.8.

## Installation
The following packages are required for this project. Specific TensorFlow and Keras versions are required. If you'd like to keep a newer version of these packages, install these packages in a virtual environment. If you don't wish to use a virtual environment, skip the following section.

### VirtualEnv
To create a new virtual environment for TensorFlow 1.4. Note: your python3 library may differ.
~~~
cd ~/
virtualenv -p /usr/bin/python3 tensorflow_1.4
~~~

To activate the virtual environment and use for training.
~~~
source ~/tensorflow_1.4/bin/activate
~~~

To deactivate this virtual environment, once finished with training.
~~~
deactive
~~~

### Automated install
Warning: running the following script will install TensorFlow, Keras, and python packages. No su permissions are necessary.
~~~
chmod +x TensorFlow/install.sh
TensorFlow/install.sh
~~~

### Manual install
How to install all of the packages manually (without install script).

#### TensorFlow
Install version 1.4 of TensorFlow (cpu or gpu). Version 1.4 is required for the TFSharp Unity package.
~~~
pip3 install tensorflow==1.4
~~~

#### Keras
Install version 2.1.3 of Keras. This version is one compatible with TensorFlow 1.4.
~~~
pip3 install keras==2.1.3
~~~

#### Python packages
Install argparse, matplotlib, and pillow.
~~~
pip3 install argparse
pip3 install matplotlib
pip3 install pillow
~~~

## Running the project

### Opening the Unity project
The Unity project is located in the following folder: `Unity-Files/MNIST`. Open this folder from Unity to get started. The example scene can be found in `Scenes > Classifier`.

### Training the TF model
Run the following python script to train the CNN model.
~~~
cd TensorFlow/
python3 mnist_cnn1.py
~~~

Parameters that can be run with this script:
~~~
--model_name            # the name of the saved model                         default='mnist_cnn1'
--export_images         # instead of training a model, save MNIST to .png's   default=False
--export_number         # number of MNIST images to export (if enabled)       default=10
--plot_images           # instead of training a model, display MNIST images   default=False
--epochs                # number of epochs to train model for                 default=5
--batch_size            # batch size to use for training                      default=128
~~~

These parameters can be applied to the python script in the following way:
~~~
python3 mnist_cnn1.py --[param1_name]=[param1_value] --[param2_name]=[param2_value]
~~~

### Adding the graph to Unity
The python script will generate a graph of the TensorFlow model once training is complete. Two files of importance are saved: *frozen_mnist_cnn1.bytes* and *opt_mnist_cnn1.bytes*. The latter is an optimized (smaller) version of the former. The model files are saved to the following location:
~~~
TensorFlow/out/frozen_mnist_cnn1.bytes
TensorFlow/out/opt_mnist_cnn1.bytes
~~~

To add these files to Unity, run the following while inside the parent directory of the project.
~~~
cp TensorFlow/out/*.bytes Unity-Files/MNIST/Assets/TensorFlow/Frozen_graphs/
~~~

Add either of the .bytes files to the *Graph Model* element in the MNIST_Classifier object. Add any image textures you wish to test with the model to the *Input Textures* element. See the following for a working example setup.

<p align="center"> 
   <img src="https://github.com/jzharris/Unity-MNIST/blob/master/Unity-Files/Screenshots/setup.png">
</p>

### Playing the game
In order to process the Unity TensorFlow Plugin, you must perform the following under Edit > Project Settings > Player > Other Settings
* Add ENABLE_TENSORFLOW to *Scripting Define Symbols*
* Change *Scripting Runtime Version* to .Net 4.x

Press play, and use the arrow keys to change the input image to the TensorFlow model.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/jzharris/Unity-MNIST/blob/master/CONTRIBUTING.md) for details on the code of conduct, and the process for submitting pull requests.

## Authors

* **Zach Harris** - *Initial work*

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/jzharris/Unity-MNIST/blob/master/LICENSE) file for details.

## Acknowledgments

* Unity Technologies - *ML Agents* - [Repository](https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity-(Experimental).md).
