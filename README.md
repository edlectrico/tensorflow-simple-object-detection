# Tensorflow-simple-object-detection
A simple image-based object detection tensorflow app in Python
Originally forked from [here](https://github.com/diegocavalca/machine-learning).

## Instructions:
#### Install Python
In my case, I have installed Python 3.6.2 via Anaconda. If you want to do that (which I recommend), download the corresponding installer from Anaconda's [website](https://www.anaconda.com/download/).

#### Download Tensorflow (via pip)
Select your OS and install Tensorflow following the [installation guide](https://www.tensorflow.org/install/).

#### Download object_detection and put it in the tensorflow installation dir (pip show tensorflow)
The object_detection module from Tensorflow is not installed by default. You should download it from the Tensorflow Github repo and place it in the Tensorflow home directory. Clone or download [this repo](https://github.com/tensorflow/models), extract the object_detection folder, and place ir in `<PATH_TO_YOUR_TF>/models`.

To make object_detection libs available, do this in Python:
```python
import sys
sys.path.append('<PATH_TO_TENSORFLOW>/models')
sys.path.append('<PATH_TO_TENSORFLOW>/models/slim')
from object_detection.utils import label_map_util # This now should work
from object_detection.utils import visualization_utils as vis_util
```
If you installed Python through Anaconda, the `<PATH_TO_TENSORFLOW>` will look like `/home/user/anaconda3/lib/python3.6/site-packages/tensorflow`.

#### Install dependencies
Make sure that you have downloaded `moviepy` and `ffmpeg` for image processing.
```
pip install moviepy
```
```
sudo apt-get install ffmpeg
```
