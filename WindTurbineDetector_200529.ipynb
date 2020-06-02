{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "WindTurbineDetector_200529.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python2",
      "display_name": "Python 2"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qRSPkQ_ryE9t",
        "colab_type": "text"
      },
      "source": [
        "# Wind Turbine Object Detection from Aerial Imagery Using TensorFlow Object Detection API and Google Colab"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9vLSXaxZ2aCT",
        "colab_type": "text"
      },
      "source": [
        "## Introduction\n",
        "\n",
        "This notebook provides the full pipeline to perform training and inference for a wind turbine object detection model using publicly available aerial images and the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). It is designed to run in [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb), a Jupyter notebook environment running on a virtual machine (VM) that provides free access to a Tesla K80 GPU for up to 12 hours.\n",
        "\n",
        "\n",
        "The aerial image data set used in this notebook is obtained from the [National Agriculture Imagery Program (NAIP) database](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) using [USGS EarthExplorer](https://earthexplorer.usgs.gov/). The particular NAIP images used to train, test, and validate this model are from three wind farms located in west-central Iowa containing turbines of varying capacity, style, and manufacturer. A sample NAIP image is presented below in the \"Sample NAIP image\" section. The original NAIP images are 5978 x 7648 so they had to be chipped into smaller individual images to avoid excessive memory use. In addition, the ratio of object size to image size is improved by this operation. An image size of 300 x 300 was chosen since the TensorFlow object detection SSD-based models rescale all input images to this size. \n",
        "\n",
        "A total of 488 images, all containing at least one full wind turbine, were collected and split into train (\\~80%), test (\\~16%), and validate (\\~4%) sets. [LabelImg](https://github.com/tzutalin/labelImg) was then used to label all the images in the train and test sets. Samples of the chipped and annotated images are shown below in the \"Sample chipped and annotated NAIP images\" section. Annotating the images in LabelImg creates an XML file corresponding to each image. These XML files must be converted to CSV and then TFRecords. Sample code for this can be found [here](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) or [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) (among other places)."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TahAg7fTspdN",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# clone wind-turbine-detector repo\n",
        "!git clone https://github.com/lbborkowski/wind-turbine-detector.git"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NIL8IS9R57U-",
        "colab_type": "text"
      },
      "source": [
        "### Sample NAIP image"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wARV6z136BrQ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from matplotlib import pyplot as plt\n",
        "from PIL import Image\n",
        "import os\n",
        "import glob\n",
        "\n",
        "%matplotlib inline\n",
        "\n",
        "image = Image.open('/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg')\n",
        "plt.figure(figsize=(12,8))\n",
        "plt.axis('off')\n",
        "plt.imshow(image)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xjj9v7yf8He4",
        "colab_type": "text"
      },
      "source": [
        "### Sample chipped and annotated NAIP images"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oJrOGAlF8KBq",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "PATH_TO_SAMPLE_IMAGES_DIR = '/content/wind-turbine-detector/images/samples/chopped'\n",
        "SAMPLE_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_SAMPLE_IMAGES_DIR, \"*.*\"))\n",
        "\n",
        "for image_path in SAMPLE_IMAGE_PATHS:\n",
        "  image = Image.open(image_path)\n",
        "  plt.figure(figsize=(8,8))\n",
        "  plt.axis('off')\n",
        "  plt.imshow(image)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GQGbNpw1oHMf",
        "colab_type": "text"
      },
      "source": [
        "## Training\n",
        "Training will be performed on the 392 labeled images in the train image set and tested against the 80 labeled test images. A pre-trained model from the [TensorFlow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) is used as a starting point. In this notebook, the ssd_inception_v2_coco model is used based on its balance of accuracy and efficiency.  \n",
        "\n",
        "\n",
        "> *Note: Training will take ~2.5 hours. If you wish to bypass the training step, you can skip the \"Train model\" and \"Export trained wind turbine detector model\" sections and uncomment the second \"PATH_TO_FROZEN_GRAPH=\" line in the \"Inference\" section to use the provided pre-trained wind turbine detection model.*\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3so3G8MuYERG",
        "colab_type": "text"
      },
      "source": [
        "Install tensorflow 1.* -- tensorflow object detection api is not compatible with tensorflow 2.*"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YQQVbTzqVcE-",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "pip install tensorflow-gpu==1.*"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JNvjsjTFoKt4",
        "colab_type": "text"
      },
      "source": [
        "### Install all required libraries\n",
        "Further details on how to install and configure TensorFlow Object Detection API can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FuQCkr_oVSoR",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!apt-get install protobuf-compiler python-pil python-lxml python-tk\n",
        "!pip install Cython\n",
        "!pip install contextlib2\n",
        "!pip install jupyter\n",
        "!pip install matplotlib"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-l-vxv7eo4zZ",
        "colab_type": "text"
      },
      "source": [
        "### Clone TensorFlow Object Detection API repo"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fr1a5vJeVc3G",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!git clone --quiet https://github.com/tensorflow/models.git"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "10Dmf9yJpKgg",
        "colab_type": "text"
      },
      "source": [
        "### COCO API installation\n",
        "This is needed if you are interested in using COCO evaluation metrics."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lK-6jvEYg2i3",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!git clone https://github.com/cocodataset/cocoapi.git\n",
        "!cd cocoapi/PythonAPI; make; cp -r pycocotools /content/models/research/"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BRugsmFnpVmI",
        "colab_type": "text"
      },
      "source": [
        "### Protobuf compilation\n",
        "The Protobuf libraries provided in the TensorFlow Object Detection API repo must be compiled in order to use the framework."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zxuEP4WvpqBh",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%cd /content/models/research\n",
        "!protoc object_detection/protos/*.proto --python_out=."
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FynXe5k8pyGZ",
        "colab_type": "text"
      },
      "source": [
        "### Add Libraries to PYTHONPATH"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bbbnMD-PVmYe",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%set_env PYTHONPATH=$PYTHONPATH:/content/models/research:/content/models/research/slim"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DQg2GpyCqSZT",
        "colab_type": "text"
      },
      "source": [
        "### Setup and run TensorBoard\n",
        "TensorBoard provides a visualization of various quantitative metrics such as loss as well as a comparison between prediction vs. ground truth for a subset of images."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ySULkGfPf4U_",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%cd /content/wind-turbine-detector"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_Gaa9JqZoQt_",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip\n",
        "!unzip -o ngrok-stable-linux-amd64.zip"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CJWt_frFoXx2",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "LOG_DIR = 'training/'\n",
        "get_ipython().system_raw(\n",
        "    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'\n",
        "    .format(LOG_DIR)\n",
        ")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jE1M9kX3ozp9",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "get_ipython().system_raw('./ngrok http 6006 &')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xv-waeebsZRH",
        "colab_type": "text"
      },
      "source": [
        "#### Get TensorBoard link\n",
        "Click on the link to launch TensorBoard. It will update once the first checkpoint is saved. The plot of the \"loss_1\" scalar will provide the loss as a function of step, matching what is printed to the screen."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yzRpzLFao2FG",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "! curl -s http://localhost:4040/api/tunnels | python3 -c \\\n",
        "    \"import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])\""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h99J1SF8sdZv",
        "colab_type": "text"
      },
      "source": [
        "### Train model\n",
        "Train the wind turbine detector model using a modified model_main.py file which includes \"tf.logging.set_verbosity(tf.logging.INFO)\" following the import statements to output the loss every 100 steps. The model configuration is provided in wind-turbine-detector/training/ssd_inception_v2_coco_WTDetector.config. This configuration file uses all the default settings provided in the sample ssd_inception_v2_coco.config file except the following:\n",
        "\n",
        "*   num_classes: 1\n",
        "*   batch_size: 12\n",
        "*   fine_tune_checkpoint: \"pre-trained-model/model.ckpt\"\n",
        "*   train_input_reader: {\n",
        "  tf_record_input_reader {\n",
        "    input_path: \"annotations/train.record\"\n",
        "  }\n",
        "  label_map_path: \"annotations/label_map.pbtxt\"\n",
        "}\n",
        "    * *Note: The 'label_map.pbtxt' file required for training contains 1 class: item {\n",
        "id: 1\n",
        "name: 'wind turbine'\n",
        "}*\n",
        "*   eval_input_reader: {\n",
        "  tf_record_input_reader {\n",
        "    input_path: \"annotations/test.record\"\n",
        "  }\n",
        "  label_map_path: \"annotations/label_map.pbtxt\"\n",
        "  shuffle: false\n",
        "  num_readers: 1\n",
        "}  \n",
        "\n",
        "\n",
        "\n",
        "Additional data (image) augmentation was prescribed in the configuration file. Combining a vertical flip and a 90 degree rotation with the default horizontal flip, the training data can be extended to contain all possible wind turbine orientations. These operations help to generalize the model.\n",
        "*   data_augmentation_options {\n",
        "    random_vertical_flip {\n",
        "    }\n",
        "  }\n",
        "*   data_augmentation_options {\n",
        "    random_rotation90 {\n",
        "    }\n",
        "  }\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "53no6WzWBQ10",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!pip install tf_slim"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "X-7qjImCBbDs",
        "colab_type": "text"
      },
      "source": [
        "Train custom object detection model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Rg_XzuZNrTok",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!python model_main.py --pipeline_config_path=training/ssd_inception_v2_coco_WTDetector.config --model_dir=training/ --num_train_steps=20000 --alsologtostderr\n",
        "#!python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco_WTDetector.config # using legacy training code"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5W2X-Uvf7Sku",
        "colab_type": "text"
      },
      "source": [
        "### Export trained wind turbine detector model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BgrAVZ0WOKhO",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!python /content/models/research/object_detection/export_inference_graph.py \\\n",
        "    --input_type=image_tensor \\\n",
        "    --pipeline_config_path=training/ssd_inception_v2_coco_WTDetector.config \\\n",
        "    --output_directory=WTDetectorModel \\\n",
        "    --trained_checkpoint_prefix=training/model.ckpt-20000"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vBOJ3rs9pZwk",
        "colab_type": "text"
      },
      "source": [
        "## *Inference*\n",
        "\n",
        "Perform inference using the newly trained wind turbine detection model on the validation image set. This set of images was kept separate from the test and train image sets and will now be used to validate the accuracy of the model.   "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dNjT4MhXIR0q",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%cd /content/models/research/object_detection"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t1s3VA-UmIjq",
        "colab_type": "text"
      },
      "source": [
        "### Imports"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "beJT-03bIq0e",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import os\n",
        "import six.moves.urllib as urllib\n",
        "import sys\n",
        "import tarfile\n",
        "import tensorflow as tf\n",
        "import zipfile\n",
        "\n",
        "from collections import defaultdict\n",
        "from io import StringIO\n",
        "from matplotlib import pyplot as plt\n",
        "from PIL import Image\n",
        "from IPython.display import display"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7yovt9fe8f-9",
        "colab_type": "text"
      },
      "source": [
        "Import the object detection module"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pYMlQoOcdKa3",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%%bash \n",
        "cd /content/models/research\n",
        "pip install ."
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xnEGa9Rf8ghe",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from object_detection.utils import ops as utils_ops\n",
        "from object_detection.utils import label_map_util\n",
        "from object_detection.utils import visualization_utils as vis_util"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "z7u3Zdi7mVdb",
        "colab_type": "text"
      },
      "source": [
        "#### Object detection imports\n",
        "Here are the imports from the object detection module."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-B5qni4GhhIe",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from utils import label_map_util\n",
        "\n",
        "from utils import visualization_utils as vis_util"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CMg14pzhmcxw",
        "colab_type": "text"
      },
      "source": [
        "### Model preparation "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Qdpy94wImdoA",
        "colab_type": "text"
      },
      "source": [
        "#### Variables\n",
        "\n",
        "Any model exported using the \"export_inference_graph.py\" tool can be loaded here simply by changing \"PATH_TO_FROZEN_GRAPH\" to point to a new .pb file.  "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JElDuCKIhhLK",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Path to frozen detection graph. This is the actual model that is used for the object detection.\n",
        "PATH_TO_FROZEN_GRAPH = '/content/wind-turbine-detector/WTDetectorModel/frozen_inference_graph.pb'\n",
        "#PATH_TO_FROZEN_GRAPH = '/content/wind-turbine-detector/trainedWTDetector/frozen_inference_graph.pb' # Uncomment this line to run inference (without training) using provided pre-trained model\n",
        "\n",
        "# List of the strings that is used to add correct label for each box.\n",
        "PATH_TO_LABELS = '/content/wind-turbine-detector/annotations/label_map.pbtxt'"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vwO9XmZlmpDY",
        "colab_type": "text"
      },
      "source": [
        "#### Load a (frozen) TensorFlow model into memory"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xHau6ysLi2yv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "detection_graph = tf.Graph()\n",
        "with detection_graph.as_default():\n",
        "  od_graph_def = tf.GraphDef()\n",
        "  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:\n",
        "    serialized_graph = fid.read()\n",
        "    od_graph_def.ParseFromString(serialized_graph)\n",
        "    tf.import_graph_def(od_graph_def, name='')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "b0VnLMMvmtPp",
        "colab_type": "text"
      },
      "source": [
        "#### Loading label map\n",
        "Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to \"airplane\".  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZofMI3VLi-Vz",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)\n",
        "print(category_index)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AMwE_dYXmxPY",
        "colab_type": "text"
      },
      "source": [
        "#### Helper code"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qVvBib_QjDl9",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def load_image_into_numpy_array(image):\n",
        "  (im_width, im_height) = image.size\n",
        "  return np.array(image.getdata()).reshape(\n",
        "      (im_height, im_width, 3)).astype(np.uint8)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Xdt4jDREiqn2",
        "colab_type": "text"
      },
      "source": [
        "#### Inference function\n",
        "Perform inference one image at a time"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TYJ5UrIMiuh2",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def run_inference_for_single_image(image, graph):\n",
        "  with graph.as_default():\n",
        "    with tf.Session() as sess:\n",
        "      # Get handles to input and output tensors\n",
        "      ops = tf.get_default_graph().get_operations()\n",
        "      all_tensor_names = {output.name for op in ops for output in op.outputs}\n",
        "      tensor_dict = {}\n",
        "      for key in [\n",
        "          'num_detections', 'detection_boxes', 'detection_scores',\n",
        "          'detection_classes', 'detection_masks'\n",
        "      ]:\n",
        "        tensor_name = key + ':0'\n",
        "        if tensor_name in all_tensor_names:\n",
        "          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(\n",
        "              tensor_name)\n",
        "      if 'detection_masks' in tensor_dict:\n",
        "        # The following processing is only for single image\n",
        "        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])\n",
        "        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])\n",
        "        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.\n",
        "        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)\n",
        "        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])\n",
        "        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])\n",
        "        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(\n",
        "            detection_masks, detection_boxes, image.shape[0], image.shape[1])\n",
        "        detection_masks_reframed = tf.cast(\n",
        "            tf.greater(detection_masks_reframed, 0.5), tf.uint8)\n",
        "        # Follow the convention by adding back the batch dimension\n",
        "        tensor_dict['detection_masks'] = tf.expand_dims(\n",
        "            detection_masks_reframed, 0)\n",
        "      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')\n",
        "\n",
        "      # Run inference\n",
        "      output_dict = sess.run(tensor_dict,\n",
        "                             feed_dict={image_tensor: np.expand_dims(image, 0)})\n",
        "\n",
        "      # all outputs are float32 numpy arrays, so convert types as appropriate\n",
        "      output_dict['num_detections'] = int(output_dict['num_detections'][0])\n",
        "      output_dict['detection_classes'] = output_dict[\n",
        "          'detection_classes'][0].astype(np.uint8)\n",
        "      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]\n",
        "      output_dict['detection_scores'] = output_dict['detection_scores'][0]\n",
        "      if 'detection_masks' in output_dict:\n",
        "        output_dict['detection_masks'] = output_dict['detection_masks'][0]\n",
        "  return output_dict"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RJK8_5ixe3a8",
        "colab_type": "text"
      },
      "source": [
        "### Detection - Validation Images\n",
        "Detect wind turbines in validation image set. \n",
        "\n",
        "This section provides the validation images containing at least one wind turbine and corresponding bounding box(es)."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ImjGe0yPe6lC",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import os\n",
        "import glob\n",
        "\n",
        "PATH_TO_TEST_IMAGES_DIR = '/content/wind-turbine-detector/images/valid'\n",
        "TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, \"*.*\"))\n",
        "print(TEST_IMAGE_PATHS)\n",
        "\n",
        "# Size, in inches, of the output images.\n",
        "IMAGE_SIZE = (6, 6)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "odIH7bYjfe-s",
        "colab_type": "text"
      },
      "source": [
        "#### Perform inference on all validation images\n",
        "Loop over validation set and output images containing detected wind turbine with its location denoted with a bounding box."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yd7jvqMxfApD",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%matplotlib inline\n",
        "for image_path in TEST_IMAGE_PATHS:\n",
        "  image = Image.open(image_path)\n",
        "  # the array based representation of the image will be used later in order to prepare the\n",
        "  # result image with boxes and labels on it.\n",
        "  image_np = load_image_into_numpy_array(image)\n",
        "  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]\n",
        "  image_np_expanded = np.expand_dims(image_np, axis=0)\n",
        "  # Actual detection.\n",
        "  output_dict = run_inference_for_single_image(image_np, detection_graph)\n",
        "  # Visualization of the results of a detection.\n",
        "  vis_util.visualize_boxes_and_labels_on_image_array(\n",
        "      image_np,\n",
        "      output_dict['detection_boxes'],\n",
        "      output_dict['detection_classes'],\n",
        "      output_dict['detection_scores'],\n",
        "      category_index,\n",
        "      instance_masks=output_dict.get('detection_masks'),\n",
        "      use_normalized_coordinates=True,\n",
        "      line_thickness=8)\n",
        "  plt.figure(figsize=IMAGE_SIZE)\n",
        "  plt.axis('off')\n",
        "  plt.imshow(image_np)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mqU0ar1Jm58Q",
        "colab_type": "text"
      },
      "source": [
        "### Detection - Full NAIP Images\n",
        "Detect wind turbines in full size NAIP images (~4 miles square) using sliding window of size 300 x 300 pixels over the entire 5978 x 7648 pixel image. \n",
        "\n",
        "This section outputs the individual images containing at least one wind turbine and corresponding bounding box(es), the full NAIP image with all the wind turbines marked, and the latitude and longitude coordinates of all detected wind turbines. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WcYD09DjjR_1",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import os\n",
        "import glob\n",
        "\n",
        "PATH_TO_TEST_IMAGES_DIR = '/content/wind-turbine-detector/images/samples/orig'\n",
        "TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, \"*.jpg\"))\n",
        "print(TEST_IMAGE_PATHS)\n",
        "\n",
        "# Size, in inches, of the output images.\n",
        "IMAGE_SIZE = (8, 8)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EwAfAVqLgDpS",
        "colab_type": "text"
      },
      "source": [
        "#### Perform inference on full NAIP images\n",
        "For each full NAIP image, perform inference on sliding 300 x 300 pixel window. Output images containing detected wind turbine with its location denoted with a bounding box."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZgbO1Ov8j7rm",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "BBsWTs = np.empty((0, 4))\n",
        "detectedWTs=[]\n",
        "chipsize = 300\n",
        "for image_path in TEST_IMAGE_PATHS:\n",
        "  image = Image.open(image_path)\n",
        "  width, height = image.size\n",
        "  ii=0\n",
        "  for x0 in range(0, width, chipsize): # width\n",
        "    ii+=1\n",
        "    jj=0\n",
        "    for y0 in range(0, height, chipsize): # height\n",
        "      jj+=1\n",
        "      box = (x0, y0,\n",
        "             x0+chipsize,\n",
        "             y0+chipsize)\n",
        "      # the array based representation of the image will be used later in order to prepare the\n",
        "      # result image with boxes and labels on it.\n",
        "      image_np = load_image_into_numpy_array(image.crop(box))\n",
        "      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]\n",
        "      image_np_expanded = np.expand_dims(image_np, axis=0)\n",
        "      # Actual detection.\n",
        "      output_dict = run_inference_for_single_image(image_np, detection_graph)\n",
        "      BBsWTs=np.append(BBsWTs,output_dict['detection_boxes'][output_dict['detection_scores']>0.5],axis=0)\n",
        "      if len(output_dict['detection_scores'][output_dict['detection_scores']>0.5])>0:\n",
        "        # Visualization of the results of a detection.\n",
        "        vis_util.visualize_boxes_and_labels_on_image_array(\n",
        "            image_np,\n",
        "            output_dict['detection_boxes'],\n",
        "            output_dict['detection_classes'],\n",
        "            output_dict['detection_scores'],\n",
        "            category_index,\n",
        "            instance_masks=output_dict.get('detection_masks'),\n",
        "            use_normalized_coordinates=True,\n",
        "            line_thickness=8)\n",
        "        plt.figure(figsize=IMAGE_SIZE)\n",
        "        plt.axis('off')\n",
        "        for kk in range(len(output_dict['detection_scores'][output_dict['detection_scores']>0.5])):\n",
        "          plt.plot(chipsize*np.mean([BBsWTs[-1-kk][1],BBsWTs[-1-kk][3]]),chipsize*np.mean([[BBsWTs[-1-kk][0],BBsWTs[-1-kk][2]]]),'bo')\n",
        "          detectedWTs.append([image_path,ii,jj])\n",
        "        plt.imshow(image_np)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SlQByra45P0B",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# total number of wind turbines detected \n",
        "print(len(detectedWTs))\n",
        "print(detectedWTs)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h0krBKFEgiK8",
        "colab_type": "text"
      },
      "source": [
        "####Bounding box center location\n",
        "Calculate normalized coordinates of bounding box center. This will be used to plot each detected wind turbine on the NAIP images and to compute the latitude and longitude of each wind turbine."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JYIabjFp8GLq",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "centerBBs = np.empty((len(BBsWTs),2))\n",
        "ii=0\n",
        "for BBs in BBsWTs:\n",
        "  centerBBs[ii][:]=[np.mean([BBs[1],BBs[3]]),np.mean([[BBs[0],BBs[2]]])]\n",
        "  ii+=1"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LK-8KtshcUZZ",
        "colab_type": "text"
      },
      "source": [
        "####Plot wind turbine locations over NAIP image\n",
        "Plot marker for each detected wind turbine on NAIP images analyzed."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XlPLn9Mtjpgn",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "for image_path in TEST_IMAGE_PATHS:\n",
        "  image = Image.open(image_path)\n",
        "  plt.figure(figsize=(width/500,height/500))\n",
        "  plt.axis('off')\n",
        "  for ll in range(len(detectedWTs)):\n",
        "    if detectedWTs[ll][0]==image_path:\n",
        "      x_plot=chipsize*(detectedWTs[ll][1]-1+centerBBs[ll][0])\n",
        "      y_plot=chipsize*(detectedWTs[ll][2]-1+centerBBs[ll][1])\n",
        "      plt.plot(x_plot,y_plot,'ro')\n",
        "  plt.imshow(image)\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "InaVNI0_bVxj",
        "colab_type": "text"
      },
      "source": [
        "####Calculate wind turbine latitude and longitude\n",
        "Perform a series of 1D linear interpolations to determine latitude and longitude coordinates of each detected wind turbine."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pc0NwxvFqst2",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# initialize dictionaries for NAIP image corners\n",
        "#   y1,x1 are lat,long of upper left corner of image\n",
        "#   y2,x2 are lat,long of upper right corner of image\n",
        "#   y3,x3 are lat,long of lower right corner of image\n",
        "#   y4,x4 are lat,long of lower left corner of image\n",
        "x1={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':-94.8178888,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':-94.8180499}\n",
        "y1={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':41.3151166,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':42.6276193}\n",
        "x2={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':-94.7464999,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':-94.7464221}\n",
        "y2={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':41.3162222,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':42.6287332}\n",
        "x3={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':-94.7446638,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':-94.7444999}\n",
        "y3={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':41.2473638,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':42.5598722}\n",
        "x4={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':-94.8159777,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':-94.8160472}\n",
        "y4={'/content/wind-turbine-detector/images/samples/orig/m_4109442_se_15_1_20170709.jpg':41.246261,\n",
        "   '/content/wind-turbine-detector/images/samples/orig/m_4209426_ne_15_1_20170707.jpg':42.5587611}\n",
        "\n",
        "# loop over all the detected wind turbines and calculate the \n",
        "# latitude and longitude coordinates based on its location in\n",
        "# full NAIP images with respect to the image corners\n",
        "latlongWTs=[]\n",
        "for ll in range(len(detectedWTs)):\n",
        "  a5=chipsize*(detectedWTs[ll][1]-1+centerBBs[ll][0])\n",
        "  b5=chipsize*(detectedWTs[ll][2]-1+centerBBs[ll][1])\n",
        "\n",
        "  p_a5=a5/width\n",
        "  p_b5=b5/height\n",
        "\n",
        "  x_le=x1[detectedWTs[ll][0]]+p_b5*(x4[detectedWTs[ll][0]]-x1[detectedWTs[ll][0]])\n",
        "  x_re=x2[detectedWTs[ll][0]]+p_b5*(x3[detectedWTs[ll][0]]-x2[detectedWTs[ll][0]])\n",
        "  x_WT=x_le+p_a5*(x_re-x_le)\n",
        "\n",
        "  y_te=y1[detectedWTs[ll][0]]+p_a5*(y2[detectedWTs[ll][0]]-y1[detectedWTs[ll][0]])\n",
        "  y_be=y4[detectedWTs[ll][0]]+p_a5*(y3[detectedWTs[ll][0]]-y4[detectedWTs[ll][0]])\n",
        "  y_WT=y_te+p_b5*(y_be-y_te)\n",
        "  latlongWTs.append([y_WT,x_WT])\n",
        "\n",
        "# print out latitude and longitude coordinates for verification or plotting\n",
        "print(latlongWTs)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "u8QmWQhTrnJO",
        "colab_type": "text"
      },
      "source": [
        "## Summary\n",
        "\n",
        "The trained model accurately detects at least 15 out of 17 wind turbines in the validation image set with high probability. This represents an accuracy of ~90%. Higher accuracy would likely be achieved by using a larger set of images (train + test) as well as using a more accurate pre-trained model. Alternative models, including those with higher mAP, can be found at the [TensorFlow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Details of the trade-offs between speed, accuracy, and memory for various object detection model architectures (Faster RCNN, SSD, R-FCN) can be found in this [paper](https://arxiv.org/pdf/1611.10012.pdf), which can serve as a good starting point in determining which architecture is best for your application.\n",
        "\n",
        "Using the trained model, images encompassing a large area (~16 sq miles) were scanned for wind turbines. Once detected, the position of each wind turbine was plotted on the original full NAIP image and its position (latitude, longitude) was output. The comprehensive pipeline including training, validation, and inference could be applied to many other applications involving detection of objects from aerial or satellite images. "
      ]
    }
  ]
}