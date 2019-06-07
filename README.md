# Wind Turbine Object Detection from Aerial Imagery Using TensorFlow Object Detection API and Google Colab
*Google Colab notebook to detect wind turbines from aerial images*

This repo contains a Jupyter notebook and supporting files to train a wind turbine object detector using [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). The notebook is run in [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) which provides a free virtual machine with TensorFlow preinstalled and access to a GPU. This simplifies the setup process required to start using TensorFlow for interesting things like object detection. Coupling Google Colab with the open source TensorFlow Object Detection API provides all the tools necessary to train a custom object detection model. In this repo, wind turbines are detected from aerial images taken over west-central Iowa. The full pipeline from training to inference is contained in the notebook with detailed explanations for each step in the process. This can serve as a tutorial for those interested in training their own custom object detection model. The figure below demonstrates a typical input image and the resulting output images with each wind turbine properly detected and classified.

Launch the notebook in Google Colab by clicking on this badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbborkowski/wind-turbine-detector/blob/master/WindTurbineDetector.ipynb)
> *Note: After launching the notebook, verify that "Python2" and "GPU" are selected for the "Runtime type" and "Hardware accelerator" in Runtime -> Change runtime type.*

![](WindTurbineDetectorFigure_5.png)
> *Note: Original aerial images obtained from [National Agriculture Imagery Program (NAIP) database](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/)*
