# DeePyMoD

## Deep learning based model discovery for ODEs and PDEs

DeePyMoD is a PyTorch-based implementation of the DeepMoD algorithm for model discovery of PDEs and ODEs. We use a neural network to model our dataset, build a library of possible terms from the networks output and employ sparse regression to find the PDE underlying the dataset. More information can be found in our paper: [arXiv:1904.09406](http://arxiv.org/abs/1904.09406) 

**What's the use case?** Classical Model Discovery methods struggle with elevated noise levels and sparse datasets due the low accuracy of numerical differentiation. DeepMoD can handle high noise and sparse datasets, making it well suited for model discovery on actual experimental data.

**What types of models can you discover?** DeepMoD can discover non-linear, multi-dimensional and/or coupled ODEs and PDEs. See our paper and the examples folder for a demonstration of each.

**How hard is it to apply it to my data?** Not at all! We've designed the code to be accessible without having in-depth knowledge of deep learning or model discovery. You can load in the data, train the model and get the result in a few lines of code. We include a few notebooks with examples in the examples folder. Feel free to open an issue if you need any additional help.

**How do I modify the code?** We provide two interfaces, an object-based and functional-based one. The object-based interface is simply a wrapper around the functional one. The code has been modularly designed and is well documented, so you should be able to plug-in another training regime, cost function or library function yourself pretty easily.

# Features

* **Many example notebooks** We have implemented a varyity of examples ranging from 2D Advection Diffusion, Burgers' equation to non-linear, higher order ODE's If you miss any example, don't hesitate to give us a heads-up.

* **Extendable** DeePyMoD is designed to be easily extendable and modifiable. You can simply plug in your own cost function, library or training regime.

* **Automatic library** The library and coefficient vectors are automatically constructed from the maximum order of polynomial and differentiation. If that doesn't cut it for your use case, it's easy to plug in your own library function.

* **Extensive logging** We provide a simple command line logger to see how training is going and an extensive custom Tensorboard logger.

* **Fast** Depending on the size of the data-set DeepMoD, running a model search with DeepMoD takes of the order of minutes/ tens of minutes on a standard CPU. Running the code on GPU's drastically improves performence. 

# How to install
We provide two ways to use DeePyMoD, either as a package or in a ready-to-use Docker container. 

## Package
DeePyMoD is released as a pip package, so simply run 

``` pip install DeePyMoD```

to install. Alternatively, you can clone the 
We currently provide two ways to use our software, either in a docker container or as a normal package. If you want to use it as a package, simply clone the repo and run:

```python setup.py install```


## Container
A GPU-ready Docker image can also be used. Once you've cloned the repo, go into the config folder and run:

```./start_notebook.sh```

This pulls our lab's standard docker image from dockerhub, mounts the project directory inside the container and starts a jupyterlab server which can be accessed through localhost:8888. You can stop the container by running the stop_notebook script.  This will stop the container; next time you run start_notebook.sh it will look if any containers from that project exist and restart them instead of building a new one, so your changes inside the container are maintained.





