# Pytorch-lightning cli MNIST example
This repository provides a simple example on how to use pytorch-lighning CLI interface to train a classifier on MNIST dataset.
Lightning-CLI is a powerful framework for training neural networks with pytorch. 
It considerably reduces the need for a boilerplate code.
Lightning-CLI implement a configurable command line tool for pytorch-lightning. It allows you to control the training parameters (learning rate, batch size, optimizer, number of GPUs...) either with config files or with the command line.

This code do not cover the full set of functionality of LightningCli. For more details, have a read at the official documentation: 
- https://lightning.ai/docs/pytorch/stable/ (pytorch-lightning)
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html (pytorch-lightning CLI)

## Installation
pytorch-lightning library is needed  to run this code. 
You can install it in a virtual environment as follows:
```
python3 -m venv env
source env/bin/activate
pip install pytorch-lightning[extra]==2.0.2 torchvision==0.15.2
```

<!-- ## Visualizing the available arguments
You can display all the available configurable arguments by running:
```
python main.py fit --print_config
```
To create your own config file, 
you can run:
```
python main.py fit --print_config > config/myconfig.yaml
```
and edit config/myconfig.yaml with chosen parameters. -->


## Training with a config file
All the training parameters can be controlled by a config file.
Have look in config/defaults.yaml for an example of config file.
 To start the training, run:
```
python main.py -c config/default.yaml fit
```
 Config file argument can be overwritten by using flag in the command line.
 For instance, to change the batch size and the number of gpus, run:

```
python main.py -c config/default.yaml fit --trainer.gpus 2 --data.batch_size 16
```
## logging 
The training is logged by default with tensorboard.
To visualize training logs, run:

```
tensorboard --logdir lightning_logs
```