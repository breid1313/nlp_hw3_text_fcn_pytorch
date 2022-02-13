# Text classification using a fully-connected neural network

In this task you will learn how to structure your DL project and implement the same kind
of neural classifier as in the previous module.
But this time, we will also experiment with batch normalization, L2-regularization, dropout
and many more.

In the end, you will use [wandb sweeps](https://docs.wandb.com/sweeps) to perform
automatic hyperparameter optimization.

## Installation

To start working, you need to install this module to your system as editable module.
Editable (`-e`) means that as soon as you modify the files, `import nn_classifier`
will always import the last version of your code.

Why do we need to install this module at all? This is the easiest way to make `cli/*.py`
scripts and tests work.

```
pip install -e .
```

## Tasks

There are 7 tasks in this assignment.
All of them have numbers and we recommend you to solve them in this order,
but it is not always nesessary.

After you complete these tasks, you need to use wandb sweeps to find
the best hyperparameters for your model.
Your ultimate goal is to achieve a better accuracy than
a linear model from the previous assignment.
If you were unable to find the best hyperparameters using sweeps, analyse your results
and try to either find better hyperparameters manually or construct a better sweep config.

After this, look into your best architecture and think what else can be improved.

### Model (`nn_classifier/modelling.py`)

This file contains two tasks: to implement model `.__init__()` and `.forward()` methods.
Note that the implementation significantly differs from the previous assignment,
because now the architecture of the network (e.g. does it have batch norm or not)
depends on the `.__init__()` arguments.
We provide more information about it and specific directions and implementation tips in the
code.

### Training (`cli/train.py`)

This file contains 4 tasks. All of them are described in detail in the code,
this is a short summary:

1. If `--device` is not specified, your code should select
GPU if it is available and CPU instead.
1. Create optimizer with a specified learning rate and weight decay.
1. Use a provided function to evaluate the model at the end of every epoch.
1. Implement early stopping - a technique that allows to automatically stop model
   training based on the validation performance.

If you are interested in how ADAMW differs from ADAM and when this difference matters,
read this
[blog post](https://towardsdatascience.com/why-adamw-matters-736223f31b5d)
and the original
[research paper](https://arxiv.org/abs/1711.05101)
.

### Interaction (`cli/interact.py`)

It is good to see your test accuracy, but what is way more fun is to interact with the
model directly! In ths script you will write text preprocessing and learn how to load
saved models in PyTorch.

### Unittests

The first thing you should check when you modify the code is unittests.
The unittests are located in the `tests` directory and you can either run them in your
favorite IDE using graphical interface, or you can simply execute

```
pytest
```

In the root directory (the same directory as this README.md file).

Note that only modelling tasks and utils are covered by tests.
Feel free to write your own tests for cli utils.
We will talk about tests in more depth in the next modules.

The solved assignment has to pass all of the tests. **Do not modify the tests**,
only write the code between comments `# YOUR CODE STARTS` and `# YOUR CODE ENDS`.
All the requirements from the previous assignment including the requirement not to use loops 
unless it is explicitly stated are applied.

## Usage

To train a model, you can use `cli/train.py`

```
train.py

arguments:
  -h, --help            show this help message and exit
  --max-vocab-size
                        maximum size of the vocabulary
  --hidden-size
                        size of the intermediate layer in the network
  --use-batch-norm
  --dropout
  --batch-size
                        number of examples in a single batch
  --max-epochs
                        number of passes through the dataset during training
  --device              device to train on, use GPU if available by default
  --output-dir
                        a directory to save the model and config, do not save the model by default
  --wandb-project
                        wandb project name to log metrics to
```

To enter an interactive mode, where you can use a trained model to
predict the sentiment of the text you enter, you can use `cli/interact.py`

```
interact.py

arguments:
  -h, --help            show this help message and exit
  --model_dir
                        path to the directory with the model, tokenizer and a toml config
  --device DEVICE       device to train on, use GPU if available by default

```

## How to submit this task

Submission instructions will be available in Blackboard in a few days.
