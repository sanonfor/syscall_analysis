# SIMIAN Unsupervised Anomaly Detection

This repository contains our data and tools for experimenting with host-based anomaly detection in system call traces using LSTM and Transformer-based language models.

## Prerequisites

* Python 3.8

A GPU with Tensorflow-supported drivers is strongly recommended, but not strictly necessary.

## Setup

``` sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Prelude

When working with this codebase, it is convenient to import the prelude, which contains a collection of common settings
and imports that are handy for almost every use case.

```python
from simian_lstm.prelude import *
```

## Datasets

This repository includes two datasets, the [ADFA-LD](https://research.unsw.edu.au/projects/adfa-ids-datasets)
Linux Dataset, and a cleaned up version of
[this cuckoo trace dataset](https://www.reddit.com/r/datasets/comments/exhy38/malware_and_benign_windows_pe_cuckoo_reports/)
posted on reddit.

The `simian_lstm.data.Dataset` class handles loading and querying the datasets for training and test traces. To load a dataset,

``` python
dataset_name = "adfa"
data = Dataset.load(dataset_name)
```

The loader will search for text files containing sequences of integers under
`simian_lstm/data/[dataset_name]/[train/validation/test/attack]`, as well as a file called
`simian_lstm/data/[dataset_name]/feature_size.txt` which contains a single integer that is
at least one greater than the maximum integer in the dataset.

Importing the prelude automatically loads ADFA-LD as `adfa` and the Reddit Windows
dataset as `windows_data`.

## Training

To train a selection of models, use the `train(dataset, tag, experiment)` utility function defined in the prelude.

```python
from simian_lstm.prelude import *
train(
    dataset=adfa,
    tag="kim-replication",
    experiment={
        "embedding": -1,
        "layers": [[200], [400], [400, 400]],
        "learning_rate": 0.0001,
        "dropout": 0.5,
        "clipnorm": 10.0,
        "epochs": 30,
        "patience": 5,
    },
    model_version=2
)
```

Keys in the `experiment` dictionary refer to fields of the `HyperParameters` class, which are documented there. Keys
with list values are treated as choices, and a model will be trained for each permutation of possible choices. For example, in the above code, three separate models will be trained. One will have a single 200-paramter LSTM layer, one will have a single 400-paramter layer, and one will have two stacked 400-parameter layers.

The `tag` value gives a human-readable handle that can be used to refer back to the models trained this way. Models are
also given a unique identity based on a hash of their tag and parameters called a hashcode. If a model with a given
hashcode has already been trained, it will be skipped.

Trained models are stored in the `results` directory.

## Calling a Model

To load a model,

```python
from simian_lstm.prelude import *

models = Model.load_tag(base_path, "model-tag")
```

This will return a list of all models with that tag.

Models are called with an input tensor of shape `(batches,sequence_length)`, and output a tensor of shape
`(batches,feature_size)`, where each element of the `feature_size` dimension corresponds to the predicted probability
of the call at that index coming next in the sequence.

Example call:

```python
import tensorflow as tf
test_data = tf.convert_to_tensor([[1,2,3,4]])
model(test_data)
```

## Classifiers

To build a classifier from a model,

```python
from simian_lstm.classifier import ModelClassifier
classifier = ModelClassifier(model)
```

Because classifying traces can take a while, we strongly recommend using our caching layer:

```python
cache_path = Path("results/cache.db")
cached_classifier = CachedClassifier.of_model(model, cache_path)
```

Classifiers expect input in the form of `SyscallTrace` objects, which can come from a dataset or be constructed
directly from numpy arrays:

```python
import numpy as np
from simian_lstm.data import SyscallTrace
trace = SyscallTrace.from_seq(np.array([1,2,3,4]))
cached_classifier.likelihood(trace)
```

## Evaluation

Our prelude includes a handy `evaluate` function which handles the entire process of evaluating a model or set of models. It takes care of loading the models, setting up classifiers, and outputs a set of ROC curve objects, as well as displaying a graph of the comparative ROCs (if run in a Jupyter notebook).

```python
from simian_lstm.prelude import *
evaluate("model-tag")
```

The Windows dataset contains additional metadata that allows traces to be grouped by PID or Cuckoo report.
To calculate a ROC of grouped traces,

```python
from simian_lstm.prelude import *
evaluate("model-tag", group_by="pid")
```

## Default Outputs

Trained models will be saved to the `results` directory, labeled by hashcode. A saved model consists of a
`{hashcode}.json` file with metadata and a `{hashcode}.model` directory containing the information Tensorflow
needs to reconstruct themodel.

Graphs and figures are saved to `results/figures` in PDF format.

A cache of `(classifier, trace) -> score` mappings are stored in cache.db. In this pairing, the classifier
is identified by a hash of its parameters as returned by the `Classifier.hashcode` method.
