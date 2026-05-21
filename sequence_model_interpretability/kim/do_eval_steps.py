import argparse
from pathlib import Path
import json
import datetime
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras.utils.np_utils as np_utils
from simian_lstm.data import Dataset
from simian_lstm.model import Model
from simian_lstm.classifier import ModelClassifier


def main(cmdLineArgs=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="model file to load")
    parser.add_argument("--dataset", type=str, default="adfa-classic")
    parser.add_argument("--trace-type", type=str, default="attack", choices=["attack", "test"])
    parser.add_argument("--lim", type=int, default=1000)
    parser.add_argument("--vocab-size", type=int, default=341)
    parser.add_argument("--window-size", type=int, default=19)
    args = parser.parse_args(cmdLineArgs)

    vocab_size = args.vocab_size
    window_size = args.window_size


    timestampStr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outputDir = Path(".") / "results" / "eval_steps" / timestampStr
    outputDir.mkdir(parents=True)

    with open(outputDir / "config.json", "w") as fjson:
        json.dump(vars(args), fjson, indent=4)


    # load model
    adfa = Dataset.load(args.dataset)
    base_path = Path("results")

    if args.model is None:
        if not args.dataset.lower().startswith("adfa"):
            raise NotImplementedError()
        tag = "adfa-classic-large"
        model = Model.load_tag(base_path, tag)[0]  # returns a list of models matching the tag
    else:
        model_path = args.model
        # raw_model = tf.keras.models.load_model(model_path)
        # model = Model(raw_model, sequence_length=window_size)
        model = Model.load(base_path, model_path)

    dataset = adfa
    traces = dataset.matching_traces(args.trace_type) #, **match_criteria, **kwargs)
    # positives = dataset.matching_traces("attack") #, **match_criteria, **kwargs)
    # negatives = dataset.matching_traces("test")#, **match_criteria, **kwargs)
    if args.lim:
        traces = traces[:args.lim]


    fresults = open(outputDir / "results.txt", 'w')

    classifier = ModelClassifier(model, sequence_length=window_size, batch_size=1024)
    for i, trace in enumerate(tqdm(traces, desc="traces:  ")):
        if len(trace.trace) > window_size + 1:
            likelihoods = classifier.step_likelihoods(trace)
            likelihoods = list(tf.stack(likelihoods).numpy())
        else:
            likelihoods = [0]
        pre = f"{trace.file}"
        print(pre, *likelihoods, file=fresults, flush=True)

    fresults.close()

    print(f"Results written to {str(outputDir)}")


if __name__ == '__main__':
    main()
