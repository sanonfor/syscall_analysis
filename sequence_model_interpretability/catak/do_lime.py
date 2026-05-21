#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import json
import argparse
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import dill

import torch

from lime import lime_text

import sys
sys.path.append("../..")

from lstm_malware_detection.train import Model, CatakDataset

from tensorflow.keras.utils import to_categorical


def main(cmdLineArgs=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpointPath", "-p", type=str, default="../../lstm_malware_detection/results/2024-07-01_17-54-47_400-epochs/checkpoints/checkpoint_000400.pth")

    parser.add_argument("--dataDir", type=str, default="../../malware_api_class")
    parser.add_argument("--outBaseDir", type=str, default="./lime-results")

    parser.add_argument("--allowPadding", type=int, default=1)
    parser.add_argument("--collapseDuplicates", type=int, default=0)
    parser.add_argument("--maxCalls", type=int, default=100)
    parser.add_argument("--removeDuplicateExamples", "--rde", type=int, default=1)
    parser.add_argument("--validationOnly", type=int, default=1)
    parser.add_argument("--batchSize", "-b", type=int, default=10)

    parser.add_argument("--nfeatures", type=int, default=10, help="Maximum number of features to use in the explanation")
    parser.add_argument("--no-bow", action="store_false", dest="do_bow", help="Don't treat the sequence as a bag of words for the purpose of the explanation")
    parser.add_argument("--ntokens", type=int, default=1, help="Number of tokens to group together for interpretable components")
    # parser.add_argument("--results-mode", default="by-components")
    parser.add_argument("--results-mode", default="by-inputs")


    args = parser.parse_args(cmdLineArgs)

    timestampObj = datetime.datetime.now()
    timestampStr = timestampObj.strftime("%Y-%m-%d_%H-%M-%S")
    outDir = Path(args.outBaseDir) / timestampStr
    outDir.mkdir(parents=True)

    configDict = vars(args)
    with open(outDir/"config.json", 'w') as fjson:
        json.dump(configDict, fjson, indent=4)

    outpath = outDir / "results_log.txt"
    fout = open(outpath, 'w')
    def dprint(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=fout)


    checkpointPath = Path(args.checkpointPath)
    # encodingDirPath = Path("/homes/<username>/projects/systemCalls/pytorch_lstm_malware_detection/data/malware_api_class/processed")
    dataDir = Path(args.dataDir)
    encodingDirPath = dataDir / "processed"
    with open(encodingDirPath / "dataEncodingMap.json") as fjson:
        encodingDict = json.load(fjson) # string labels to ints
    with open(encodingDirPath / "labelsEncodingMap.json") as fjson:
        labelsDict = json.load(fjson)

    dprint(labelsDict)
    
    decodingDict = {v:k for k,v in encodingDict.items()}
    decodingDict = {**{0: "PAD"}, **decodingDict}
    encodingDict = {**{"PAD": 0}, **encodingDict}


    model = torch.load(
        checkpointPath,
        map_location=torch.device("cpu"),
    )

    if not args.validationOnly:
        raise NotImplementedError()
    
    if not args.removeDuplicateExamples:
        raise NotImplementedError()

    dataset = CatakDataset(
        data_repo_dir=dataDir,
        max_seq_len=args.maxCalls,
        train=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
    )

    Xs = []
    y_preds = []
    y_trues = []
    p_vecs = []
    # model.eval()
    for Xs_local, ys_local in tqdm(loader, desc="eval batches: "):
        p_vecs_local = model(Xs_local)
        p_vecs.extend(p_vecs_local)
        y_preds.extend(torch.argmax(p_vecs_local, axis=1).numpy())
        y_trues.extend([y.item() for y in ys_local])
        Xs.extend(Xs_local)


    # p_vecs_tensor = p_vecs
    p_vecs = torch.stack(p_vecs)
    Xs = torch.stack(Xs)


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=y_trues, y_pred=y_preds)
    # dprint("Confusion matrix (full dataset): ")
    # dprint(cm)

    def words_to_tokens(words, padlength=None, padval=0):
        tokens = [encodingDict[word] for word in words]
        if padlength is not None and len(tokens) < padlength:
            tokens = [padval]*(padlength-len(tokens)) + tokens
        return tokens

    def tokens_to_text_list(seq):
        if isinstance(seq, torch.TensorType):
            seq = seq.numpy()
        tokens = list(seq)
        prefix = [0] * (ntokens - (len(tokens) % ntokens))
        tokens = prefix + tokens
        token_groups = np.array(tokens)
        token_groups = token_groups.reshape(-1, ntokens)
        text_list = [sep.join(decodingDict[t] for t in g) for g in token_groups]
        return text_list

    sep = ";"
    ntokens = args.ntokens
    if ntokens == 1:
        def tokens_to_text(seq):
            text = " ".join(decodingDict[token] for token in seq.numpy())
            return text

        def text_to_tokens(text, **kwargs):
            words = text.split()
            tokens = words_to_tokens(words, **kwargs)
            return tokens
    else:
        def tokens_to_text(seq):
            text_list = tokens_to_text_list(seq)
            text = " ".join(text_list)
            return text

        def text_to_tokens(text, **kwargs):
            groups = text.split()
            words = [w for g in groups for w in g.split(sep)]
            tokens = words_to_tokens(words, **kwargs)
            return tokens

    def f_model(text_batches):
        tokens_batches = []
        padlength = None
        for text in text_batches:
            tokens = text_to_tokens(text, padlength=padlength)
            tokens_batches.append(tokens)
            if padlength is None:
                padlength = len(tokens)
        tokens_batches = torch.tensor(tokens_batches)
        with torch.no_grad():
            output = model(tokens_batches)
        return output.detach().cpu().numpy()
    

    class_names = list(labelsDict.keys())
    do_bow = args.do_bow
    dprint(f"Warning:  Bag of words is {do_bow}")
    explainer = lime_text.LimeTextExplainer(
        class_names=class_names,
        bow=do_bow,
        mask_string="PAD",
        split_expression=r"\s+",
    )

    for idx in tqdm(range(len(Xs)), desc="Eval idxs"):
        seq = Xs[idx]
        text = tokens_to_text(seq)

        with torch.no_grad():
            p_model = model(seq.unsqueeze(0)).squeeze()
            p_model = p_model.cpu().numpy()

        to_explain = list(range(len(class_names))) # labels to explain
        exp = explainer.explain_instance(text, f_model, num_features=args.nfeatures, labels=to_explain)

        dprint()
        gt = y_trues[idx]
        pred = y_preds[idx]
        dprint(f"Explaining instance with idx={idx}; gt={class_names[gt]}; pred={class_names[pred]}")
        if args.results_mode == "by-components":
            inds = list(range(len(class_names)))
            inds.remove(gt)
            prefix = [gt]
            if pred != gt:
                inds.remove(pred)
                prefix.append(pred)
            inds = prefix + inds
            for icls in inds:
                name = class_names[icls]
                dprint(f"\t\tExplanation for class {name} (p = {p_model[icls]})")
                # dprint('\n'.join(map(lambda s: "\t\t"+str(s), exp.as_list(label=icls))))
                tups = exp.as_list(label=icls)
                for tup in tups:
                    words, val = tup
                    new_tup = (" ".join(words.split(sep)), val)
                    dprint("\t\t", new_tup, sep=None) 
        elif args.results_mode == "by-inputs":
            icls = pred
            name = class_names[icls]
            dprint(f"\t\tExplanation for class {name} (p = {p_model[icls]})")
            # break input into text blocks of length ntokens
            text_list = tokens_to_text_list(seq)
            vals_list = [0]*len(text_list)
            # search for occurrences of interpretable components in text blocks
            tups = exp.as_list(label=icls)
            for tup in tups:
                words, val = tup
                for i_text, sub_text in enumerate(text_list):
                    if sub_text == words:
                        vals_list[i_text] = val
            
            for words, val in zip(text_list, vals_list):
                if val == 0:
                    val = ""
                new_tup = (" ".join(words.split(sep)), val)
                dprint("\t\t", new_tup, sep=None) 

        else:
            raise ValueError(f"Unknown results_mode: {args.results_mode}")
        dprint()
        
        fout.flush()


    dprint()
    dprint(f"Saved results to {outDir}")
    fout.close()



if __name__ == '__main__':
    main()


