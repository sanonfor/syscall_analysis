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

import sys
sys.path.append("../..")

# from plmodule.LstmClassifierLitModule import LstmClassifierLitModule
# from pldatamodule.catakPLDataModule import CatakPLDataModule

from lstm_malware_detection.train import Model, CatakDataset

# import keras.utils.np_utils as np_utils
from tensorflow.keras.utils import to_categorical

# from clearml import Task

def main(cmdLineArgs=None):
    # task = Task.init(project_name="jjc-pl-lstm-classifier/analyze", task_name="check_catak")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpointPath", "-p", type=str, default="../../lstm_malware_detection/results/2024-07-01_17-54-47_400-epochs/checkpoints/checkpoint_000400.pth")


    # plDir = Path("..")
    # plLogPath = Path("../lightning_logs/version_18/")
    # eventsPath = sorted(plLogPath.glob("events.out*"))[-1]
    # checkpointPath = sorted((plLogPath/"checkpoints").glob("*"))[-1]

    parser.add_argument("--dataDir", type=str, default="../../malware_api_class")
    parser.add_argument("--outBaseDir", type=str, default="./deepaid-results")

    parser.add_argument("--allowPadding", type=int, default=1)
    parser.add_argument("--collapseDuplicates", type=int, default=0)
    parser.add_argument("--maxCalls", type=int, default=100)
    parser.add_argument("--removeDuplicateExamples", "--rde", type=int, default=1)
    parser.add_argument("--validationOnly", type=int, default=1)
    parser.add_argument("--batchSize", "-b", type=int, default=10)

    parser.add_argument("--skipMyDeepaid", "--skipmd", type=int, default=0)
    parser.add_argument("--multiSteps", type=int, default=0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--maxIter", type=int, default=100)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optim", type=str, default="adam")


    args = parser.parse_args(cmdLineArgs)

    timestampObj = datetime.datetime.now()
    timestampStr = timestampObj.strftime("%Y-%m-%d_%H-%M-%S")
    outDir = Path(args.outBaseDir) / timestampStr
    outDir.mkdir(parents=True)

    configDict = vars(args)
    with open(outDir/"config.json", 'w') as fjson:
        json.dump(configDict, fjson, indent=4)


    checkpointPath = Path(args.checkpointPath)
    # encodingDirPath = Path("/homes/<username>/projects/systemCalls/pytorch_lstm_malware_detection/data/malware_api_class/processed")
    dataDir = Path(args.dataDir)
    encodingDirPath = dataDir / "processed"
    with open(encodingDirPath / "dataEncodingMap.json") as fjson:
        encodingDict = json.load(fjson)
    with open(encodingDirPath / "labelsEncodingMap.json") as fjson:
        labelsDict = json.load(fjson)

    print(labelsDict)
    
    decodingDict= {v:k for k,v in encodingDict.items()}
    if args.allowPadding:
        decodingDict = {**{0: "PAD"}, **decodingDict}


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
    for Xs_local, ys in tqdm(loader, desc="eval batches: "):
        p_vecs_local = model(Xs_local)
        p_vecs.extend(p_vecs_local)
        y_preds.extend(torch.argmax(p_vecs_local, axis=1).numpy())
        y_trues.extend([y.item() for y in ys])
        Xs.extend(Xs_local)


    # p_vecs_tensor = p_vecs
    p_vecs = torch.stack(p_vecs)
    Xs = torch.stack(Xs)


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=y_trues, y_pred=y_preds)
    # print("Confusion matrix (full dataset): ")
    # print(cm)

    import sklearn



    # ## Run model with embedding implemented as matrix multiplication
    nFeatures = len(decodingDict)
    # featureDesc = {k:str(k) for k in range(nFeatures+1)} 
    featureDesc = decodingDict

    if not args.skipMyDeepaid:

        myModel = MyModel(model, vocab_size=nFeatures)

        my_y_preds = []
        my_p_vecs = []
        # myModel.eval()
        for Xs_local, ys in tqdm(loader, desc="myModel eval batches: "):
            seq = list(to_categorical(Xs_local.detach().numpy(), num_classes=nFeatures))
            seq = torch.from_numpy(np.array(seq, dtype=float)).float()
            p_vecs_local = myModel(seq)
            my_p_vecs.extend(p_vecs_local)
            my_y_preds.extend(torch.argmax(p_vecs_local, axis=1).numpy())

        my_cm = confusion_matrix(y_true=y_trues, y_pred=my_y_preds)
        # print("[myModel] Confusion matrix (full dataset): ")
        # print(my_cm)

        # print("my_cm - cm:")
        # print(my_cm - cm)

    # print("cm row sums")
    # print(np.sum(cm, axis=-1))

    # ## compare softmaxed outputs from the two models

    # for i, (p_vec, my_p_vec) in enumerate(zip(p_vecs, my_p_vecs)):
    #     print(f"{i=}\n   {p_vec=}\n{my_p_vec=}\n")
    #     if i > 100:
    #         break


    diffVal = 0 if args.allowPadding else 1
    # gtClass = 0
    # predClass = 1
    # print(f"{gtClass=}\n{predClass=}\n", file=None)

    # # ## Attempt to use deepaid
    # print("\n\n####  Attempt to use DeepAID  ####\n\n")


    # sys.path.append("../../deepaid/DeepAID")



    # doWrite = True
    # if doWrite:
    #     fpath = outDir / "results.txt"
    #     fResults = open(fpath, "w")
    #     print(f"Saving results to {fpath.resolve()}")
    # else:
    #     fResults = None


    # print(f"{gtClass=}\n{predClass=}\n", file=fResults)

    # for i, (X, y, yhat, ps) in enumerate(zip(Xs, y_trues, y_preds, p_vecs)):
    #     if y != gtClass or yhat != predClass:
    #         continue
    #     print(f"Index={i}", file=fResults)
            
    #     """Step 1: Load your model"""
    #     # model
    #     # model.one_hot = True

    #     """Step 2: Find an anomaly you are interested in, and convert to the right format"""
    #     seqPlusLabel_oneHot = list(np_utils.to_categorical((X-diffVal).detach().numpy(), num_classes=nFeatures))

    # #     *seq, label = seqPlusLabel_oneHot
    # #     seq = torch.from_numpy(np.array(seq, dtype=float))
    # #     label = torch.from_numpy(np.array(label, dtype=int))

    #     seq = seqPlusLabel_oneHot
    #     label = gtClass
    #     seq = torch.from_numpy(np.array(seq, dtype=float)).float()
    #     label = torch.from_numpy(np.array(label, dtype=int))
    #     label = label.unsqueeze(0)


    #     anomaly_timeseries = X.detach().numpy()
    #     #seq, label, anomaly_timeseries = deeplogtools_SeqFormatAnomalies(model, abnormal_data, num_candidates=9)(index=idx)
    #     # print(seq.shape,label.shape)

    #     """Step 3: Create a DeepAID Interpreter"""
    #     from deepaid.interpreters.timeseries_onehot import UniTimeseriesAID
    #     my_interpreter = UniTimeseriesAID(myModel, feature_desc=decodingDict, class_num=nFeatures)

    #     """Step 4: Interpret your anomaly and show the result"""
    #     interpretation = my_interpreter(seq, label, file=fResults)
    #     my_interpreter.show_table(anomaly_timeseries, interpretation, file=fResults)

    #     if not doWrite:
    #         break

    # if doWrite:
    #     fResults.close()


    doWrite = True
    if doWrite:
        fpath = outDir / "my_results.txt"
        fResults = open(fpath, "w")
        fjsonl = open(outDir / "summaries.jsonl", "w")
        print(f"Saving results to {fpath.resolve()}")
    else:
        fResults = None
    
    def dprint(*args, **kwargs):
        print(*args, **kwargs, file=fResults)

    dprint("Confusion matrix:")
    dprint(cm)

    import sklearn
    precisions = sklearn.metrics.precision_score(y_trues, y_preds, average=None, zero_division=0)
    recalls = sklearn.metrics.recall_score(y_trues, y_preds, average=None, zero_division=0)
    f1s = sklearn.metrics.f1_score(y_trues, y_preds, average=None, zero_division=0)

    avgStr = "weighted"
    avg_precisoin = sklearn.metrics.precision_score(y_trues, y_preds, average=avgStr, zero_division=0)
    avg_recall = sklearn.metrics.recall_score(y_trues, y_preds, average=avgStr, zero_division=0)
    avg_f1 = sklearn.metrics.f1_score(y_trues, y_preds, average=avgStr, zero_division=0)

    # scoresDict = {"precision": precisions, "recall": recalls, "f1": f1s}
    # df = pd.DataFrame(scoresDict)

    scoresDict = {
        "precision": [avg_precisoin]+list(precisions),
        "recall": [avg_recall]+list(recalls),
        "f1": [avg_f1]+list(f1s)
    }
    scoresDf = pd.DataFrame(scoresDict, index=["Weighted Avg"]+list(labelsDict.keys()))
    dprint()
    dprint(scoresDf)
    dprint()


    if args.skipMyDeepaid:
        if doWrite:
            fResults.close()
        return

    #### My own reimplementation ####
    print("\n\n####  Take deepaid steps using my reimplementation ####\n\n")

    # print(f"{gtClass=}\n{predClass=}\n", file=fResults)

    # model.one_hot = False

    ### replace embedding with one-hot times weight tensor
    model.one_hot = False
    myModel = MyModel(model, vocab_size=nFeatures)
    interp = MyInterp(model=myModel, vocab_size=nFeatures, lr=args.lr, optim=args.optim, k=args.k, max_iter=args.maxIter)
    # interp = MyInterp(model=model)


    summaryDicts = []
    jsonDicts = []

    for i, (X, y, yhat, ps) in enumerate(zip(tqdm(Xs, desc="examples:  "), y_trues, y_preds, p_vecs)):
        gtClass = int(y)
        predClass = int(yhat)
        if y != gtClass or yhat != predClass:
            continue
        dprint(f"Index={i}", flush=True)
        X_original = X
        seq = list(to_categorical((X-diffVal).detach().numpy(), num_classes=nFeatures))
        seq = torch.from_numpy(np.array(seq, dtype=float)).float()
        X = seq

        nSteps = 0

        if not args.multiSteps:
            X_star, loss_star, loss0 = interp.do_full_step(X, target_ind=gtClass)
        else:
            X_star = X
            loss_star = None
            while nSteps <= args.maxIter:
                X_star_new, n_iter, loss_new, loss0_new, *_ = interp.do_multi_steps(X_star, X, target_ind=gtClass, file=fResults)
                if loss_star is None or loss_new < loss_star:
                    if loss_star is None:
                        loss0 = loss0_new
                    loss_star = loss_new
                    X_star = X_star_new
                nSteps += n_iter

        X_flat = torch.argmax(X, dim=-1) + diffVal
        X_callsList = []
        for callInd in X_flat:
            callInd = int(callInd)
            callStr = decodingDict[callInd]
            X_callsList.append(callStr)

        X_star_flat = torch.argmax(X_star, dim=-1) + diffVal
        diff = X_star_flat - X_flat
        istar = torch.argmax(torch.abs(diff))
        dprint(f"Final loss:  {loss_star}")
        dprint(f"({X_flat[istar]} ({X_callsList[istar]}) -> {X_star_flat[istar]} ({decodingDict[int(X_star_flat[istar])]}) at position {istar})")

        # dprint(np.array(X_callsList))
        # dprint(X_flat)
        # dprint("-->")
        # dprint(X_star_flat)
        # dprint("diff: (second - first)")
        # dprint(diff)


        # {
        #     "inds": list(X_flat),
        #     "calls": [decodingDict[i] for i in X_flat],
        #     "inds_star": list(X_star_flat),
        #     "calls_star": [decodingDict[i] for i in X_star_flat],
        # }

        ps_star = myModel(X_star.unsqueeze(0))[0]
        predClass_star = torch.argmax(ps_star)

        summaryDict = {
            "index": i,
            "gt_class": gtClass,
            "pred_class": predClass,
            "pred_class_star": predClass_star.item(),
            "p_gt": ps[gtClass].item(),
            "p_gt_star": ps_star[gtClass].item(),
            "loss0": loss0.item(),
            "loss_star":  loss_star.item(),
        }
        if args.k == 1:
            callsDict = {
                "istar": istar.item(),
                "call_ind": X_flat[istar].item(),
                "call_ind_star": X_star_flat[istar].item(),
                "call": X_callsList[istar],
                "call_star": decodingDict[int(X_star_flat[istar])],
            }
            summaryDict = {**summaryDict, **callsDict}
        
        callsDicts = []
        for istar, (x_flat, x_star_flat) in enumerate(zip(X_flat, X_star_flat)):
            if x_star_flat == x_flat:
                continue
            
            callsDict = {
                "index": i,
                "istar": istar,
                "call_ind": X_flat[istar].item(),
                "call_ind_star": X_star_flat[istar].item(),
                "call": X_callsList[istar],
                "call_star": decodingDict[int(X_star_flat[istar])],
            }
            callsDicts.append(callsDict)
        
        summaryDicts.append(summaryDict)
        jsonDict = summaryDict.copy()
        jsonDict["replacedCalls"] = callsDicts
        json.dump(jsonDict, fjsonl, indent=4)
        jsonDicts.append(jsonDict)




        if not doWrite:
            break
        else:
            summaryDf = pd.DataFrame(summaryDicts)
            with open(outDir / "summary.dill", 'wb') as fdill:
                dill.dump(summaryDf, fdill)
            
            with open(outDir / "summary.json", "w") as fjson:
                json.dump(jsonDicts, fjson, indent=4)


    if doWrite:
        fResults.close()
        fjsonl.close()


    # df = summaryDf
    # df = df[df.gt_class != df.pred_class]
    # df = df[df.gt_class == df.pred_class_star]
    # df = df[df.gt_class == 0]

    # print(f"Fount {len(df)} examples")


class MyEmbedding(torch.nn.Module):
    """
    Convert a pytorch embedding to matrix multiplication
    
    The previous embedding takes integer token indices,
    while the new one takes one-hots
    """
    def __init__(self, embed, vocab_size):
        super().__init__()
        self.old_embed = embed
        self.vocab_size = vocab_size
        self.embed_dim = embed.embedding_dim

        indices = np.arange(0, vocab_size)
        indices = torch.from_numpy(indices)
        self.W = self.old_embed(indices).T.float()

        self.dense = torch.nn.Linear(self.vocab_size, self.embed_dim)
        params = torch.nn.Parameter(self.W)
        self.dense.weight = params

    def forward(self, X):
        return self.dense(X)


class MyModel(torch.nn.Module):
    def __init__(self, model, vocab_size):
        super().__init__()
        self.model = model
        self.embed = MyEmbedding(model.embedding, vocab_size)
    
    def forward(self, X):
        X = self.embed(X)
        X = self.model(X, skip_embedding=True)
        return X




def to_one_hot(X):
    # X.shape == (nSeq, nVocab)
    inds = torch.argmax(X, dim=-1)
    X = torch.zeros(X.shape)
    for i, j in enumerate(inds):
        X[i, j] = 1
    return X


        
class MyInterp:
    def __init__(
        self,
        model,
        vocab_size,
        lr=0.01,
        optim="adam",
        k=1,
        max_iter=100,
    ):
        self.model = model
        self.lr = lr
        self.optim = optim
        self.vocab_size = vocab_size
        self.k = k
        self.max_iter = max_iter


    def do_full_step(self, X, target_ind):
        # take optimizer step
        # (loss fcn = 1 - fAnom)

        w = X.unsqueeze(0).clone().detach()
        w = w.float() ## prevent complaint that only floating point type tensors can require gradients
        w.requires_grad = True

        if self.optim.lower() == "adam":
            optimizer = torch.optim.Adam([w], lr=self.lr)
        elif self.optim.lower() == "sgd":
            optimizer = torch.optim.SGD([w], lr=self.lr)
        else:
            raise NotImplementedError(f"Unknown optimizer:  {self.optim}")
        output = self.model(w)
        loss = 1 - output[0][target_ind]
        loss0 = loss.detach().cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        X_star = w[0]

        # compute gradient at new position
        output_star = self.model(w)
        loss_star = 1 - output_star[0][target_ind]
        optimizer.zero_grad()
        loss_star.backward()

        grad_star = (w.grad)[0]  ## shape should be (seq x nVocab)

        # reset all but the most salient sequence elements
        grad_norm = torch.linalg.norm(grad_star, dim=-1)  ## shape = (nSeq)
        i_star = torch.argmax(grad_norm, dim=0).item()

        X_star_clamped = X.clone()
        X_star_clamped[i_star] = X_star[i_star]

        # force modified sequence elements to one-hot values
        # X_star_clamped[i_star] = torch.argmax(X_star_clamped[i_star], dim=-1)
        # j_star = torch.argmax(X_star_clamped[i_star], dim=-1)
        j_star = torch.argmax(X_star[i_star], dim=-1)  ## X_star.shape = (nSeq, nVocab)
        X_star_clamped[i_star] = 0
        X_star_clamped[i_star, j_star] = 1

        w = X_star_clamped.unsqueeze(0).clone().detach()
        output = self.model(w)
        loss = 1 - output[0][target_ind]
        loss_star = loss.detach().cpu()

        return X_star_clamped, loss_star, loss0

        if j_star != torch.argmax(X[i_star], dim=-1):
            return X_star_clamped
        else:
            return X_star
    
    def do_multi_steps(self, X, X0, target_ind, file=None):
        # X = to_one_hot(X)

        w = X.unsqueeze(0).clone().detach()
        w = w.float() ## prevent complaint that only floating point type tensors can require gradients
        w.requires_grad = True

        if self.optim.lower() == "adam":
            optimizer = torch.optim.Adam([w], lr=self.lr)
        elif self.optim.lower() == "sgd":
            optimizer = torch.optim.SGD([w], lr=self.lr)
        else:
            raise NotImplementedError(f"Unknown optimizer:  {self.optim}")

        i_iter = 0
        loss0 = None
        while True:

            optimizer.zero_grad()
            output = self.model(w)
            loss = 1 - output[0][target_ind]
            if loss0 is None:
                loss0 = loss.clone()

            loss.backward()
            optimizer.step()

            X_star = w[0]

            # compute gradient at new position
            optimizer.zero_grad()
            output_star = self.model(w)
            loss_star = 1 - output_star[0][target_ind]
            loss_star.backward()

            grad_star = (w.grad)[0]  ## shape should be (seq x nVocab)

            # reset all but the most salient sequence elements
            grad_norm = torch.linalg.norm(grad_star, dim=-1)  ## shape = (nSeq)
            # i_star = torch.argmax(grad_norm, dim=0).item()
            i_stars = torch.argsort(grad_norm, dim=0, descending=True)

            X_star_clamped = X0.clone()
            for i, i_star in enumerate(i_stars):
                X_star_clamped[i_star] = X_star[i_star]

                # force modified sequence elements to one-hot values
                j_star = torch.argmax(X_star[i_star], dim=-1)  ## X_star.shape = (nSeq, nVocab)
                X_star_clamped[i_star] = 0
                X_star_clamped[i_star, j_star] = 1

                if i >= self.k - 1:
                    break
            
            w_clamped = X_star_clamped.unsqueeze(0).clone().detach()
            output_clamped = self.model(w_clamped)
            loss_clamped = 1 - output_clamped[0][target_ind]

            # print(f"\tLoss:  loss0={loss0.item()}; loss={loss.item()}; loss_star={loss_star.item()}", file=file)

            if not torch.all(torch.eq(X_star_clamped, X)) and loss_clamped < loss0:
                print(f"\tFound clamped interpretation after {i_iter+1} iterations", file=file)
                break
            elif i_iter >= self.max_iter + 1:
                print(f"\tStopping after {i_iter+1} iterations", file=file)
                break

            i_iter += 1
        print(f"\t\tLoss:  {loss0} --> {loss_star} --> {loss_clamped} (delta={loss_clamped - loss0})", file=file)
        nSteps = i_iter+1
        return X_star_clamped, nSteps, loss_clamped, loss0, None


if __name__ == '__main__':
    main()


