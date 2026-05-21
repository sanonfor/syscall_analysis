import argparse
from pathlib import Path
import datetime
import json
import csv

from simian.v1.lm_dataset import LMDataset
from simian.v1.indices import IndexedDicts
from simian.v1.data_reader import DataReader
from simian.v1.corpus import Corpus
from simian.ml import collate_fn, correct
from main import get_valid_and_test_sampler

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


def main(cmd_line_args=None):
    """
    Compute system-call-level anomaly scores and save to file,
    given an index directory containing train and test npz files
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--mlm", action="store_true", help="Whether the model is an MLM")
    parser.add_argument("--max-length", type=int, default=256, help="Max length of the sequence")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--results-base-dir", default="/audit_logs/<username>/results")
    
    parser.add_argument("--idx-dir", default="/audit_logs/<username>/corpus")
    # parser.add_argument("--data-path", default="/audit_logs/<username>/stable-cache/tce4_25p_27p.csv")

    # tag = "output-ep500-seed0-fs2a-tce4-10p-80p-10p-0.25p-train-25p-27p-test"
    tag = "output-ep500-seed0-fs2a-tce4-10p-80p-10p-0.25p-train-90p-92p-test"
    # tag = "output-ep500-seed0-fs2a-tce4-10p-80p-10p-0.25p-train-92p-94p-test"
    parser.add_argument("--corpus-path", default=f"/audit_logs/<username>/models/{tag}/corpus")
    parser.add_argument("--model-path", default=f"/audit_logs/<username>/models/{tag}/lstm/fournier_proxy_minus/model_file")
    args = parser.parse_args(cmd_line_args)

    results_base_dir = Path(args.results_base_dir)
    time_format = "%Y%m%d-%H%M%S"
    timestamp = datetime.datetime.now().strftime(time_format)
    results_dir = results_base_dir / timestamp
    results_dir.mkdir()

    with open(results_dir / "config.json", 'w') as fjson:
        json.dump(vars(args), fjson, indent=4)

    log_path = results_dir / "log.txt"
    flog = open(log_path, 'w')

    def dprint(*args, echo=True, **kwargs):
        timestamp = datetime.datetime.now().strftime(time_format)
        timestamp_slug = f"[{timestamp}]"
        print(timestamp_slug, *args, **kwargs, file=flog)
        if echo:
            print(timestamp_slug, *args, **kwargs)
    
    indices_dir_path = args.corpus_path
    test_corpus_path = str( Path(args.corpus_path) / "test.npz" )


    dprint("Starting")
    dprint("Loading model")
    if args.mlm:
        raise NotImplementedError()
    
    device = torch.device("cpu") # temp fix for cuda error
    model = torch.load(
        args.model_path,
        map_location=device,
    )

    # dprint("Creating idx_dicts")
    # idx_dicts = IndexedDicts.create(args.idx_dir)

    # dprint("Building data reader")
    # test_data_reader = DataReader.create(args.data_path)

    # dprint("Creating corpus")
    # test_corpus_path = args.idx_dir
    # test_corpus = Corpus.create(
    #     test_corpus_path, args.max_length, idx_dicts, test_data_reader
    # )


    dprint("Loading idx_dicts")
    idx_dicts = IndexedDicts.load(indices_dir_path)

    dprint("Loading test corpus")
    test_corpus = Corpus.load(test_corpus_path)

    vocabulary_size = idx_dicts.get_vocabulary_size()

    dprint("Tensorizing corpus")
    test_corpus.tensorize()

    dprint("Creating dataloader")
    valid_sampler, test_sampler = get_valid_and_test_sampler(args.validation_fraction, test_corpus)
    lm_test_loader = DataLoader(LMDataset(test_corpus),
                                batch_size=args.batch_size,
                                sampler=test_sampler,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                num_workers=0)
    
    # evaluate model
    dprint("Evaluating model")
    model.eval()
    total_val_pred, total_val_correct = 0, 0


    fpath_csv = results_dir / "eval_step_results.csv"
    fcsv = open(fpath_csv, 'w')
    writer = csv.writer(fcsv)
    headers = ["nll", "is_acc"]
    writer.writerow(headers)

    with torch.no_grad():
        for data in tqdm(lm_test_loader, "eval batches: "):
            # send tensors to device
            data = [d.to(device) for d in data]

            # get the pad_mask and the output from the data
            data, y, pad_mask = data[:-2], data[-2], data[-1]

            # get prediction
            out = model(*data, pad_mask, args.mlm, chk=False)

            # collect metrics
            total_val_pred += float(torch.nonzero(y).size(0))
            total_val_correct += correct(out, y, vocabulary_size)

            # calculate step metrics
            p_out = torch.softmax(out, dim=-1)   # (batch_size, window_size, vocab_size)

            # p_gt:  essentially perform p_out[y]; see https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms/51032153#51032153
            # demo:  
            #   aa = torch.tensor([ [[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[13,14,15],[16,17,18]] ])
            #   bb = torch.tensor([[0,1,0],[2,0,1]])
            #   torch.gather(aa, -1, bb.unsqueeze(-1)).squeeze() == torch.tensor([[1,5,7],[12,13,17]])
            p_gt = torch.gather(p_out, -1, y.unsqueeze(-1)).squeeze()  # (batch_size, window_size)
            nll = -np.log(p_gt)

            # average over window_size
            p_gt_avg = torch.sum(p_gt, dim=-1) / p_gt.shape[-1]  # (batch_size)
            nll_avg = torch.sum(nll, dim=-1) / nll.shape[-1]  # (batch_size)

            pred = torch.argmax(out, dim=-1)
            is_acc = (pred == y)

            # save nll and is_acc for each elem in the batch
            rows = []
            # for row_nll, row_is_acc in zip(nll, is_acc): # each nrows == batch_size
            #     for single_nll, single_is_acc in zip(row_nll, row_is_acc):
            for i in range(nll.shape[0]):
                for j in range(nll.shape[1]):
                    row = [
                        nll[i][j].item(),
                        is_acc[i][j].item(),
                    ]
                    rows.append(row)
            writer.writerows(rows)
            fcsv.flush()

    
    dprint("Done with eval")
    dprint(f"Accuracy:  {total_val_correct}/{total_val_pred} = {total_val_correct/total_val_pred*100} %")

    dprint(f"Saved results to {results_dir}")

if __name__ == '__main__':
    main()
