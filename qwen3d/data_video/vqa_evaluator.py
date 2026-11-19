# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
import torch
import random
import detectron2.utils.comm as comm
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from detectron2.evaluation import DatasetEvaluator
from qwen3d.utils.misc import all_gather, is_dist_avail_and_initialized
from torch.nn import functional as F
from qwen3d.data_video.referrential_grounding_evaluator import ReferrentialGroundingEvaluator
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import re
from collections import Counter

import ipdb
st = ipdb.set_trace


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()

VLMCD_DIST_THRESHOLDS = (0.1, 0.25, 0.5)  # metres


def match_move_distances(pred, gt):
    """Pair predicted and ground-truth moved objects of one example to score their distances.

    Within each class, predicted and true moves are paired one to one, minimising the
    total absolute distance error (``scipy.optimize.linear_sum_assignment``). This is the
    best case when an example moves several objects of the same class; with one object
    per class, which is the usual case, it is simply the pair.

    Parameters
    ----------
    pred, gt : list of tuple
        ``parse_describe_records`` output.

    Returns
    -------
    tuple
        ``(errors, n_gt, n_missing)``: absolute errors in metres of the matched pairs,
        ground-truth moves that have a distance, and of those, ones matched to a
        prediction of the class that gave no distance.
    """
    from scipy.optimize import linear_sum_assignment
    errors, n_gt, n_missing = [], 0, 0
    classes = {c for a, c, d in gt if a == "move" and d is not None}
    for cls in classes:
        g = [d for a, c, d in gt if a == "move" and c == cls and d is not None]
        p_all = [d for a, c, d in pred if a == "move" and c == cls]
        p = [d for d in p_all if d is not None]
        n_gt += len(g)
        if p:
            cost = np.abs(np.array(p)[:, None] - np.array(g)[None, :])
            rows, cols = linear_sum_assignment(cost)
            errors += [float(cost[r, c]) for r, c in zip(rows, cols)]
        # Further true moves of the class, predicted as moved but without a distance.
        n_missing += max(0, min(len(g), len(p_all)) - len(p))
    return errors, n_gt, n_missing


def vlmcd_metrics(dataset_name, res):
    """Task metrics for ``vlmcd_<task>_<split>`` predictions.

    Parameters
    ----------
    dataset_name : str
    res : list of tuple
        ``(prediction, [answer])`` pairs, lower-cased.

    Returns
    -------
    dict
        ``describe``: micro precision / recall / F1 over ``(action, category)`` changes,
        and per-action F1. For moved objects whose true distance is given (seen in both
        scans), paired by class (``match_move_distances``): ``move_dist_mae_m`` and
        ``move_dist_medae_m`` (absolute error of the paired distances),
        ``move_dist_acc_<t>m`` (share of pairs within t metres), ``move_dist_recall_0.25m``
        (share of all such true moves found with the right class and a distance within
        0.25 m), ``move_dist_pairs`` and ``move_dist_missing`` (paired moves without a
        predicted distance). JSON answers add ``json_valid`` (share of predictions that
        parse as a whole). ``count``: per-action and all-actions count accuracy, and the
        mean absolute error of the total.
    """
    from qwen3d.data_video.datasets.vlm_cd import (ACTIONS, dataset_task, is_json_answer, parse_count,
                                                   parse_describe_records)
    out = {}
    task = dataset_task(dataset_name)
    if task == "describe":
        tp = fp = fn = 0
        per = {a: [0, 0, 0] for a in ACTIONS}
        errors, n_gt_dist, n_missing, n_json, json_task = [], 0, 0, 0, False
        for pred, answers in res:
            pr, gr = parse_describe_records(pred), parse_describe_records(answers[0])
            p = Counter((a, c) for a, c, _ in pr)
            g = Counter((a, c) for a, c, _ in gr)
            e, n, m = match_move_distances(pr, gr)
            errors += e
            n_gt_dist += n
            n_missing += m
            if is_json_answer(answers[0]):
                json_task = True
                n_json += is_json_answer(pred)
            for key in set(p) | set(g):
                t = min(p[key], g[key])
                tp += t
                fp += p[key] - t
                fn += g[key] - t
                per[key[0]][0] += t
                per[key[0]][1] += p[key] - t
                per[key[0]][2] += g[key] - t
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.update(change_precision=prec, change_recall=rec,
                   change_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        for a, (t, f_p, f_n) in per.items():
            out[f"{a}_f1"] = 2 * t / (2 * t + f_p + f_n) if t + f_p + f_n else 0.0
        if n_gt_dist:
            err = np.array(errors) if errors else np.array([np.nan])
            out["move_dist_mae_m"] = float(np.mean(err))
            out["move_dist_medae_m"] = float(np.median(err))
            for t in VLMCD_DIST_THRESHOLDS:
                out[f"move_dist_acc_{t}m"] = float(np.mean(err <= t)) if errors else 0.0
            out["move_dist_recall_0.25m"] = sum(x <= 0.25 for x in errors) / n_gt_dist
            out["move_dist_pairs"] = float(len(errors))
            out["move_dist_missing"] = float(n_missing)
        if json_task:
            out["json_valid"] = n_json / max(len(res), 1)
    elif task == "count":
        correct = {a: 0 for a in ACTIONS}
        all_correct, abs_err = 0, 0.0
        for pred, answers in res:
            p, g = parse_count(pred), parse_count(answers[0])
            ok = [p[a] == g[a] for a in ACTIONS]
            for a, o in zip(ACTIONS, ok):
                correct[a] += o
            all_correct += all(ok)
            abs_err += abs(sum(p[a] or 0 for a in ACTIONS) - sum(g[a] or 0 for a in ACTIONS))
        n = max(len(res), 1)
        out.update({f"{a}_count_acc": correct[a] / n for a in ACTIONS})
        out.update(count_all_acc=all_correct / n, total_count_mae=abs_err / n)
    print(", ".join(f"{k}: {v:.4f}" for k, v in out.items()))
    return out


class VQAEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        evaluate_detection,
        cfg=None
    ):
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.num_viz = 0
        self.results = []
        self.vlmcd_records = []
        self.metric2scorer = {'cider': Cider(), 'bleu': Bleu(), 'meteor': Meteor(), 'rouge': Rouge()}

        self.evaluate_detection = evaluate_detection
        print(f"evaluates grounding: {self.evaluate_detection}")
        if self.evaluate_detection:
            self.detection_evaluator = ReferrentialGroundingEvaluator(
                dataset_name,
                thresholds=[0.25, 0.5, 0.75],
                topks=[1, 2, 5],
                cfg=cfg
            )

        self._cpu_device = torch.device("cpu")


    def reset(self):
        self.results = []
        self.vlmcd_records = []
        self.num_viz = 0
        if self.evaluate_detection:
            self.detection_evaluator.reset()

    def _save_vlmcd_predictions(self, eval_results):
        """Write per-example predictions and the metrics to the inference folder."""
        import json
        import os
        root = getattr(self.cfg, "OUTPUT_DIR", None) if self.cfg is not None else None
        if not root:
            return
        out_dir = os.path.join(root, "inference")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{self.dataset_name}_predictions.json"), "w") as f:
            json.dump({"metrics": eval_results, "predictions": self.vlmcd_records}, f, indent=1)

    # called per batch
    def process(self, inputs, outputs):
        assert(len(inputs) == 1)
        for i in range(len(outputs[0])):
            if 'sr3d_data' in inputs[0]:
                if self.dataset_name == "mmlupro_text_bench":
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
                elif self.dataset_name == 'realworld_vqa_bench':
                    self.results.append((outputs[0][i].lower(), [answer.lower() for answer in inputs[0]['sr3d_data'][i]['answers']]))
                elif 'scanqa' in self.dataset_name or 'sqa3d' in self.dataset_name:
                    self.results.append((outputs[0][i].lower(), [answer.lower() for answer in inputs[0]['sr3d_data'][i]['answers']]))
                elif self.dataset_name.startswith('vlmcd_'):
                    self.results.append((outputs[0][i].lower().strip(), [answer.lower() for answer in inputs[0]['sr3d_data'][i]['answers']]))
                    self.vlmcd_records.append({
                        "example_id": inputs[0]['sr3d_data'][i]['example_id'],
                        "frame_ids": inputs[0].get('frame_ids'),
                        "prediction": outputs[0][i].strip(),
                        "answer": inputs[0]['sr3d_data'][i]['answers'][0],
                    })
            else:
                if self.dataset_name == "mmlupro_text_bench":
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
                elif self.dataset_name == 'realworld_vqa_bench':
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
                elif 'scanqa' in self.dataset_name or 'sqa3d' in self.dataset_name:
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
        if self.evaluate_detection:
            self.detection_evaluator.process(inputs, outputs[1])

    # called per dataset
    def evaluate(self):
        if is_dist_avail_and_initialized():
            # InferenceSampler gives each rank a contiguous slice; concatenate
            # every rank's list (not just rank 0's) to score the whole split.
            res = list(itertools.chain(*all_gather(self.results)))
            self.vlmcd_records = list(itertools.chain(*all_gather(self.vlmcd_records)))
            comm.synchronize()
            if self.evaluate_detection:
                detection_result = self.detection_evaluator.evaluate()
            if not comm.is_main_process():
                comm.synchronize()
                return {}
        else:
            res = self.results
            if self.evaluate_detection:
                detection_result = self.detection_evaluator.evaluate()

        gts = {}
        preds = {}
        eval_results = {}
        exact_matches = 0
        tokenizer = PTBTokenizer()

        for i in range(len(res)):
            if "bench" in self.dataset_name:
                res[i] = (res[i][0].lower().strip().rstrip("."), res[i][1].lower().strip().rstrip("."))
                if res[i][0] in res[i][1]:
                    exact_matches += 1
                else:
                    print(f"Incorrect: Predicted: {res[i][0]}, Ground Truth: {res[i][1]}")
            elif self.dataset_name.startswith('vlmcd_'):
                # Long answers: count exact matches without printing every miss.
                exact_matches += res[i][0].rstrip(".") in [a.rstrip(".") for a in res[i][1]]
            elif 'scanqa' in self.dataset_name or 'sqa3d' in self.dataset_name:
                if res[i][0] in res[i][1]:
                    exact_matches += 1
                else:
                    print(f"Incorrect: Predicted: {res[i][0]}, Ground Truth: {res[i][1]}")
                
            preds[i] = [{'caption': res[i][0]}]
            gts[i] = [{'caption': ans} for ans in res[i][1]]
        

        gts = tokenizer.tokenize(gts)
        preds = tokenizer.tokenize(preds)

        eval_results['exact_match'] = exact_matches / len(res)
        print(f'exact match: {np.round(exact_matches / len(res), 4)}, ')

        for metric_name in self.metric2scorer.keys():
            # Report the scorer's corpus-level score (standard pycocoevalcap protocol).
            # The mean of per-sentence scores is wrong for BLEU-4 (~0 for the 1-3 word
            # answers here) and is not METEOR's aggregate. CIDEr/ROUGE are unaffected.
            score = self.metric2scorer[metric_name].compute_score(gts, preds)[0]
            if metric_name == 'bleu':
                score = score[-1] # bleu-4
            print(f'{metric_name}: {np.round(score, 4)}, ', end='')
            eval_results[metric_name] = float(score)
        print()

        if self.evaluate_detection:
            eval_results.update(detection_result)

        if self.dataset_name.startswith('vlmcd_'):
            eval_results.update(vlmcd_metrics(self.dataset_name, res))
            self._save_vlmcd_predictions(eval_results)

        if is_dist_avail_and_initialized():
            comm.synchronize()

        return eval_results
