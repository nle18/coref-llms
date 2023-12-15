"""Taken and modified from fast-coref repo: https://github.com/shtoshni/fast-coref"""
import os
import copy
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

from utils.io_utils import *


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


def convert_list_to_map(clusters):
    """Convert list of clusters to map of mention-clusters pair"""
    map = dict()
    for cluster in clusters:
        for mention in cluster:
            map[mention] = cluster
    return map


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold):
        """
        predicted: list of clusters, each cluster is a list of span, each span = [start, end]
        gold: list of clusters, same format as above
        """
        # first map all predicted and gold spans into tuple (for easier dict access)
        predicted = [tuple(tuple(span) for span in cluster) for cluster in predicted]
        gold = [tuple(tuple(span) for span in cluster) for cluster in gold]

        # create mention-cluster mapping
        mention_to_predicted = convert_list_to_map(predicted)
        mention_to_gold = convert_list_to_map(gold)

        # now do evaluation
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_component_f1(self):
        # return {repr(e): e.get_f1() for e in self.evaluators}
        return {repr(e): e.get_prf_str() for e in self.evaluators}


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def __repr__(self) -> str:
        return self.metric.__name__

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den

    def get_prf_str(self):
        # perf_str = (
        #     f"Recall: {self.get_recall() * 100:.1f}, Precision: {self.get_precision() * 100:.1f}, "
        #     f"F-score: {self.get_f1() * 100: .1f}\n"
        # )
        perf_str = (
            "Recall: {0:.1f}, Precision: {1:.1f}, ".format(
                self.get_recall() * 100, self.get_precision() * 100
            ),
            "F-score: {0: .1f}\n".format(self.get_f1() * 100),
        )

        return perf_str


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    # clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    # matching = linear_assignment(-scores)
    matching = linear_sum_assignment(-scores)  # ie Kuhn-Munkres algorithm
    matching = np.asarray(matching)
    matching = np.transpose(matching)

    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1 :]:
                    if (
                        m2 in mention_to_gold
                        and mention_to_gold[m] == mention_to_gold[m2]
                    ):
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem


def get_evaluations(doc_predictions: list, exp_dir: str):

    # if num_gpus, then we aggregate all the doc_predictions
    print(f"Evaluating {len(doc_predictions)} docs")

    evaluator = CorefEvaluator()
    doc_F1s = dict()
    for doc_example in doc_predictions:

        doc_key = doc_example["doc_key"]

        if len(doc_example["predicted_clusters"]) == 0:
            continue

        evaluator.update(
            doc_example["predicted_clusters"],
            doc_example["gold_clusters"],
            # [c for c in doc_example["gold_clusters"] if len(c) > 1], # not evaluating singletons
        )

        # get doc-wise F1 for error analysis
        doc_eval = CorefEvaluator()
        doc_eval.update(
            doc_example["predicted_clusters"],
            doc_example["gold_clusters"],
            # [c for c in doc_example["gold_clusters"] if len(c) > 1], # not evaluating singletons
        )
        doc_F1s[doc_key] = {
            "CoNLL_F1": doc_eval.get_f1(),
            "Detailed_F1": doc_eval.get_component_f1(),
        }

    print("CoNLL F1 =", evaluator.get_f1())
    print("Detailed F1s =", evaluator.get_component_f1())
    write_json(
        {
            "CoNLL_F1": evaluator.get_f1(),
            "Detailed_F1": evaluator.get_component_f1(),
        },
        os.path.join(exp_dir, "overall_F1.json"),
    )
    write_json(
        doc_F1s,
        os.path.join(exp_dir, "doc_F1.json"),
    )
