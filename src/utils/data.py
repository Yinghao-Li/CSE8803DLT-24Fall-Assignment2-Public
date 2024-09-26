"""
# Author: Yinghao Li
# Modified: September 13th, 2023
# ---------------------------------------
# Description: Utility functions for data processing 
"""

import operator
import torch
import copy
import logging
import itertools
import numpy as np

from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)


def span_to_label(labeled_spans: Dict[Tuple[int, int], str], tokens: List[str]) -> List[str]:
    """
    Convert entity spans to labels

    Parameters
    ----------
    labeled_spans: labeled span dictionary: {(start, end): label}
    tokens: a list of tokens, used to check if the spans are valid.

    Returns
    -------
    a list of string labels
    """
    if labeled_spans:
        assert list(labeled_spans.keys())[-1][1] <= len(tokens), ValueError("label spans out of scope!")

    labels = ["O"] * len(tokens)
    for (start, end), label in labeled_spans.items():
        if type(label) == list or type(label) == tuple:
            lb = label[0][0]
        else:
            lb = label
        labels[start] = "B-" + lb
        if end - start > 1:
            labels[start + 1 : end] = ["I-" + lb] * (end - start - 1)

    return labels


def label_to_span(labels: List[str], scheme: Optional[str] = "BIO") -> dict:
    """
    convert labels to spans
    :param labels: a list of labels
    :param scheme: labeling scheme, in ['BIO', 'BILOU'].
    :return: labeled spans, a list of tuples (start_idx, end_idx, label)
    """
    assert scheme in ["BIO", "BILOU"], ValueError("unknown labeling scheme")

    labeled_spans = dict()
    i = 0
    while i < len(labels):
        if labels[i] == "O":
            i += 1
            continue
        else:
            if scheme == "BIO":
                if labels[i][0] == "B":
                    start = i
                    ent = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] == "I":
                            # B-ent followed by I- with different entity; discard the incorrect I- labels
                            if labels[i][2:] != ent:
                                break
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = ent
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = ent
                        i += 1
                # this should not happen
                elif labels[i][0] == "I":
                    i += 1
            elif scheme == "BILOU":
                if labels[i][0] == "U":
                    start = i
                    end = i + 1
                    ent = labels[i][2:]
                    labeled_spans[(start, end)] = ent
                    i += 1
                elif labels[i][0] == "B":
                    start = i
                    ent = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] != "L":
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = ent
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = ent
                        break
                    i += 1
                else:
                    i += 1

    return labeled_spans


def span_dict_to_list(span_dict: Dict[Tuple[int], str]):
    """
    convert entity label span dictionaries to span list

    Parameters
    ----------
    span_dict

    Returns
    -------
    span_list
    """
    span_list = list()
    for (s, e), v in span_dict.items():
        span_list.append([s, e, v])
    return span_list


def span_list_to_dict(span_list: List[list]) -> Dict[Tuple[int, int], Union[str, tuple]]:
    """
    convert entity label span list to span dictionaries

    Parameters
    ----------
    span_list

    Returns
    -------
    span_dict
    """
    span_dict = dict()
    for span in span_list:
        span_dict[(span[0], span[1])] = span[2]
    return span_dict


def one_hot(x, n_class=None):
    """
    x : LongTensor of shape (batch size, sequence max_seq_length)
    n_class : integer

    Convert batch of integer letter indices to one-hot vectors of dimension S (# of possible x).
    """

    if n_class is None:
        n_class = np.max(x) + 1
    one_hot_vec = np.zeros([int(np.prod(x.shape)), n_class])
    indices = x.reshape([-1])
    one_hot_vec[np.arange(len(indices)), indices] = 1.0
    one_hot_vec = one_hot_vec.reshape(list(x.shape) + [n_class])
    return one_hot_vec


def probs_to_ids(probs: Union[torch.Tensor, np.ndarray]):
    """
    Convert label probability labels to index

    Parameters
    ----------
    probs: label probabilities

    Returns
    -------
    label indices (shape = one_hot_lbs.shape[:-1])
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    lb_ids = probs.argmax(axis=-1)
    return lb_ids


def ids_to_lbs(ids: Union[torch.Tensor, np.ndarray], label_types: List[str]):
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    np_map = np.vectorize(lambda lb: label_types[lb])
    return np_map(ids)


def probs_to_lbs(probs: Union[torch.Tensor, np.ndarray], label_types: List[str]):
    """
    Convert label probability labels to index

    Parameters
    ----------
    probs: label probabilities
    label_types: label types, size = probs.shape[-1]

    Returns
    -------
    labels (shape = one_hot_lbs.shape[:-1])
    """
    np_map = np.vectorize(lambda lb: label_types[lb])
    lb_ids = probs_to_ids(probs)
    return np_map(lb_ids)


def entity_to_bio_labels(entities: List[str]):
    bio_labels = ["O"] + ["%s-%s" % (bi, label) for label in entities for bi in "BI"]
    return bio_labels


def merge_list_of_lists(lists: List[list]):
    merged = list(itertools.chain.from_iterable(lists))
    return merged


def split_list_by_lengths(input_list: list, lengths: List[int]) -> List[list]:
    """
    Split a list into several lists given the lengths of each target list

    Parameters
    ----------
    input_list: the list to split
    lengths: the length of each target list

    Returns
    -------
    a list of split lists
    """
    ends = list(itertools.accumulate(lengths, operator.add))
    assert ends[-1] <= len(input_list), ValueError("The lengths does not match the input list!")

    starts = [0] + ends[:-1]
    output = [input_list[s:e] for s, e in zip(starts, ends)]
    return output


def sort_tuples_by_element_idx(
    tups: List[tuple], idx: Optional[int] = 0, reverse: Optional[bool] = False
) -> List[tuple]:
    """
    Function to sort the list of tuples by the second item

    Parameters
    ----------
    tups: a list of tuples to sort
    idx: will sort according to the ith element
    reverse: set True to sort in descending order

    Returns
    -------
    sorted tuples
    """
    tup_ = copy.deepcopy(tups)
    tup_.sort(key=lambda x: x[idx], reverse=reverse)
    return tup_


def merge_overlapped_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapped spans

    Parameters
    ----------
    spans: input list of spans

    Returns
    -------
    merged spans
    """

    span_sets = list()
    for span in spans:
        span_set = set(range(span[0], span[1]))
        if not span_sets:
            span_sets.append(span_set)
        elif span_sets[-1] & span_set:
            if span_set - span_sets[-1]:
                span_sets[-1] = span_sets[-1] | span_set
        else:
            span_sets.append(span_set)

    merged_spans = list()
    for span_set in span_sets:
        merged_spans.append((min(span_set), max(span_set) + 1))

    return merged_spans


def rand_argmax(x, **kwargs):
    """
    a random tie-breaking argmax
    https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking/42071648
    """
    return np.argmax(np.random.random(x.shape) * (x == np.amax(x, **kwargs, keepdims=True)), **kwargs)


def lengths_to_mask(lengths: Union[torch.Tensor, List[int]]):
    """
    Convert sequence lengths to boolean mask.

    Parameters
    ----------
    lengths: sequence lengths; Should be Int/Long Tensor with shape (batch_size, )

    Returns
    -------
    torch.Tensor, Boolean Tensor with shape (batch_size, maximum length).
    The paddings are marked as False.
    """
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths, dtype=torch.long)
    max_length = lengths.max()
    padding_masks = torch.arange(max_length)[None, :] < lengths[:, None]
    return padding_masks
