"""
# Author: Yinghao Li
# Modified: September 12th, 2023
# ---------------------------------------
# Description: Auxiliary functions for batch processing
"""

import torch
from typing import Optional


class Batch:
    """
    A batch of data instances, each is initialized with a dict with attribute names as keys
    """

    def __init__(self, **kwargs):
        self.size = 0
        super().__init__()
        self._tensor_members = dict()
        for k, v in kwargs.items():
            if k == "batch_size":
                self.size = v
            setattr(self, k, v)
            self.register_tensor_members(k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def as_dict(self):
        return {k: getattr(self, k) for k in self._tensor_members.keys()}

    def register_tensor_members(self, k, v):
        """
        Register tensor members to the batch
        """
        if isinstance(v, torch.Tensor) or callable(getattr(v, "to", None)):
            self._tensor_members[k] = v

    def to(self, device):
        """
        Move all tensor members to the target device
        """
        for k, v in self._tensor_members.items():
            setattr(self, k, v.to(device))
        return self

    def __len__(self):
        return len(tuple(self._tensor_members.values())[0]) if not self.size else self.size


def pack_instances(**kwargs) -> list[dict]:
    """
    Convert attribute lists to a list of data instances, each is a dict with attribute names as keys
    and one datapoint attribute values as values
    """

    instance_list = list()
    keys = tuple(kwargs.keys())

    for inst_attrs in zip(*tuple(kwargs.values())):
        inst = dict(zip(keys, inst_attrs))
        instance_list.append(inst)

    return instance_list


def unpack_instances(instance_list: list[dict], attr_names: Optional[list[str]] = None):
    """
    Convert a list of dict-type instances to a list of value lists,
    each contains all values within a batch of each attribute

    Parameters
    ----------
    instance_list: list[dict]
        a list of attributes
    attr_names: list[str], optional
        the name of the needed attributes. Notice that this variable should be specified
        for Python versions that does not natively support ordered dict
    """
    if not attr_names:
        attr_names = list(instance_list[0].keys())
    attribute_tuple = [[inst[name] for inst in instance_list] for name in attr_names]

    return attribute_tuple
