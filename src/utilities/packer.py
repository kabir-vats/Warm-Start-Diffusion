from typing import Dict, List

import torch
import torch.jit
from tensordict import TensorDict, TensorDictBase


class NoPacker:
    def pack(self, tensors: Dict[str, torch.Tensor], axis=0) -> torch.Tensor:
        return tensors

    def unpack(self, tensor: torch.Tensor, axis=0) -> Dict[str, torch.Tensor]:
        return tensor


class Packer:
    """
    Responsible for packing tensors into a single tensor.
    """

    def __init__(self, names: List[str], axis=None, axis_pack=None, axis_unpack=None):
        self.names = names
        if axis is not None:
            assert axis_pack is None, "Cannot specify both axis and axis_pack"
            assert axis_unpack is None, "Cannot specify both axis and axis_unpack"
            self.axis_pack = axis
            self.axis_unpack = axis
        else:
            assert axis_pack is not None, "Must specify either axis or axis_pack"
            assert axis_unpack is not None, "Must specify either axis or axis_unpack"
            self.axis_pack = axis_pack
            self.axis_unpack = axis_unpack

    def pack(self, tensors: Dict[str, torch.Tensor], axis=None) -> torch.Tensor:
        """
        Packs tensors into a single tensor, concatenated along a new axis

        Args:
            tensors: Dict from names to tensors.
            axis: index for new concatenation axis.
        """
        axis = axis if axis is not None else self.axis_pack
        return _pack(tensors, self.names, axis=axis)

    def unpack(self, tensor: torch.Tensor, axis=None) -> TensorDict:
        axis = axis if axis is not None else self.axis_unpack
        # packed shape is tensor.shape with axis removed
        packed_shape = list(tensor.shape)
        packed_shape.pop(axis)
        assert tensor.shape[axis] == len(self.names), f"{tensor.shape=}, {len(self.names)=}"
        return TensorDict(_unpack(tensor, self.names, axis=axis), batch_size=packed_shape)

    def unpack_simple(self, tensor: torch.Tensor, axis=None) -> Dict[str, torch.Tensor]:
        axis = axis if axis is not None else self.axis_unpack
        return _unpack(tensor, self.names, axis=axis)

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {"names": self.names, "axis": self.axis}

    @classmethod
    def from_state(self, state) -> "Packer":
        """
        Loads state from a serializable data structure.
        """
        return Packer(state["names"], state["axis"])


@torch.jit.script
def _pack(tensors: Dict[str, torch.Tensor], names: List[str], axis: int) -> torch.Tensor:
    return torch.stack([tensors[n] for n in names], dim=axis)


@torch.jit.script
def _unpack(tensor: torch.Tensor, names: List[str], axis: int) -> Dict[str, torch.Tensor]:
    return {n: tensor.select(axis, index=i) for i, n in enumerate(names)}


class PackerDict:
    def __init__(self, packers, remove_key_type: bool = True, key_types=("preds", "targets", "inputs")):
        self.packers = packers
        self.packer_names = packers.keys()
        self.remove_key_type = remove_key_type
        self.key_types = key_types

    def k_to_base_key(self, k):
        if self.remove_key_type:
            for key_type in self.key_types:
                k = k.replace(key_type, "")
        return k

    def pack(self, tensors: TensorDict, axis=None) -> TensorDict:
        """Modifies tensors in place"""
        for k in tensors.keys():
            assert self.k_to_base_key(k) in self.packers, f"key {k} not in packers"
            tensors[k] = self.packers[self.k_to_base_key(k)].pack(tensors[k], axis=axis)
        return tensors

    def unpack(self, tensors: TensorDictBase, axis=None) -> TensorDict:
        unpacked = {k: self.packers[self.k_to_base_key(k)].unpack(v, axis=axis) for k, v in tensors.items()}
        if isinstance(tensors, TensorDictBase):
            return TensorDict(unpacked, batch_size=tensors.batch_size)
        return unpacked
