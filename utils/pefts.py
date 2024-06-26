import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False

UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention"}
DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

######################################
# 1.1 SODA_SVD (3 kronecker)         #
######################################
class InjectedSODA_SVD(nn.Module):
    def __init__(
        self, in_features, out_features, 
        bias=False, dropout_p=0.0, scale=1.0, 
        tmp_l=None, # to hardcode the shape of kronecker matrices
        merge_kron=None, merge_svd=None # to merge different checkpoints
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        if merge_kron == None:
            # SVD delta
            self.delta = nn.Parameter(torch.zeros(out_features))

            # init delta to be 1e3
            nn.init.constant_(self.delta, 1e-3)

            # Q
            # automatically get kron size
            # size_kron_Q1, size_kron_Q2, size_kron_Q3 = get_abc(in_features)
            svd_l=tmp_l
            size_kron_Q1, size_kron_Q2, size_kron_Q3 = svd_l[in_features]
            self.kron_Q1 = nn.Parameter(torch.eye(size_kron_Q1, size_kron_Q1))
            self.kron_Q2 = nn.Parameter(torch.eye(size_kron_Q2, size_kron_Q2))
            self.kron_Q3 = nn.Parameter(torch.eye(size_kron_Q3, size_kron_Q3))

        else:
            self.kron_Q1 = nn.Parameter(merge_kron[0])
            self.kron_Q2 = nn.Parameter(merge_kron[1])
            self.kron_Q3 = nn.Parameter(merge_kron[2])

            self.delta = nn.Parameter(merge_svd)
    
    def SVD_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.U, self.S, self.V = torch.linalg.svd(W, full_matrices=False)

        del self.linear.weight

    def forward(self, input):
        # add residual
        S = F.relu(self.S + self.delta)
        W = self.U @ torch.diag(S) @ self.V
        # rotate
        orthogonal_matrix = torch.kron(torch.kron(self.kron_Q1, self.kron_Q2), self.kron_Q3)
        W_t = orthogonal_matrix @ W.t()
        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_soda_svd(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    tmp_l=None, # to hardcode the shape of kronecker matrices
    subject_list=None, # saved weights for the subject customization
    style_list=None, # saved weights for the style customization
    db_lambda=1.0, style_lambda=1.0, # weights for merging the saved weights
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    # merge lora_list and style_list
    if subject_list is not None:
        db_kron = list(itertools.chain([subject_list.pop(0)], [subject_list.pop(0)], [subject_list.pop(0)]))
        db_lora= subject_list.pop(0)
        
        style_kron = list(itertools.chain([style_list.pop(0)], [style_list.pop(0)], [style_list.pop(0)]))
        style_lora = style_list.pop(0)

        merge_svd = db_lora * db_lambda + style_lora * style_lambda
        merge_kron = [db_kron[0] * db_lambda + style_kron[0] * style_lambda, db_kron[1] * db_lambda + style_kron[1] * style_lambda, db_kron[2] * db_lambda + style_kron[2] * style_lambda]
    else:
        merge_kron = None
        merge_svd = None

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = InjectedSODA_SVD(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None, 
            dropout_p=dropout_p,
            scale=scale,
            tmp_l=tmp_l,
            merge_kron=merge_kron,
            merge_svd=merge_svd
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.SVD_decompose() # to perform QR decomposition

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append([_module._modules[name].kron_Q1])
        require_grad_params.append([_module._modules[name].kron_Q2])
        require_grad_params.append([_module._modules[name].kron_Q3])
        require_grad_params.append([_module._modules[name].delta])

        if subject_list is None:
            _module._modules[name].kron_Q1.requires_grad = True
            _module._modules[name].kron_Q2.requires_grad = True
            _module._modules[name].kron_Q3.requires_grad = True
            _module._modules[name].delta.requires_grad = True
        names.append(name)

    return require_grad_params,  names

######################################
# 2. Utils                            #
######################################
def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = None
    #[
    #    LoraInjectedLinear,
    #    LoraInjectedConv2d,
    #],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module

_find_modules = _find_modules_v2