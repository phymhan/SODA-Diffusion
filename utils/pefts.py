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
# 1. LoRA                            #
######################################
class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (
            self.linear(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)

def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []
    parms_for_save = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        parms_for_save.append(_module._modules[name].lora_up.weight)
        parms_for_save.append(_module._modules[name].lora_down.weight)

        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        names.append(name)

    return require_grad_params, parms_for_save, names

class LoraInjectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (
            self.conv(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)

def extract_lora_ups_down(model, target_replace_module=DEFAULT_TARGET_REPLACE):
    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        loras.append((_child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras

######################################
# 1.1 FT                             #
######################################
def inject_trainable_ft(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []
    parms_for_save = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight

        require_grad_params.append([weight])

        parms_for_save.append([weight])

        weight.requires_grad = True

    return require_grad_params, parms_for_save


######################################
# 2.1 Kron_Q                         #
######################################

class KronQRInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, r=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        #self.qr_r = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # automatically get size_kron_q
        if in_features % r == 0:
            size_kron_1, size_kron_2 = in_features // r, r
        else:
            raise ValueError("the in features channel number can not be divided by r, please choose another r")
        #print(size_kron_1, size_kron_2)
        self.kron_1 = nn.Parameter(torch.eye(size_kron_1, size_kron_1))
        self.kron_2 = nn.Parameter(torch.eye(size_kron_2, size_kron_2))
        #nn.init.zeros_(self.qr_r.weight)

    def forward(self, input):
        # rotate
        rotate_matrix = torch.kron(self.kron_1, self.kron_2)
        rotated_weight =  torch.transpose(rotate_matrix @ torch.transpose(self.linear.weight.data, 0, 1), 0, 1)
        out = self.dropout(F.linear(input, rotated_weight, bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_qr(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=rank,
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append([_module._modules[name].kron_1])
        require_grad_params.append([_module._modules[name].kron_2])
        #require_grad_params.append(_module._modules[name].qr_r.parameters())

        if loras != None:
            _module._modules[name].kron_1 = loras.pop(0)
            _module._modules[name].kron_2 = loras.pop(0)
            #_module._modules[name].qr_r.weight = loras.pop(0)

        _module._modules[name].kron_1.requires_grad = True
        _module._modules[name].kron_2.requires_grad = True
        #_module._modules[name].qr_r.weight.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 2.1.a Kron_Q3                      #
######################################
svd_l = {
    640:  (8, 8, 10),
    2048: (4, 8,  64),
    1280: (4, 16, 20),
    768:  (6, 8,  16)
} # keep the param same as svd

svd_l = {
    640:  (4, 8, 20),
    2048: (2, 16,  64),
    1280: (4, 8, 40),
    768:  (2, 4,  96)
} # keep the param same as oft

def get_abc(dim):
    exp = math.log(dim, 2)
    a = b = exp // 3
    a = 2 ** a
    b = 2 ** b
    c = dim // (a * b)
    return int(a), int(b), int(c)

class KronQRInjectedLinear3(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, r=2, tmp_l=None
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        #self.qr_r = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # automatically get kron size
        svd_l = tmp_l
        size_kron_1, size_kron_2, size_kron_3 = svd_l[in_features] 
        self.kron_1 = nn.Parameter(torch.eye(size_kron_1, size_kron_1))
        self.kron_2 = nn.Parameter(torch.eye(size_kron_2, size_kron_2))
        self.kron_3 = nn.Parameter(torch.eye(size_kron_3, size_kron_3))

    def forward(self, input):
        # rotate
        rotate_matrix = torch.kron(torch.kron(self.kron_1, self.kron_2), self.kron_3)
        rotated_weight =  torch.transpose(rotate_matrix @ torch.transpose(self.linear.weight.data, 0, 1), 0, 1)
        out = self.dropout(F.linear(input, rotated_weight, bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron3(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    tmp_l=None
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear3(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=rank,
            dropout_p=dropout_p,
            scale=scale,
            tmp_l=tmp_l
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append([_module._modules[name].kron_1])
        require_grad_params.append([_module._modules[name].kron_2])
        require_grad_params.append([_module._modules[name].kron_3])

        _module._modules[name].kron_1.requires_grad = True
        _module._modules[name].kron_2.requires_grad = True
        _module._modules[name].kron_3.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 2.1.b Kron_Q3_Cayley               #
######################################

def inject_trainable_kron3_cayley(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 2,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    block_share: bool = True,
    tmp_l=None
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias

        # use fixed r from rl.txt
        r = rl.pop(0)

        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRLinearLayer3_cayley(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            block_share=block_share,
            tmp_l=tmp_l
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].kron_1)
        require_grad_params.append(_module._modules[name].kron_2)
        require_grad_params.append(_module._modules[name].kron_3)

        _module._modules[name].kron_1.requires_grad = True
        _module._modules[name].kron_2.requires_grad = True
        _module._modules[name].kron_3.requires_grad = True
        names.append(name)

    return require_grad_params, names

class KronQRLinearLayer3_cayley(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_share=False, eps=6e-5, r=4, is_coft=False, tmp_l=None):
        super(KronQRLinearLayer3_cayley, self).__init__()

        self.r = r
        # Define the reduction rate:
        # automatically get kron size
        svd_l = tmp_l
        size_kron_1, size_kron_2, size_kron_3 = svd_l[in_features] 
        self.kron_1 = nn.Parameter(torch.zeros(size_kron_1, size_kron_1))
        self.kron_2 = nn.Parameter(torch.zeros(size_kron_2, size_kron_2))
        self.kron_3 = nn.Parameter(torch.zeros(size_kron_3, size_kron_3))
        
        # Check whether to use the constrained variant COFT 
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        # self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        # self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        
        # Define the fixed Linear layer: v
        # self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        self.linear = nn.Linear(in_features, out_features, bias)

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        self.block_share = block_share

    def forward(self, x):
        orig_dtype = x.dtype
        dtype = self.kron_1.dtype
        attn = self.linear

        # Cayley transform kron1,2,3
        kron_1 = self.cayley(self.kron_1)
        kron_2 = self.cayley(self.kron_2)
        kron_3 = self.cayley(self.kron_3)
        R = torch.kron(torch.kron(kron_1, kron_2), kron_3)

        # Block-diagonal parametrization
        block_diagonal_matrix = R

        # fix filter
        fix_filt = attn.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
 
        # Apply the trainable identity matrix
        bias_term = attn.bias.data if attn.bias is not None else None
        if bias_term is not None:
            bias_term = bias_term.to(orig_dtype)

        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias_term)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I - skew, torch.inverse(I + skew))

        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out


######################################
# 2.2 Kron_R                         #
######################################
class KronQRInjectedLinear_R(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, r=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        self.lora = torch.nn.Parameter(torch.zeros(out_features, r))
        nn.init.normal_(self.lora, std=1 / r)
    
    def QR_decompose(self):
        W = self.linear.weight
        self.Q, self.R = torch.linalg.qr(W.t())
        with torch.no_grad():
            self.R = self.R - torch.triu(self.lora.to(self.R.device)@ self.lora.to(self.R.device).t())
        del self.linear.weight

    def forward(self, input):
        # rotate
        delta_R = torch.triu(self.lora @ self.lora.t())
        R = self.R + delta_R
        W_t = self.Q @ R
        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_qr_R(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_R(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=rank,
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.QR_decompose() # to perform QR decomposition

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora)

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].lora.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 2.2.a Kron_R_diag                  #
######################################
class KronQRInjectedLinear_R2(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, r=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        self.lora = nn.Parameter(torch.zeros(out_features))
    
    def QR_decompose(self):
        W = self.linear.weight
        self.Q, self.R = torch.linalg.qr(W.t())
        #with torch.no_grad():
        #    self.R = self.R - torch.triu(self.lora.to(self.R.device)@ self.lora.to(self.R.device).t())
        del self.linear.weight

    def forward(self, input):
        R = self.R + torch.diag(self.lora)
        W_t = self.Q @ R
        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_qr_R2(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_R2(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=rank,
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.QR_decompose() # to perform QR decomposition

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora)

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].lora.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 2.2 Kron_QR                         #
######################################
class KronQRInjectedLinear_QR(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # R
        self.lora = torch.nn.Parameter(torch.zeros(out_features, r_rank))
        nn.init.normal_(self.lora, std=1 / r_rank)

        # Q
        if in_features % q_rank == 0:
            size_kron_1, size_kron_2 = in_features // q_rank, q_rank
        else:
            raise ValueError("the in features channel number can not be divided by r, please choose another r")
        self.kron_1 = nn.Parameter(torch.eye(size_kron_1, size_kron_1))
        self.kron_2 = nn.Parameter(torch.eye(size_kron_2, size_kron_2))
    
    def QR_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.Q, self.R = torch.linalg.qr(W.t())
            self.R = self.R - torch.triu(self.lora.to(self.R.device)@ self.lora.to(self.R.device).t())
        #del self.linear.weight

    def forward(self, input):
        # rotate
        delta_R = torch.triu(self.lora @ self.lora.t())
        rotation_matrix = torch.kron(self.kron_1, self.kron_2)
        R = self.R + delta_R
        W_t = (rotation_matrix @ self.Q) @ R
        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_qr_QR(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_QR(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.QR_decompose() # to perform QR decomposition

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append([_module._modules[name].kron_1])
        require_grad_params.append([_module._modules[name].kron_2])
        require_grad_params.append([_module._modules[name].lora])

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].lora.requires_grad = True
        _module._modules[name].kron_1.requires_grad = True
        _module._modules[name].kron_2.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 2.3 Kron_QR (Q lambda QT)          #
######################################
class KronQRInjectedLinear_QR2(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # R
        if out_features % r_rank == 0:
            size_kron_1, size_kron_2 = out_features // r_rank, r_rank
        else:
            raise ValueError("the in features channel number can not be divided by r, please choose another r")
        self.kron_R1 = nn.Parameter(torch.eye(size_kron_1, size_kron_1))
        self.kron_R2 = nn.Parameter(torch.eye(size_kron_2, size_kron_2))
        nn.init.normal_(self.kron_R1, std=1 / size_kron_1)
        nn.init.normal_(self.kron_R2, std=1 / size_kron_2)

        self.lambda_matrix = nn.Parameter(torch.zeros(out_features))

        # Q
        if in_features % q_rank == 0:
            size_kron_1, size_kron_2 = in_features // q_rank, q_rank
        else:
            raise ValueError("the in features channel number can not be divided by r, please choose another r")
        self.kron_Q1 = nn.Parameter(torch.eye(size_kron_1, size_kron_1))
        self.kron_Q2 = nn.Parameter(torch.eye(size_kron_2, size_kron_2))
    
    def QR_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.Q, self.R = torch.linalg.qr(W.t())
            #self.R = self.R - torch.triu(self.lora.to(self.R.device)@ self.lora.to(self.R.device).t())

        del self.linear.weight

    def forward(self, input):
        # rotate
        rotation_matrix = torch.kron(self.kron_Q1, self.kron_Q2)
        # Q lambda QT
        orthogonal_matrix = torch.kron(self.kron_R1, self.kron_R2)
        R = self.R + orthogonal_matrix @ torch.diag(self.lambda_matrix) @ orthogonal_matrix.t()
        W_t = (rotation_matrix @ self.Q) @ R
        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_qr_QR2(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_QR2(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.QR_decompose() # to perform QR decomposition

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append([_module._modules[name].kron_Q1])
        require_grad_params.append([_module._modules[name].kron_Q2])
        require_grad_params.append([_module._modules[name].kron_R1])
        require_grad_params.append([_module._modules[name].kron_R2])
        require_grad_params.append([_module._modules[name].lambda_matrix])

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].kron_Q1.requires_grad = True
        _module._modules[name].kron_Q2.requires_grad = True
        _module._modules[name].kron_R1.requires_grad = True
        _module._modules[name].kron_R2.requires_grad = True
        _module._modules[name].lambda_matrix.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 2.4 Kron_SVD ()          #
######################################
class KronQRInjectedLinear_SVD(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # delta
        self.delta = nn.Parameter(torch.zeros(out_features))

        # Q
        if in_features % q_rank == 0:
            size_kron_1, size_kron_2 = in_features // q_rank, q_rank
        else:
            raise ValueError("the in features channel number can not be divided by r, please choose another r")
        self.kron_Q1 = nn.Parameter(torch.eye(size_kron_1, size_kron_1))
        self.kron_Q2 = nn.Parameter(torch.eye(size_kron_2, size_kron_2))
    
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
        orthogonal_matrix = torch.kron(self.kron_Q1, self.kron_Q2)
        W_t = orthogonal_matrix @ W.t()
        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_svd(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_SVD(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
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
        require_grad_params.append([_module._modules[name].delta])

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].kron_Q1.requires_grad = True
        _module._modules[name].kron_Q2.requires_grad = True
        _module._modules[name].delta.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 2.4.a Kron3_SVD                    #
######################################
class KronQRInjectedLinear_SVD3(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2, tmp_l=None,
        merge_kron=None, merge_lora=None
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        if merge_kron == None:
            # delta
            self.delta = nn.Parameter(torch.zeros(out_features))

            # init delta to be 1e3
            nn.init.constant_(self.delta, 1e-3)

            # Q
            # automatically get kron size
            #size_kron_Q1, size_kron_Q2, size_kron_Q3 = get_abc(in_features)
            svd_l=tmp_l
            size_kron_Q1, size_kron_Q2, size_kron_Q3 = svd_l[in_features]
            self.kron_Q1 = nn.Parameter(torch.eye(size_kron_Q1, size_kron_Q1))
            self.kron_Q2 = nn.Parameter(torch.eye(size_kron_Q2, size_kron_Q2))
            self.kron_Q3 = nn.Parameter(torch.eye(size_kron_Q3, size_kron_Q3))

        else:
            self.kron_Q1 = nn.Parameter(merge_kron[0])
            self.kron_Q2 = nn.Parameter(merge_kron[1])
            self.kron_Q3 = nn.Parameter(merge_kron[2])

            self.delta = nn.Parameter(merge_lora)
    
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

def inject_trainable_kron_svd3(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    tmp_l=None,
    db_list=None,
    style_list=None,
    db_lambda=1.0,
    style_lambda=1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    params_to_save = []
    names = []

    # merge lora_list and style_list
    if db_list is not None:
        db_kron = list(itertools.chain([db_list.pop(0)], [db_list.pop(0)], [db_list.pop(0)]))
        db_lora= db_list.pop(0)
        
        style_kron = list(itertools.chain([style_list.pop(0)], [style_list.pop(0)], [style_list.pop(0)]))
        style_lora = style_list.pop(0)

        merge_lora = db_lora * db_lambda + style_lora * style_lambda
        merge_kron = [db_kron[0] * db_lambda + style_kron[0] * style_lambda, db_kron[1] * db_lambda + style_kron[1] * style_lambda, db_kron[2] * db_lambda + style_kron[2] * style_lambda]
    else:
        merge_kron = None
        merge_lora = None

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_SVD3(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
            tmp_l=tmp_l,
            merge_kron=merge_kron,
            merge_lora=merge_lora
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

        params_to_save.append(_module._modules[name].kron_Q1)
        params_to_save.append(_module._modules[name].kron_Q2)
        params_to_save.append(_module._modules[name].kron_Q3)
        params_to_save.append(_module._modules[name].delta)

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].kron_Q1.requires_grad = True
        _module._modules[name].kron_Q2.requires_grad = True
        _module._modules[name].kron_Q3.requires_grad = True
        _module._modules[name].delta.requires_grad = True
        names.append(name)

    return require_grad_params, params_to_save, names

######################################
# 2.4.b Kron3_QR                    #
######################################
class KronQRInjectedLinear_QR3(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # delta
        self.delta = nn.Parameter(torch.zeros(out_features))

        # Q
        # automatically get kron size
        #size_kron_Q1, size_kron_Q2, size_kron_Q3 = get_abc(in_features)
        size_kron_Q1, size_kron_Q2, size_kron_Q3 = svd_l[in_features]
        self.kron_Q1 = nn.Parameter(torch.eye(size_kron_Q1, size_kron_Q1))
        self.kron_Q2 = nn.Parameter(torch.eye(size_kron_Q2, size_kron_Q2))
        self.kron_Q3 = nn.Parameter(torch.eye(size_kron_Q3, size_kron_Q3))
    
    def QR_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.Q, self.R = torch.linalg.qr(W.t())

        del self.linear.weight

    def forward(self, input):
        # add residual
        R = self.R + torch.diag(self.delta)
      
        # rotate
        orthogonal_matrix = torch.kron(torch.kron(self.kron_Q1, self.kron_Q2), self.kron_Q3)
        W_t = (orthogonal_matrix @ self.Q) @ R

        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_qr3(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    params_to_save = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_QR3(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.QR_decompose() # to perform QR decomposition

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append([_module._modules[name].kron_Q1])
        require_grad_params.append([_module._modules[name].kron_Q2])
        require_grad_params.append([_module._modules[name].kron_Q3])
        require_grad_params.append([_module._modules[name].delta])

        params_to_save.append(_module._modules[name].kron_Q1)
        params_to_save.append(_module._modules[name].kron_Q2)
        params_to_save.append(_module._modules[name].kron_Q3)
        params_to_save.append(_module._modules[name].delta)

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].kron_Q1.requires_grad = True
        _module._modules[name].kron_Q2.requires_grad = True
        _module._modules[name].kron_Q3.requires_grad = True
        _module._modules[name].delta.requires_grad = True
        names.append(name)

    return require_grad_params, params_to_save, names

######################################
# 2.4.c Kron3_SVD_Residual          #
######################################
class KronQRInjectedLinear_SVD_RE(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2, tmp_l=None
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # delta
        self.delta = nn.Parameter(torch.zeros(out_features))

        # Q
        # automatically get kron size
        #size_kron_Q1, size_kron_Q2, size_kron_Q3 = get_abc(in_features)
        svd_l=tmp_l
        size_kron_Q1, size_kron_Q2, size_kron_Q3 = svd_l[in_features]
        self.kron_Q1 = nn.Parameter(torch.eye(size_kron_Q1, size_kron_Q1))
        self.kron_Q2 = nn.Parameter(torch.eye(size_kron_Q2, size_kron_Q2))
        self.kron_Q3 = nn.Parameter(torch.eye(size_kron_Q3, size_kron_Q3))

        nn.init.orthogonal_(self.kron_Q1)
        nn.init.orthogonal_(self.kron_Q2)
        nn.init.orthogonal_(self.kron_Q3)
    
    def SVD_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.U, self.S, self.V = torch.linalg.svd(W, full_matrices=False)
            # re-initialize self.U and V in a orthogonal way
            nn.init.orthogonal_(self.U)
            nn.init.orthogonal_(self.V)
            nn.init.constant_(self.delta, torch.min(self.S))

    def forward(self, input):
        W = self.linear.weight

        # compute svd residual
        orthogonal_matrix = torch.kron(torch.kron(self.kron_Q1, self.kron_Q2), self.kron_Q3)
        delta_W = self.U @ F.relu(torch.diag(self.delta)) @ self.V
        delta_W_t = orthogonal_matrix @ delta_W.t()
        
        W_t = W.t() + delta_W_t

        out = self.dropout(F.linear(input, W_t.t(), bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_svd_re(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    tmp_l=None,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    params_to_save = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_SVD_RE(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
            tmp_l=tmp_l,
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

        params_to_save.append(_module._modules[name].kron_Q1)
        params_to_save.append(_module._modules[name].kron_Q2)
        params_to_save.append(_module._modules[name].kron_Q3)
        params_to_save.append(_module._modules[name].delta)

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].kron_Q1.requires_grad = True
        _module._modules[name].kron_Q2.requires_grad = True
        _module._modules[name].kron_Q3.requires_grad = True
        _module._modules[name].delta.requires_grad = True
        names.append(name)

    return require_grad_params, params_to_save, names


######################################
# 3. OFT                           #
######################################

# read rl.txt which contains the following: 32 16 32 ... into a list that contains [32, 16, 32, ...]
def read_rl_txt(file_path):
    with open(file_path, 'r') as f:
        rl = f.readline().split()
    return [int(r) for r in rl]

rl = read_rl_txt('rl.txt')

def inject_trainable_oft(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 2,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    block_share: bool = True,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias

        # use fixed r from rl.txt
        r = rl.pop(0)

        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = OFTLinearLayer(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            block_share=block_share,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].R)

        _module._modules[name].R.requires_grad = True
        names.append(name)

    return require_grad_params, names

class OFTLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_share=False, eps=6e-5, r=4, is_coft=False):
        super(OFTLinearLayer, self).__init__()

        # Define the reduction rate:
        self.r = r
        
        # Check whether to use the constrained variant COFT 
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        # self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        # self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        
        # Define the fixed Linear layer: v
        # self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        self.linear = nn.Linear(in_features, out_features, bias)

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        self.block_share = block_share
        # Define the trainable matrix parameter: R
        if self.block_share:
            # Initialized as an identity matrix
            self.R_shape = [in_features // self.r, in_features // self.r]
            self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [self.r, in_features // self.r, in_features // self.r]
            R = torch.zeros(self.R_shape[1], self.R_shape[1])
            R = torch.stack([R] * self.r)
            self.R = nn.Parameter(R, requires_grad=True)
            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, x):
        orig_dtype = x.dtype
        dtype = self.R.dtype
        attn = self.linear

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = attn.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
 
        # Apply the trainable identity matrix
        bias_term = attn.bias.data if attn.bias is not None else None
        if bias_term is not None:
            bias_term = bias_term.to(orig_dtype)

        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias_term)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I - skew, torch.inverse(I + skew))

        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out

######################################
# 3.a OFT-Stiefel                    #
######################################

# read rl.txt which contains the following: 32 16 32 ... into a list that contains [32, 16, 32, ...]
def read_rl_txt(file_path):
    with open(file_path, 'r') as f:
        rl = f.readline().split()
    return [int(r) for r in rl]

rl = read_rl_txt('rl.txt')

def inject_trainable_oft_st(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 2,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    block_share: bool = True,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias

        # use fixed r from rl.txt
        r = rl.pop(0)

        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = OFTLinearLayer_st(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            block_share=block_share,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].R)

        _module._modules[name].R.requires_grad = True
        names.append(name)

    return require_grad_params, names

class OFTLinearLayer_st(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_share=False, eps=6e-5, r=4, is_coft=False):
        super(OFTLinearLayer_st, self).__init__()

        # Define the reduction rate:
        self.r = r
        
        # Check whether to use the constrained variant COFT 
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        # self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        # self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        
        # Define the fixed Linear layer: v
        # self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        self.linear = nn.Linear(in_features, out_features, bias)

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        self.block_share = block_share
        # Define the trainable matrix parameter: R
        if self.block_share:
            # Initialized as an identity matrix
            self.R_shape = [in_features // self.r, in_features // self.r]
            self.R = nn.Parameter(torch.eye(self.R_shape[0], self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [self.r, in_features // self.r, in_features // self.r]
            R = torch.zeros(self.R_shape[1], self.R_shape[1])
            R = torch.stack([R] * self.r)
            self.R = nn.Parameter(R, requires_grad=True)
            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, x):
        orig_dtype = x.dtype
        dtype = self.R.dtype
        attn = self.linear

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.R
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.R

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = attn.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
 
        # Apply the trainable identity matrix
        bias_term = attn.bias.data if attn.bias is not None else None
        if bias_term is not None:
            bias_term = bias_term.to(orig_dtype)

        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias_term)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I - skew, torch.inverse(I + skew))

        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out

######################################
# 4.SVD                              #
######################################
class InjectedLinear_SVD(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

        # delta
        self.delta = nn.Parameter(torch.zeros(out_features))
    
    def SVD_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.U, self.S, self.V = torch.linalg.svd(W, full_matrices=False)
            # print the first 10 singular values
            #print(self.S)

        del self.linear.weight

    def forward(self, input):
        # add residual
        #S = F.relu(self.S + self.delta)
        #S = self.S + self.delta
        S = F.softplus(self.S + self.delta, beta=5)
        W = self.U @ torch.diag(S) @ self.V
        out = self.dropout(F.linear(input, W, bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_svd(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = InjectedLinear_SVD(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.SVD_decompose() # to perform QR decomposition

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append([_module._modules[name].delta])

        if loras != None:
            _module._modules[name].lora = loras.pop(0)

        _module._modules[name].delta.requires_grad = True
        names.append(name)

    return require_grad_params, names

######################################
# 4. Utils                            #
######################################
def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedLinear,
        LoraInjectedConv2d,
    ],
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

def count_params(params):

    total_params = 0
    for param in list(params):
        for p in list(param):
            if p.requires_grad:
                # param.numel() gives the total number of elements in the parameter tensor, 
                # which corresponds to the number of trainable parameters for that tensor.
                total_params += p.numel()
    
    return total_params

def save_safeloras(
    modelmap: Dict[str, nn.Module],
    filename: str,
    target_class
):
    weights = {}
    metadata = {}

    for name, (model, target_replace_module) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        # extract all the target_replace_module
        for _module, name, _child_module in _find_modules(
            model, target_replace_module, search_class=target_class
        ):
            weights[name] = _child_module.realize_as_lora()

######################################
# 5.1 Kron3_SVD_merge                    #
######################################
class KronQRInjectedLinear_SVD3_merge(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2, tmp_l=None,
        merge_kron=None, merge_lora=None
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

    def SVD_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.U, self.S, self.V = torch.linalg.svd(W, full_matrices=False)

    def forward(self, input):
        W = self.linear.weight
        out = self.dropout(F.linear(input, W, bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_kron_svd3_merge(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    tmp_l=None,
    db_list=None,
    style_list=None,
    db_lambda=1.0,
    style_lambda=1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    params_to_save = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = KronQRInjectedLinear_SVD3_merge(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
            tmp_l=tmp_l,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.SVD_decompose() # to perform QR decomposition

        db_kron = list(itertools.chain([db_list.pop(0)], [db_list.pop(0)], [db_list.pop(0)]))
        db_lora= db_list.pop(0)
        
        style_kron = list(itertools.chain([style_list.pop(0)], [style_list.pop(0)], [style_list.pop(0)]))
        style_lora = style_list.pop(0)

        # calculate residual W
        W_0 = weight
        S = F.relu(_tmp.S + db_lora)
        W_1 = _tmp.U @ torch.diag(S) @ _tmp.V
        # rotate
        orthogonal_matrix = torch.kron(torch.kron(db_kron[0], db_kron[1]), db_kron[2])
        W_t = orthogonal_matrix @ W_1.t()
        residul_W_db = W_t.t() - W_0

        S = F.relu(_tmp.S + style_lora)
        W_1 = _tmp.U @ torch.diag(S) @ _tmp.V
        # rotate
        orthogonal_matrix = torch.kron(torch.kron(style_kron[0], style_kron[1]), style_kron[2])
        W_t = orthogonal_matrix @ W_1.t()
        residul_W_style = W_t.t() - W_0

        merge_W_residual = residul_W_db * db_lambda + residul_W_style * style_lambda

        W_merge = W_0 + merge_W_residual
        _tmp.linear.weight = nn.Parameter(W_merge)

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        del _child_module.weight
        del _tmp.U
        del _tmp.S
        del _tmp.V

    return 

######################################
# 5.2 SVD_merge                    #
######################################
class InjectedLinear_SVD_merge(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2, tmp_l=None,
        merge_kron=None, merge_lora=None
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

    def SVD_decompose(self):
        W = self.linear.weight
        with torch.no_grad():
            self.U, self.S, self.V = torch.linalg.svd(W, full_matrices=False)

    def forward(self, input):
        S = F.relu(self.S + self.lora)
        W = self.U @ torch.diag(S) @ self.V
        out = self.dropout(F.linear(input, W, bias=self.linear.bias))
        return (
            out * self.scale
        )

def inject_trainable_svd_merge(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    tmp_l=None,
    db_list=None,
    style_list=None,
    db_lambda=1.0,
    style_lambda=1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    params_to_save = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = InjectedLinear_SVD_merge(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
            tmp_l=tmp_l,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
        _tmp.SVD_decompose() # to perform QR decomposition


        db_lora= db_list.pop(0)
        style_lora = style_list.pop(0)
        #print(torch.min(db_lora))

        # calculate merge lora

        merge_lora = db_lora * db_lambda + style_lora * style_lambda
        _tmp.lora = nn.Parameter(merge_lora)

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        del _child_module.weight

    return 

######################################
# 5.3 lora_merge                    #
######################################
class InjectedLinear_DB_merge(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, dropout_p=0.1, scale=1.0, q_rank=2, r_rank=2, tmp_l=None,
        merge_kron=None, merge_lora=None
    ):
        super().__init__()
        r=1
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)

        self.db_lora_down = nn.Linear(in_features, r, bias=False)
        self.db_lora_up = nn.Linear(r, out_features, bias=False)
        self.style_lora_down = nn.Linear(in_features, r, bias=False)
        self.style_lora_up = nn.Linear(r, out_features, bias=False)

        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale
        self.selector = nn.Identity()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale

    def forward(self, input):
        db_residual = self.dropout(self.db_lora_up(self.selector(self.db_lora_down(input))))
        style_residual = self.dropout(self.style_lora_up(self.selector(self.style_lora_down(input))))

        residual = db_residual * self.db_lambda + style_residual * self.style_lambda

        return (
            self.linear(input)
            + residual
            * self.scale
        )

def inject_trainable_db_merge(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    q_rank: int = 4,
    r_rank: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    tmp_l=None,
    db_list=None,
    style_list=None,
    db_lambda=1.0,
    style_lambda=1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    params_to_save = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = InjectedLinear_DB_merge(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r_rank=r_rank,
            q_rank=q_rank, 
            dropout_p=dropout_p,
            scale=scale,
            tmp_l=tmp_l,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        db_lora_up= db_list.pop(0)
        db_lora_down = db_list.pop(0)
        style_lora_up = style_list.pop(0)
        style_lora_down = style_list.pop(0)

        # calculate merge lora

        _tmp.db_lora_up.weight = nn.Parameter(db_lora_up)
        _tmp.db_lora_down.weight = nn.Parameter(db_lora_down)
        _tmp.style_lora_up.weight = nn.Parameter(style_lora_up)
        _tmp.style_lora_down.weight = nn.Parameter(style_lora_down)
        _tmp.db_lambda = db_lambda
        _tmp.style_lambda = style_lambda

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

    return 