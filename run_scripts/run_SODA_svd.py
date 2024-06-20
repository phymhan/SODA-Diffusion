# system
import argparse
import os, sys

# config the path
script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(script_directory)
util_directory = os.path.join(root_directory, "utils")
if root_directory not in sys.path:
    sys.path.append(root_directory)
if util_directory not in sys.path:
    sys.path.append(util_directory)

import logging
import itertools
import math
from tqdm.auto import tqdm
import numpy as np
import json

# accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger

# diffusers
import diffusers
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler

# transformers
import transformers
from transformers import AutoTokenizer, PretrainedConfig

# torch
import torch
import torch.nn.functional as F

# peft
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# my_scripts 
from utils.pefts import (
    inject_trainable_kron_svd3, 
)
from utils.data import TrainDataset, collate_fn
from utils.StiefelOptimizers import StiefelAdam

svd_l = {
    640:  (8, 8, 10),
    2048: (4, 8,  64),
    1280: (4, 16, 20),
    768:  (6, 8,  16)
} # keep the param same as svd

logger = get_logger(__name__)
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # pretrained model config
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",)
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    # data config
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--place_holder", type=str, default=" sks ")
    # validation config
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is used during validation to verify that the model is learning.",)
    parser.add_argument("--num_validation_images", type=int, default=0, help="Number of images that should be generated during validation with `validation_prompt`.",)
    parser.add_argument("--validation_epochs", type=int, default=50000)
    parser.add_argument("--validation_starting_epochs", type=int, default=50000)
    # use prior preservation
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.",)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    # save config
    parser.add_argument("--output_dir", type=str, default="outdir", help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # dataloader config
    parser.add_argument("--resolution", type=int, default=1024, help="The resolution for input images, all the images in the train/validation dataset will be resized to this")
    parser.add_argument("--crops_coords_top_left_h", type=int, default=0, help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),)
    parser.add_argument("--crops_coords_top_left_w", type=int, default=0, help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),)
    parser.add_argument("--center_crop", default=False, action="store_true", help=("Whether to center crop the input images to the resolution. If not set, the images will be randomly"    " cropped. The images will be resized to the resolution first before cropping."),)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."),)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint.")
    # train config
    parser.add_argument("--dcoloss_beta", type=float, default=1000, help="Sigloss value for DCO loss, use -1 if do not using dco loss")
    parser.add_argument("--train_text_encoder_ti", action="store_true", help=("Whether to use textual inversion"),)
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",)    
    # optimizer config
    parser.add_argument("--which_optim", type=str, default="adam", help="Type for Optimizer [adam, adamw, sgd]",)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--delta_learning_rate", type=float, default=5e-5, help="Initial learning rate for the delta parameters.",)
    parser.add_argument("--text_encoder_lr", type=float, default=5e-6, help="Text encoder learning rate to use.",)
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'    ' "constant", "constant_with_warmup"]'),)
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    # optimizer config
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer and Prodigy optimizers.",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # save config
    parser.add_argument("--logging_dir", type=str, default="logs", help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"    " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
    parser.add_argument("--allow_tf32", action="store_true", help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"    " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),)
    parser.add_argument("--report_to", type=str, default="tensorboard", help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'    ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="    " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"    " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--rank", type=int, default=32, help=("The dimension of the LoRA update matrices."),)
    parser.add_argument("--offset_noise", type=float, default=0.0)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--sample_total_num", type=int, default=8)
    # wandb config
    parser.add_argument("--wandb_key", type=str, default="9d61360e0722073614d3edd016df312b3f6e2aa2")
    parser.add_argument("--wandb_project_name", type=str, default="QR-Dreambooth")
    parser.add_argument("--wandb_group_name", type=str, default=None)
    parser.add_argument("--wandb_exp_name", type=str, default=None)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def main(args):
    #####################################
    # 0.0 Init                          #
    #####################################
    logging_dir = os.path.join(args.output_dir, args.wandb_group_name, args.wandb_exp_name)
    os.makedirs(logging_dir, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, # default fp16
        #log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )


    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    #####################################
    # 1.0 Load Pretrained Models        #
    #####################################         
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        variant=args.variant,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        variant=args.variant,
        use_fast=False,
    )
    
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")    
    
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    vae_scaling_factor = vae.config.scaling_factor

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    unet_freeze = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    #####################################
    # 1.1 Preprocess Pretrained Models  #
    #####################################

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    unet_freeze.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    unet_freeze.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
    
    #####################################
    # 1.2 Register PEFT Modules         #
    #####################################
    unet_kron_qr_params, _, _ = inject_trainable_kron_svd3(
            unet, loras=None, tmp_l=svd_l
        )

    unet_kron_qr_params_tosave = list(itertools.chain(unet_kron_qr_params))
    unet_kron_qr_params_kron_1 = itertools.islice(unet_kron_qr_params_tosave, 0, None, 4)
    unet_kron_qr_params_kron_2 = itertools.islice(unet_kron_qr_params_tosave, 1, None, 4) 
    unet_kron_qr_params_kron_3 = itertools.islice(unet_kron_qr_params_tosave, 2, None, 4)
    unet_kron_qr_params_lora = itertools.islice(unet_kron_qr_params_tosave, 3, None, 4) 
    # merge kron123
    unet_kron_qr_params_kron = itertools.chain(unet_kron_qr_params_kron_1, unet_kron_qr_params_kron_2, unet_kron_qr_params_kron_3)
    
    if args.train_text_encoder:

        text_encoder_one_kron_qr_params, text_encoder_one_params_to_save, _ = inject_trainable_kron_svd3(
            text_encoder_one,
            target_replace_module=["CLIPAttention"],tmp_l=svd_l
        )
        # use itertools to slize as the unet did
        text_encoder_one_kron_qr_params_tosave = list(itertools.chain(text_encoder_one_kron_qr_params))
        text_encoder_one_kron_qr_params_kron_1 = itertools.islice(text_encoder_one_kron_qr_params_tosave, 0, None, 4)
        text_encoder_one_kron_qr_params_kron_2 = itertools.islice(text_encoder_one_kron_qr_params_tosave, 1, None, 4)
        text_encoder_one_kron_qr_params_kron_3 = itertools.islice(text_encoder_one_kron_qr_params_tosave, 2, None, 4)
        text_encoder_one_kron_qr_params_lora = itertools.islice(text_encoder_one_kron_qr_params_tosave, 3, None, 4)
        # merge kron123
        text_encoder_one_kron_qr_params_kron = itertools.chain(text_encoder_one_kron_qr_params_kron_1, text_encoder_one_kron_qr_params_kron_2, text_encoder_one_kron_qr_params_kron_3)

        # inject text encoder 2
        text_encoder_two_kron_qr_params, text_encoder_two_params_to_save, _ = inject_trainable_kron_svd3(
            text_encoder_two,
            target_replace_module=["CLIPAttention"],tmp_l=svd_l
        )
        # use itertools to slize as the unet did
        text_encoder_two_kron_qr_params_tosave = list(itertools.chain(text_encoder_two_kron_qr_params))
        text_encoder_two_kron_qr_params_kron_1 = itertools.islice(text_encoder_two_kron_qr_params_tosave, 0, None, 4)
        text_encoder_two_kron_qr_params_kron_2 = itertools.islice(text_encoder_two_kron_qr_params_tosave, 1, None, 4)
        text_encoder_two_kron_qr_params_kron_3 = itertools.islice(text_encoder_two_kron_qr_params_tosave, 2, None, 4)
        text_encoder_two_kron_qr_params_lora = itertools.islice(text_encoder_two_kron_qr_params_tosave, 3, None, 4)
        # merge kron123
        text_encoder_two_kron_qr_params_kron = itertools.chain(text_encoder_two_kron_qr_params_kron_1, text_encoder_two_kron_qr_params_kron_2, text_encoder_two_kron_qr_params_kron_3)
    
    # Make sure the trainable params are in float32.
    total_params = 0
    
    models = [unet]
    if args.train_text_encoder:
        models.extend([text_encoder_one, text_encoder_two])
    for model in models:
        for param in model.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                if args.mixed_precision == "fp16":
                    param.data = param.to(torch.float32)  
                total_params += param.numel()
    
    print(total_params)
    
    #####################################
    # 1.3 Register Save-Load Hookers    #
    #####################################
    def save_peft(path):
        unet_param_list =unet_kron_qr_params_tosave
        text_encoder_one_param_list = text_encoder_one_kron_qr_params_tosave
        text_encoder_two_param_list = text_encoder_two_kron_qr_params_tosave
        table = {
            "unet": unet_param_list,
            "text_encoder_one": text_encoder_one_param_list,
            "text_encoder_two": text_encoder_two_param_list,
        }
        # save the table as pt
        torch.save(table, path)

    #####################################
    # 2.1 Setting Optimizers            #
    #####################################
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    if args.which_optim == 'adam':
        optimizer_class = StiefelAdam
        optim_args = dict(
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            #weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lora_optim_class = torch.optim.AdamW
        lora_optim_args = dict(
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    text_lr = (
        args.learning_rate
        if args.text_encoder_lr is None
        else args.text_encoder_lr
    )

    unet_kron_qr_params_kron_list  = list(itertools.chain(*unet_kron_qr_params_kron))
    unet_kron_qr_params_lora_list  = list(itertools.chain(*unet_kron_qr_params_lora))
    # chain the text encoders as well
    if args.train_text_encoder:
        text_encoder_one_kron_qr_params_kron_list = list(itertools.chain(*text_encoder_one_kron_qr_params_kron))
        text_encoder_one_kron_qr_params_lora_list = list(itertools.chain(*text_encoder_one_kron_qr_params_lora))
        text_encoder_two_kron_qr_params_kron_list = list(itertools.chain(*text_encoder_two_kron_qr_params_kron))
        text_encoder_two_kron_qr_params_lora_list = list(itertools.chain(*text_encoder_two_kron_qr_params_lora))

    # kron
    params_to_optimize_kron = (
        [
            {"params": unet_kron_qr_params_kron_list, "lr": args.learning_rate},
            {"params": text_encoder_one_kron_qr_params_kron_list, "lr": text_lr},
            {"params": text_encoder_two_kron_qr_params_kron_list, "lr": text_lr},
        ]
        if args.train_text_encoder
        else unet_kron_qr_params_kron_list
    )

    optim_args['lr'] = args.learning_rate
    optimizer_kron = optimizer_class(
            params_to_optimize_kron,
            **optim_args,
        )

    # lora
    params_to_optimize_lora = (
        [
            {"params": unet_kron_qr_params_lora_list, "lr": args.delta_learning_rate},
            {"params": text_encoder_one_kron_qr_params_lora_list, "lr": args.delta_learning_rate*0.1},
            {"params": text_encoder_two_kron_qr_params_lora_list, "lr": args.delta_learning_rate*0.1},
        ]
        if args.train_text_encoder
        else unet_kron_qr_params_lora_list
    )

    lora_optim_args['lr'] = args.delta_learning_rate
    optimizer_svd = lora_optim_class(
        params_to_optimize_lora,
        **lora_optim_args,
    )

    #####################################
    # 2.3 Setting Datasets              #
    #####################################
    # Dataset and DataLoaders creation:
    train_dataset = TrainDataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args),
        num_workers=args.dataloader_num_workers,
    )

    #####################################
    # 2.4 Prepare for SDXL              #
    #####################################
    # Computes additional embeddings/ids required by the SDXL UNet.
    def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    # Handle instance prompt.
    instance_time_ids = compute_time_ids()
    add_time_ids = instance_time_ids
    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        add_time_ids = torch.cat([add_time_ids, class_time_ids], dim=0)
        
    #####################################
    # 2.5 Prepare for Training          #
    #####################################
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_kron = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_kron,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)
    if not freeze_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer_kron, optimizer_svd, train_dataloader, lr_scheduler_kron = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer_kron, optimizer_svd, train_dataloader, lr_scheduler_kron
        )
    else:
        unet, optimizer_kron_1, optimizer_kron_2, train_dataloader, lr_scheduler_kron_1, lr_scheduler_kron_2 = accelerator.prepare(
            unet, optimizer_kron_1, optimizer_kron_2, train_dataloader, lr_scheduler_kron_1, lr_scheduler_kron_2
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    #if accelerator.is_main_process:
    #    accelerator.init_trackers("fine-tune sdxl", config=vars(args))
    
    #####################################
    # 3.1 Prepare for Training          #
    #####################################
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    #####################################
    # 3.2 Wandb                         #
    #####################################
    if args.report_to =='wandb' and args.wandb_key is not None:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_exp_name,
            config=args,
            group=args.wandb_group_name
        )

        wandb.log({'params_num': total_params}, commit=False)

    #####################################
    # 3.3 Training Loop                 #
    #####################################
    for epoch in range(first_epoch, args.num_train_epochs):
        # if performing any kind of optimization of text_encoder params
        if args.train_text_encoder or args.train_text_encoder_ti:
            text_encoder_one.train()
            text_encoder_two.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            if args.train_text_encoder:
                text_encoder_one.text_model.embeddings.requires_grad_(True)
                text_encoder_two.text_model.embeddings.requires_grad_(True)

        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                prompts = batch["prompts"]
                if args.train_text_encoder_ti and (args.dcoloss_beta > 0.0):
                    base_prompts = batch["base_prompts"]
                    base_prompt_embeds, base_add_embeds = compute_text_embeddings(
                        base_prompts, text_encoders, tokenizers
                    )
                # encode batch prompts when custom prompts are provided for each image -
                # if train_dataset.custom_instance_prompts:
                if freeze_text_encoder:
                    prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                        prompts, text_encoders, tokenizers
                    )
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts)
                    tokens_two = tokenize_prompt(tokenizer_two, prompts)

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = model_input * vae_scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                noise = noise + args.offset_noise * torch.randn(model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                elems_to_repeat_text_embeds = 1
                elems_to_repeat_time_ids = bsz // 2 if args.with_prior_preservation else bsz

                # Predict the noise residual
                if freeze_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1),
                        "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                    }
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
                    if args.dcoloss_beta > 0.0:
                        with torch.no_grad():
                            #cross_attention_kwargs = {"scale": 0.0}
                            refer_pred = unet_freeze(
                                noisy_model_input, 
                                timesteps, 
                                prompt_embeds_input,
                                added_cond_kwargs=unet_added_conditions,
                                #cross_attention_kwargs=cross_attention_kwargs,
                            ).sample
                else:
                    unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1)}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        noisy_model_input, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                    ).sample
                    if args.dcoloss_beta > 0.0:
                        base_prompts = batch["base_prompts"]
                        with torch.no_grad():
                            base_prompt_embeds, base_add_embeds = compute_text_embeddings(
                                base_prompts, text_encoders, tokenizers
                            )
                            #cross_attention_kwargs = {"scale": 0.0}
                            base_added_conditions = {"time_ids": add_time_ids, "text_embeds": base_add_embeds}
                            refer_pred = unet_freeze(
                                noisy_model_input, 
                                timesteps, 
                                base_prompt_embeds, 
                                added_cond_kwargs=base_added_conditions, 
                                #cross_attention_kwargs=cross_attention_kwargs
                            ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                if args.snr_gamma is None:
                    if args.dcoloss_beta > 0.0:
                        loss_model = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        loss_refer = F.mse_loss(refer_pred.float(), target.float(), reduction="mean")
                        diff = loss_model - loss_refer
                        inside_term = -1 * args.dcoloss_beta * diff
                        loss = -1 * torch.nn.LogSigmoid()(inside_term)                     
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    if args.with_prior_preservation:
                        # if we're using prior preservation, we calc snr for instance loss only -
                        # and hence only need timesteps corresponding to instance images
                        snr_timesteps, _ = torch.chunk(timesteps, 2, dim=0)
                    else:
                        snr_timesteps = timesteps

                    snr = compute_snr(noise_scheduler, snr_timesteps)
                    base_weight = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight
                        
                    if args.dcoloss_beta > 0.0:
                        loss_model = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss_model = loss_model.mean(dim=list(range(1, len(loss_model.shape)))) * mse_loss_weights
                        loss_model = loss_model.mean()
                        loss_refer = F.mse_loss(refer_pred.float(), target.float(), reduction="none")
                        loss_refer = loss_refer.mean(dim=list(range(1, len(loss_refer.shape)))) * mse_loss_weights
                        loss_refer = loss_refer.mean()
                        diff = loss_model - loss_refer
                        inside_term = -1 * args.dcoloss_beta * diff
                        loss = -1 * torch.nn.LogSigmoid()(inside_term)                     
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()
                    
                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_kron_qr_params_kron_list, unet_kron_qr_params_lora_list,
                                        text_encoder_one_kron_qr_params_kron_list, text_encoder_one_kron_qr_params_lora_list,
                                        text_encoder_two_kron_qr_params_kron_list, text_encoder_two_kron_qr_params_lora_list)
                        if args.train_text_encoder
                        else itertools.chain(unet_kron_qr_params_kron_1_list, unet_kron_qr_params_kron_2_list)
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if global_step % 2 == 0:
                    optimizer_svd.step()
                else:
                    optimizer_kron.step()
                lr_scheduler_kron.step()
                optimizer_kron.zero_grad()
                optimizer_svd.zero_grad()

                # every step, we reset the embeddings to the original embeddings.
                if args.train_text_encoder_ti:
                    embedding_handler.retract_embeddings()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # TODO implement saving
                """if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")"""

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler_kron.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                if global_step % args.validation_epochs == 0 and global_step >= args.validation_starting_epochs:
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        #f" {args.validation_prompt}."
                    )
                    # create pipeline
                    if freeze_text_encoder:
                        text_encoder_one = text_encoder_cls_one.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        text_encoder_two = text_encoder_cls_two.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder_2",
                            revision=args.revision,
                            variant=args.variant,
                        )
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                        safety_checker=None,
                    )

                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                    scheduler_args = {}

                    if "variance_type" in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_args["variance_type"] = variance_type

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config, **scheduler_args
                    )

                    pipeline = pipeline.to(accelerator.device)
                    #pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    with open(args.config_dir, 'r') as data_config:
                        js_table = json.load(data_config)[args.config_name]
                        eval_prompts = js_table['eval_prompts']
                        class_name = js_table['class']
                    
                    for prompt_id, eval_prompt in enumerate(eval_prompts):

                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                        if args.place_holder == "":
                            pipeline_args = {"prompt": [eval_prompt.replace(class_name, f'{class_name}')] * args.sample_batch_size}
                        else:
                            pipeline_args = {"prompt": [eval_prompt.replace(class_name, f'{args.place_holder} {class_name}')] * args.sample_batch_size}

                        images = []
                        with torch.cuda.amp.autocast():
                            for i in range(args.sample_total_num // args.sample_batch_size):
                                images.extend(pipeline(**pipeline_args, generator=generator).images)

                        for i, image in enumerate(images):
                            image.save(os.path.join(logging_dir, f'{global_step}_{prompt_id}_{i}.png'))

                        if args.report_to == "wandb":
                            wandb.log(
                                {
                                    eval_prompt.replace(class_name, f'{args.place_holder} {class_name}'): [
                                        wandb.Image(image) for i, image in enumerate(images)
                                    ]
                                }, commit=False if prompt_id < len(eval_prompts) - 1 else True
                            )

                    del pipeline
                    torch.cuda.empty_cache()
                    save_peft(os.path.join(logging_dir, f'peft_{global_step}.pt'))
            
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)