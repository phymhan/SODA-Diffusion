from torch.utils.data import Dataset
from torchvision import transforms
import json
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch

class TrainDataset(Dataset):
    def __init__(self, args):
        self.size = args.resolution
        self.center_crop = args.center_crop
        self.config_dir = args.config_dir
        self.config_name = args.config_name
        self.train_with_dco_loss = (args.dcoloss_beta > 0.)
        self.train_text_encoder_ti = args.train_text_encoder_ti
        self.with_prior_preservation = args.with_prior_preservation
        self.place_holder = args.place_holder
        
        with open(self.config_dir, 'r') as data_config:
            data_cfg = json.load(data_config)[self.config_name]
        
        self.instance_images = [Image.open(path) for path in data_cfg["images"]]
        self.class_name = data_cfg["class"]

        if self.place_holder == "":
            self.instance_prompts = [prompt for prompt in data_cfg["caption"]]
        else:
            self.instance_prompts = [prompt.replace(data_cfg["class"], f'{self.place_holder} {data_cfg["class"]}') for prompt in data_cfg["caption"]]
        
        if self.train_with_dco_loss:
            self.base_prompts = [prompt for prompt in data_cfg["caption"]]
        
        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images
        
        if self.with_prior_preservation:
            self.num_class_images = args.num_class_images
            class_dir = data_cfg["class_images_dir"]
            self.class_images = [Image.open(class_dir+f"/{i}.png") for i in range(self.num_class_images)]
            self.class_prompts = [prompt for prompt in data_cfg["class_prompts"]]
            self._length = max(self.num_class_images, self.num_instance_images)
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images[index % self.num_instance_images]
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        
        prompt = self.instance_prompts[index % self.num_instance_images]
        example["instance_prompt"] = prompt
        if self.train_with_dco_loss:
            base_prompt = self.base_prompts[index % self.num_instance_images]
            example["base_prompt"] = base_prompt
        
        if self.with_prior_preservation:
            class_image = self.class_images[index % self.num_class_images]
            class_image = exif_transpose(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example

def collate_fn(examples, args):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    
    if args.dcoloss_beta > 0.:
        base_prompts = [example["base_prompt"] for example in examples]

    if args.with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        if args.dcoloss_beta > 0.0:
            base_prompts += [example["class_prompt"] for example in examples]
 
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    if args.dcoloss_beta > 0.0:
        batch.update({"base_prompts": base_prompts})
    return batch