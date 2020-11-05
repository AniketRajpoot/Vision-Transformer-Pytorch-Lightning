# Vision Transformer in PyTorch Lightning
This is a third party implementation of the **Visual Transformer [paper](https://arxiv.org/pdf/2010.11929.pdf)** in 
**[PyTorch Lightning](https://www.pytorchlightning.ai)** with focus on transparency in training/fine-tuning the model.  
Heavily based on Google's [official implementation in Flax](https://github.com/google-research/vision_transformer)

## Features to be implemented:
- [:heavy_check_mark:] Architecture as PyTorch modules.
> TODO: Sparse and Linear Transformers utilities
- [:heavy_check_mark:] Support for loading checkpoints (i.e, pretrained weights) saved as .npz by Flax model into an identical PyTorch model in terms of architecture and naming conventions.
> Have to look at `load_pretrained` function in `checkpoints.py` thoroughly for conversion into torch.nn.Module.state_dict() format.
- [:heavy_check_mark:] General model architecture as a pl.LightningModule object with transparent code, with readable code for tokenisation, training steps etc.   
- [ ] Implementation of 4 variations of ViT (b16, b32, l16, l32) in PyTorch based on `configs.py` in the official repo.
> Have to remove the hardcoded variables and write them in terms of `self.hparams`
- [ ] Implementation of `training step` and `configure optimizers` in the LightningModule to truly support fine-tuning on custom dataset.
- [ ] Implementation of a reusable torchvision.Dataset class to output tokenised images (with positional encodings) for usage in ViT.
> Have to look at `prefetch`, `get_data` and `get_dataset_info` in `train.py` for this.
- [ ] Support for Multi-GPU training/fine-tuning using pl.LightningModule's features.
