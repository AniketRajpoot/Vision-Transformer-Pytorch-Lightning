# Vision Transformer in PyTorch Lightning
This is a third party implementation of the **Visual Transformer [paper](https://arxiv.org/pdf/2010.11929.pdf)** in 
**[PyTorch Lightning](https://www.pytorchlightning.ai)** with focus on transparency in training/fine-tuning the model.  
Heavily based on Google's [official implementation in Flax](https://github.com/google-research/vision_transformer)

* Features to be implemented:
- [ ] Architecture as PyTorch modules (including Sparse and Linear Transformers).  
- [ ] Support for loading checkpoints (i.e, pretrained weights) saved as .npz by Flax model into an identical PyTorch model in terms of architecture.
> Have to look at `load_pretrained` function in `checkpoints.py` thoroughly for conversion into torch.nn.Module.state_dict() format.
- [ ] General model architecture as a pl.LightningModule object with transparent code, with readable code tokenisation, training steps etc.   
- [ ] Implementation of 4 variations of ViT (b16, b32, l16, l32) in PyTorch based on `configs.py` in the official repo.  
- [ ] Implementation of a reusable torchvision.Dataset class to output tokenised images (with positional encodings) for usage in ViT.
> Have to look at `prefetch`, `get_data` and `get_dataset_info` in `train.py` for this.
- [ ] Support for Multi-GPU training/fine-tuning using pl.LightningModule's features.
