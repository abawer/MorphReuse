# MorphReuse

**Shaped Weight Sharing with Per-Layer Scaling for Lighter Computational Networks**  

[DOI Placeholder – Zenodo Link Here](https://zenodo.org)


## Overview


This repository explores **MorphReuse**, an efficient parameter-sharing technique that replaces conventional network layers with a shared trainable core and per-layer scaling adapters. Key features:

- **Shared Core MLP**: Central trainable transformation module
- **Per-Layer Scaling**: Learnable scaling factors for layer-specific adaptation
- **Residual Connections**: Identity + scaled transformation for stable training
- **CPU-Optimized**: Designed for accessible experimentation on standard hardware

Benchmarked across vision and text tasks:
- **Image Classification**: MNIST, FashionMNIST, CIFAR10
- **Text Classification**: SST-2 sentiment analysis


## Method by Dataset

We evaluate three training strategies—**Baseline**, **MorphReuse**, and **LoRA**—on three datasets: MNIST, FashionMNIST, CIFAR-10, and SST-2. Each method modifies which layers are trained or reused.


### Image Classification (MNIST, FashionMNIST, CIFAR10)

| Method       | Architecture                          | Trainable Components               |
|--------------|---------------------------------------|------------------------------------|
| **Baseline** | Full MLP/ConvNet                      | All parameters                     |
| **MorphReuse**| Shared core + scaling adapters       | Core + adapters only (10-26% params)|
| **LoRA**     | Low-Rank Adapters                     | Adapter matrices only (1-2% params)|

**Key Implementation**:
```python
class MorphReuseCore(nn.Module):
    def __init__(self, dim=128, expand=2.5):
        super().__init__()
        hidden = int(dim * expand)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        self.scale_param = nn.Parameter(torch.tensor([-3.0]))  # α-residual control

    def forward(self, x):
      transformed = self.net(x)
      scale = torch.sigmoid(self.scale_param) * 2  # 0-2 range
      return x + scale * transformed
```

**Baseline**  
- Two-layer MLP classifier. All weights are trained end-to-end.

**MorphReuse**  
- Both MLP layers are replaced with a shared MLP core and per-layer scaling adapters.  
- Outer wrappers are frozen; only the core and adapters are trained.

**LoRA**  
- Injects LoRA adapters into each `nn.Linear`.  
- Base weights are frozen; only adapters are trainable.


### CIFAR-10 (RGB Images 3×32×32, Conv Backbone + Classifier MLP)

Same MLP as above on top of a Convolutional Backbone net for feature extraction.

**Baseline**  
- Convolutional feature extractor followed by a fully connected MLP classifier.  
- All layers are trained.

**MorphReuse**  
- The convolutional extractor backcbone is trained.
- The classifier is replaced by two shared MorphReuse layers with scaling adapters.  
- Only the shared core and extractor backbone is trained, adaptor weights are frozen.

**LoRA**  
- The convolutional extractor backcbone is trained.
- LoRA adapters are added to the classifier MLP.  
- Only LoRA layers are trained out of the MLP.



### SST-2 (Sentiment Classification, Pretrained TinyBERT)

| Method       | Architecture                          | Trainable Components               |
|--------------|---------------------------------------|------------------------------------|
| **Baseline** | Pretrained TinyBERT                   | Three top layers                   |
| **MorphReuse**| Pretrained TinyBERT + Surrogate Shared Weights for three top layers + Non-Linear activation       | Surrogate Weights|
| **LoRA**     | Pretrained TinyBERT+Low-Rank Adapters for three top layer  | LoRA residual weights|


**Baseline**  
- Uses `huawei-noah/TinyBERT_General_4L_312D`.  
- Unfreezes top 2 transformer encoder layers + classifier head.  
- All unfrozen parts are trainable.

**MorphReuse**  
- All encoder layers are frozen.  
- The top 2 layers and classifier are wrapped with `MorphReuseLinearWrapperShared`.  
- Shared core is used across `q`, `k`, `v`, intermediate, output, and classifier projections.  
- Only the core and adapters are trained.

**LoRA**  
- Applies LoRA to top 2 layers and classifier.  
- Base TinyBERT is frozen.  
- Only LoRA modules are trained.



## Comparative Results


### Colab Notebook Specs
```
OS: Linux-6.1.123+-x86_64-with-glibc2.35
CPU: AMD EPYC 7B12 (2×2.25 GHz)
RAM: 12977 MB
Python: 3.11.13
PyTorch: 2.3.1+cpu
Device: cpu
```

### 1. MNIST (MLP for 1×28×28)

![MNIST](https://github.com/user-attachments/assets/cad4d968-fb9c-4e25-97fd-1e5d548e65f1)


### 2. FashionMNIST (MLP for 1×28×28)

![FashionMNIST](https://github.com/user-attachments/assets/26d71ee7-845f-4dc3-8f87-71d4bf74799e)



### 3. CIFAR-10 (ConvNet + MorphReuse Classifier)

![CIFAR10](https://github.com/user-attachments/assets/b4c876d2-8cf4-4cb0-828b-1c5132330f10)


### 4. SST-2 (Text Dataset using Headless TinyBERT + MorphReuse)

*Accuracy and F1-score benchmarks coming soon.*



## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for full text.
