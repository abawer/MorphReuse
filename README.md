# MorphReuse

**Shaped Weight Sharing with Per-Layer Scaling for Lighter Computational Networks**  

[DOI Placeholder – Zenodo Link Here](https://zenodo.org)


## Overview

This repository explores **MorphReuse**, a method for reusing a **trainable core MLP** with **per-layer scaling** adapters, replacing conventional multi-layer networks with a shared, shaped transformation. The approach is benchmarked across:

1. **Image Classification Networks**  
   - Replacing MLP or classifier top layers with a shared MorphReuse core.  
   - Includes both flat grayscale and convolutional color image inputs.

2. **LLM-based Text Classification**  
   - Reusing shared weights across top layers of a frozen pretrained Transformer.  
   - Applies to headless TinyBERT adapted for sentiment classification.



## Method by Dataset

We evaluate three training strategies—**Baseline**, **MorphReuse**, and **LoRA**—on three datasets: MNIST, CIFAR-10, and SST-2. Each method modifies which layers are trained or reused.

### MNIST and FashionMNIST (Flat Grayscale 1×28×28 Images)

**Baseline**  
- Two-layer MLP classifier. All weights are trained end-to-end.

**MorphReuse**  
- Both MLP layers are replaced with a shared MLP core and per-layer scaling adapters.  
- Outer wrappers are frozen; only the core and adapters are trained.

**LoRA**  
- Injects LoRA adapters into each `nn.Linear`.  
- Base weights are frozen; only adapters are trainable.


### CIFAR-10 (RGB Images 3×32×32, Conv Backbone + Classifier MLP)

**Baseline**  
- Convolutional feature extractor followed by a fully connected MLP classifier.  
- All layers are trainable.

**MorphReuse**  
- The convolutional extractor is frozen.  
- The classifier is replaced by two shared MorphReuse layers with scaling adapters.  
- Only the shared core and adapters are trained.

**LoRA**  
- Same conv backbone as baseline but frozen.  
- LoRA adapters are added to the classifier MLP.  
- Only LoRA layers are trained.



### SST-2 (Sentiment Classification, Pretrained TinyBERT)

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

| Method      | Image Classification      | Text Classification (SST-2)          |
|-------------|---------------------------|-------------------------------------|
| Baseline    | Full MLP training         | Fine-tune top 3 LLM layers          |
| MorphReuse  | Shared MLP core + frozen adapters| Surrogate shared weight for LLM heads |
| LoRA        | LoRA adapters on MLP      | LoRA adapters on LLM heads           |



## Colab Notebook Specs
```
OS: Linux-6.1.123+-x86_64-with-glibc2.35
CPU: AMD EPYC 7B12 (2×2.25 GHz)
RAM: 12977 MB
Python: 3.11.13
PyTorch: 2.3.1+cpu
Device: cpu
```

## Sample Benchmarks

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
