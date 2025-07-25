# MorphReuse
MorphReuse: Shape-Level Weight Sharing with Per-Layer Scaling for Lightweight tuning

1) Image classification nets

Reusing trainable MLP core as a surrogate to a fully connected top layers

1.a) Flat Image Case

1.b) Convolutional Image Case

2) Linear LLM

Reusing shared weights along top layers of pretrained text LLM for a new task

# Comparative Results

Baseline 
a. Full training (Image Nets)
b. Training top 3 layers of pretrained LLM for sentiment classification task

MorphReuse
a. Training Image Nets
b. Training top 3 layers of pretrained LLM for sentiment classification task

LoRA
a. Training Image Nets
b. Training top 3 layers of pretrained LLM for sentiment classification task


Colab Notebook Spec:

```
OS: Linux-6.1.123+-x86_64-with-glibc2.35
CPU: AMD EPYC 7B12 (2×2.25 GHz)
RAM: 12977 MB
Python: 3.11.13
PyTorch: 2.3.1+cpu
Device: cpu
```

1) MNIST (MLP for Flat Grayscale Image 1x28x28)
   
<img width="1990" height="1572" alt="image" src="https://github.com/user-attachments/assets/cad4d968-fb9c-4e25-97fd-1e5d548e65f1" />

   
3) FashionMNIST (MLP for Flat Grayscale Image 1x28x28)

  <img width="1990" height="1572" alt="image" src="https://github.com/user-attachments/assets/26d71ee7-845f-4dc3-8f87-71d4bf74799e" />

  
4) CIFAR10 (Convlutional Backbone for 3x32x32 Color Depth Images with classification MLP)

<img width="1990" height="1572" alt="image" src="https://github.com/user-attachments/assets/b4c876d2-8cf4-4cb0-828b-1c5132330f10" />

5) SST2 (Text dataset for Headless Pretrained LLM)

