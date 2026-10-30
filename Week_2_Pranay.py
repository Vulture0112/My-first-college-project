# Generated from: Week_2_CNN_to_ResNet50_Assignment.ipynb
# Converted at: 2026-08-29T12:56:19.944Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Week 2: Deep Learning Assignment — From Custom CNNs to Pretrained ResNet-50
# **Physics and Astronomy Club (PaAC), IIT Roorkee**  
# **Course:** Fine-Tuning Deep Learning Models on Astrophysical Datasets (TMI-102)
# 
# ---
# 
# ## 🎯 Learning Objectives
# 1. Understand the core mathematical building blocks of Convolutional Neural Networks: convolution kernels, receptive fields, stride, and padding.
# 2. Implement a custom multi-layer CNN in PyTorch using `torch.nn.Module`.
# 3. Understand the vanishing gradient problem and how Residual Skip Connections ($x + \mathcal{F}(x)$) solve it.
# 4. Load and inspect the architecture of **ResNet-50** from `torchvision.models`.
# 5. Extract and visualize the 64 learned convolutional filters from the stem `conv1` layer.
# 6. Register forward hooks to inspect intermediate feature maps of deep layers.
# 
# ---
# 
# ## 📦 Setup & Dependencies
# Run the cell below to install and import the required libraries.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# ## 🧠 Part 1: Theoretical & Architectural Questions
# 
# ### Question 1.1: The Output Dimension Formula
# Given an input image of size $W \times W$, a convolutional kernel of size $K \times K$, padding $P$, and stride $S$:
# 1. Write down the formula for the output spatial dimension $W_{out}$.
# 2. If an astrophysical image of size $224 \times 224$ passes through a `conv2d` layer with $K=7, S=2, P=3$, calculate the exact output height and width.
# 
# *Type your answer below:*
# ```text
#! Formula: W_out = [(W-K+2P)/S]+ 1
# Calculation for 224x224 (K=7, S=2, P=3):
#! W_out = [(224-7+6)/2]+1
#! = [223/2]+1
#! = [111.5]+1
#! = 111+1
#! = 112
#! Output will be of size 112x112
# ```


# ### Question 1.2: Why Skip Connections?
# In deep networks (e.g., 20+ layers), standard feedforward architectures suffer from degradation and vanishing/exploding gradients.
# 1. Explain how the residual formulation $H(x) = F(x) + x$ ensures that a 50-layer network performs at least as well as a shallower network.
# 2. What is the role of the $1 \times 1$ convolution in the ResNet Bottleneck block?
# 
# *Type your answer below:*
# ```text
# Answer:
#! Answer of Q1:
#!  The formulation $H(x) = F(x) + x$ ensures baseline performance by providing the network with a mathematical bypass. If the additional layers in a 50-layer model do not actually help improve predictions, the network naturally learns to push their weight values ($F(x)$) to zero. The equation then collapses to $H(x) = 0 + x$, meaning the data ($x$) is passed straight through completely unaltered. Because a deep network can effectively turn off its extra layers to perfectly mimic a shallower version of itself, its accuracy will never degrade below the performance of that shallower network.
#! 


#! Answer of Q2:
#!  The $1 \times 1$ convolution functions as a dimensional funnel to drastically reduce computational workload. Instead of scanning a grid for spatial shapes, it scans a single pixel across all channels to compress the data. In a Bottleneck block, the first $1 \times 1$ layer shrinks the channel count (e.g., from 256 down to 64) so that the heavily intensive $3 \times 3$ convolution can run on a fraction of the data. A final $1 \times 1$ layer then expands the channels back to their original size. This squeeze-and-expand method allows the model to remain deep and highly accurate without running out of memory.
# ```


# ## 💻 Part 2: Building a Custom 3-Layer Galaxy CNN from Scratch
# 
# Implement `GalaxyCNN` with the following architecture:
# - **Stem / Layer 1:** Conv2d (in_channels=3, out_channels=16, kernel_size=3, padding=1), BatchNorm2d(16), ReLU, MaxPool2d(kernel_size=2, stride=2)
# - **Layer 2:** Conv2d (in_channels=16, out_channels=32, kernel_size=3, padding=1), BatchNorm2d(32), ReLU, MaxPool2d(kernel_size=2, stride=2)
# - **Layer 3:** Conv2d (in_channels=32, out_channels=64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU, AdaptiveAvgPool2d((1, 1))
# - **Classifier:** Flatten -> Linear(64, 2) (Spiral vs. Elliptical classification)


class GalaxyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(GalaxyCNN, self).__init__()
        # TODO: Implement the layers described above
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Test your model with a batch of 4 sample galaxy images (3 channels, 224x224)
dummy_input = torch.randn(4, 3, 224, 224)
custom_model = GalaxyCNN(num_classes=2)
output = custom_model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output logits shape: {output.shape}") # Should be [4, 2]
total_custom_params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)
print(f"Total trainable parameters in GalaxyCNN: {total_custom_params:,}")

# ## 🔬 Part 3: Inspecting Pretrained ResNet-50 Architecture
# 
# Now load the pretrained ResNet-50 model from `torchvision.models` and inspect its parameters and layers.


# TODO: Load weights using torchvision.models.ResNet50_Weights.DEFAULT
weights = models.ResNet50_Weights.DEFAULT
resnet50 = models.resnet50(weights=weights)
resnet50.eval()

total_resnet_params = sum(p.numel() for p in resnet50.parameters())
print(f"Total Parameters in ResNet-50: {total_resnet_params:,}")

# Compare parameter count of Custom CNN vs ResNet-50
print(f"ResNet-50 is {total_resnet_params / total_custom_params:.1f}x larger than our Custom CNN!")

# ## 📊 Part 4: Step-by-Step Tensor Dimension Tracing through ResNet-50
# 
# Trace a sample galaxy tensor `[1, 3, 224, 224]` through each major block of ResNet-50:
# 1. `conv1` -> `bn1` -> `relu` -> `maxpool`
# 2. `layer1`
# 3. `layer2`
# 4. `layer3`
# 5. `layer4`
# 6. `avgpool` -> `fc`


sample_galaxy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    # Stem
    x = resnet50.conv1(sample_galaxy)
    print(f"After conv1 (Stem): {x.shape}")
    x = resnet50.bn1(x)
    x = resnet50.relu(x)
    x = resnet50.maxpool(x)
    print(f"After stem maxpool: {x.shape}")
    
    # Residual Stages
    x1 = resnet50.layer1(x)
    print(f"After Layer 1: {x1.shape}")
    
    x2 = resnet50.layer2(x1)
    print(f"After Layer 2: {x2.shape}")
    
    x3 = resnet50.layer3(x2)
    print(f"After Layer 3: {x3.shape}")
    
    x4 = resnet50.layer4(x3)
    print(f"After Layer 4: {x4.shape}")
    
    # Head
    x_pool = resnet50.avgpool(x4)
    print(f"After avgpool: {x_pool.shape}")

    #avgpool to fc
    x_flattened = torch.flatten(x_pool,1)
    x_fc = resnet50.fc(x_flattened)
    print(f"After fc: {x_fc.shape}")

# ## 🎨 Part 5: Visualizing the 64 Stem Filters of ResNet-50
# 
# The first convolutional layer `conv1` has weights of shape `[64, 3, 7, 7]`.  
# These represent 64 distinct $7 \times 7$ RGB feature detectors (edges, color gradients, textures).
# 
# Plot all 64 filters in an $8 \times 8$ grid using Matplotlib.


# Extract conv1 weights
conv1_w = resnet50.conv1.weight.data.cpu().numpy()
print(f"conv1 weight shape: {conv1_w.shape}") # [64, 3, 7, 7]

# Normalize each filter to [0, 1] for visualization
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    # Filter shape: (3, 7, 7) -> transpose to (7, 7, 3) for RGB display
    filt = conv1_w[i].transpose(1, 2, 0)
    filt_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
    ax.imshow(filt_norm)
    ax.axis('off')

plt.suptitle("ResNet-50 Stem (conv1) 64 Learned Filters [7x7 RGB]", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ## 🚀 Submission Instructions
# 1. Ensure all code cells run cleanly and outputs/plots are generated.
# 2. Answer the theoretical questions in Part 1.
# 3. Export and submit your completed notebook (`.ipynb`) on the Google Form before **30th August, 11:59 PM**.