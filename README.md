# 🧠 Retinal OCT Image Synthesis using Diffusion Models


---

## 🌟 Overview

This project presents a deep learning framework based on Denoising Diffusion Probabilistic Models (DDPMs) for retinal Optical Coherence Tomography (OCT) image synthesis and refinement. The model learns the structural distribution of retinal scans and reconstructs meaningful anatomical patterns through iterative reverse diffusion.

The objective of this work is to generate realistic retinal OCT images while preserving important retinal structures such as layered tissue boundaries, retinal pigment epithelium (RPE) regions, and disease-related pathological features. The framework is designed to demonstrate the effectiveness of diffusion models in medical image generation and enhancement tasks.

---

## 🔬 What is a Diffusion Model?

A diffusion model is a generative deep learning architecture that learns how to progressively reconstruct data from noise. During training, clean images are gradually corrupted using Gaussian noise through a forward diffusion process. The neural network then learns the reverse denoising trajectory required to recover the original image structure.

Unlike adversarial generative approaches, diffusion models learn the probability distribution of the data in a stable and iterative manner. This makes them highly effective for generating high-quality and structurally consistent medical images.

The generation process begins with random noise and progressively reconstructs a coherent OCT image over multiple denoising steps.


---

## ⚙️ Model Architecture

The framework uses a time-conditioned U-Net architecture as the core noise prediction network. Temporal embeddings are injected into residual convolutional blocks to guide the denoising process at different timesteps.

The encoder extracts hierarchical retinal features while the decoder reconstructs spatial details using skip connections. Group Normalization and SiLU activations are used throughout the architecture to improve stability and convergence during training.



---

## 🎯 Project Objectives

The primary objectives of this project are:

- Retinal OCT image synthesis using diffusion models
- Structural refinement of noisy retinal scans
- Preservation of retinal anatomical consistency
- Demonstration of reverse diffusion reconstruction
- Medical image generation for research applications
- Exploration of diffusion-based enhancement techniques

---

## 🩺 Why Retinal OCT?

Retinal OCT imaging is widely used in ophthalmology for detecting retinal disorders such as:

- Choroidal Neovascularization (CNV)
- Diabetic Macular Edema (DME)
- Drusen
- Normal retinal anatomy

OCT scans contain extremely fine retinal layer structures that are difficult to model using conventional generative approaches. Diffusion models provide improved training stability and preserve detailed structural information more effectively than many GAN-based approaches.

---

## 📊 Diffusion Process

The workflow followed in this project includes:

1. OCT image preprocessing and normalization
2. Forward diffusion through Gaussian noise injection
3. Noise prediction using a U-Net backbone
4. Reverse diffusion reconstruction
5. Image refinement and structural recovery
6. Quantitative and qualitative analysis

---

## ✨ Generated Outputs

The framework supports:

- Synthetic OCT image generation
- Retinal structure refinement
- Reverse diffusion visualization
- Error heatmap analysis
- Structural comparison grids



---

## 📈 Evaluation Metrics

The generated and refined outputs are evaluated using:

- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Fréchet Inception Distance (FID)
- Qualitative structural analysis

These metrics help evaluate anatomical preservation, image realism, and reconstruction quality.

---

## 📁 Repository Structure

```text
retinal-oct-diffusion/
│
├── assets/
├── data/
├── results/
├── refine_outputs/
├── train.py
├── sample.py
├── refine_2.py
├── analysis.py
├── requirements.txt
└── README.md
