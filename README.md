
# ZEPHYX: Multimodal Edge-AI for Rural Healthcare Diagnostics

## 📌 Project Overview
ZEPHYX is a decentralized, offline-first AI framework designed for regions with limited medical infrastructure. It integrates Computer Vision and Acoustic Signal Processing to provide real-time screening for dermal and respiratory conditions.

## 🔬 Core Features
- **Visual Texture Analysis:** Uses Laplacian Variance to detect fungal/bacterial patterns.
- **Acoustic Biomarkers:** Implements MFCC-based analysis to differentiate cough types (Dry vs Wet).
- **Edge Optimization:** Compressed TensorFlow Lite (TFLite) models for high-speed inference on low-power devices.

## 🏔️ Impact Statement
Developed with the Himalayan landscape of Uttarakhand in mind, ZEPHYX ensures patient data privacy and zero cloud dependency, bridging the gap between urban specialists and rural patients.

### 🔬 Experimental Results & HUD Analysis

<p align="center">
  <img src="Screenshot 2026-03-21 171559.png" width="45%" alt="Fungal Pattern Detection">
  <img src="Screenshot 2026-03-21 171802.png" width="45%" alt="Healthy Skin Validation">
</p>

- **Detection Logic:** The framework identified a **Moderate Risk** fungal pattern with a **Texture Variance of 308.80**.
- **Edge Efficiency:** These results were generated using a **TFLite quantized model**, ensuring sub-100ms inference on a standard laptop CPU without GPU or Internet.
- **Multimodal Status:** The system concurrently monitors **Acoustic Biomarkers** while performing visual segmentation.
