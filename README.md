# Multimodal Co-Speech Gesture Generation System

## Introduction

This repository hosts a collaborative project focused on 3D Co-Speech Gesture Generation. The system aims to synthesize natural, diverse, and semantically aligned gestures for virtual avatars based on different input modalities.

The project is developed by a team of three, building upon a shared baseline architecture. Each member has branched out to optimize specific components of the generation pipeline: generative algorithms, semantic representation, and multimodal input processing.

## Module Overview

This repository is organized into three distinct sub-modules. **For detailed instructions regarding environment setup, model training, and inference scripts, please navigate to the README.md file located within each specific subdirectory.**

### 1. Diffusion-based Gesture Generation

* **Directory:** `./Diffusion_Text2Gesture/` 
* **Maintainer:** [WEI LI]
* **Description:** This module replaces the traditional regression-based output layer with a Transformer-based Gaussian Diffusion Model. It addresses the "Mean Collapse" problem common in deterministic models, resulting in gestures with realistic dynamics and high diversity.
* **Key Features:** 12D feature representation, iterative denoising, and text-conditioned generation.
* **Documentation:** [Read Module Documentation](https://www.google.com/search?q=./Diffusion_Text2Gesture/README.md)

### 2. BERT-based Semantic Embedding

* **Directory:** `./BERT_Embedding_Module/` 
* **Maintainer:** [Mateus DE BRITO GUIRARDELLO]
* **Description:** This module integrates pre-trained BERT embeddings to replace standard word vectors. It enhances the model's ability to capture complex linguistic nuances, emotional tone, and semantic context from long text inputs.
* **Key Features:** Context-aware text encoding and semantic alignment optimization.
* **Documentation:** [Read Module Documentation](https://www.google.com/search?q=./BERT_Embedding_Module/README.md)

### 3. Voice-driven Motion Interface

* **Directory:** `./Voice_Input_Module/` (Please adjust folder name if necessary)
* **Maintainer:** [Yujia GUO]
* **Description:** This module expands the system's input modality to support direct audio signals. It extracts acoustic features (such as MFCC and prosody) to synchronize gesture rhythm and intensity with speech audio.
* **Key Features:** Audio feature extraction, cross-modal alignment, and speech-driven synthesis.
* **Documentation:** [Read Module Documentation](https://www.google.com/search?q=./Voice_Input_Module/README.md)




## Contributors

* **[WEI LI]:** Diffusion Model Implementation & Optimization.
* **[Mateus DE BRITO GUIRARDELLO]:** BERT Representation Learning & Text Encoder.
* **[Yujia GUO]:** Audio Feature Extraction & Voice Pipeline.
