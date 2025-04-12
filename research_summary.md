# Audio Deepfake Detection — Research Summary

This document summarizes three promising approaches for detecting AI-generated speech, focusing on applicability to real-world conversations and potential for near real-time detection.

## 1. RawNet2

- Model Type: End-to-End CNN + GRU on raw waveform input  
- Key Technical Innovation:
  - Operates directly on raw audio signals (no handcrafted features like MFCC)
  - Combines CNN layers for feature extraction with a GRU layer for temporal dependencies
  - Uses center loss along with softmax loss to improve discriminative power
- Reported Performance:
  - Equal Error Rate (EER): 1.5% on ASVspoof 2019 LA dataset
- Why It’s Promising:
  - Lightweight architecture, fast inference
  - No reliance on traditional spectral features; robust to varied environments
  - Suitable for real-time analysis
- Potential Limitations:
  - Performance may degrade with high background noise
  - Lacks interpretability compared to feature-based models
- Source: [Paper](https://arxiv.org/abs/2006.12179), [GitHub](https://github.com/asvspoof-challenge/RawNet2)

## 2. AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Modeling)

- Model Type: CNN + Transformer-based Attention Network  
- Key Technical Innovation:
  - Leverages both spectral and temporal cues using self-attention
  - Introduces spectro-temporal graph attention blocks (STGABs)
  - Integrates CNN backbones and Transformer layers for enhanced generalization
- Reported Performance:
  - EER: 0.42% on ASVspoof 2021 Deepfake (DF) Task
- Why It’s Promising:
  - State-of-the-art accuracy on most audio spoofing tasks
  - Better at learning from limited data
  - Generalizes well to unknown spoofing attacks
- Potential Limitations:
  - Higher computational cost, less ideal for edge/real-time environments
  - Larger model size requires optimized inference for production
- Source: [Paper](https://arxiv.org/abs/2110.01200), [GitHub](https://github.com/clovaai/aasist)

## 3. LFCC + GMM (Baseline Traditional Method)

- Model Type: Linear Frequency Cepstral Coefficients + Gaussian Mixture Models  
- Key Technical Innovation:
  - Uses LFCC features instead of MFCC (better suited for spoof detection)
  - Applies traditional GMMs to classify bonafide vs. spoofed audio
- Reported Performance:
  - EER: 3.4% on ASVspoof 2019 (Baseline)
- Why It’s Promising:
  - Extremely fast and lightweight
  - Minimal training required
  - Useful for embedded systems or mobile deployment
- Potential Limitations:
  - Lower performance compared to deep learning models
  - Poor generalization to unseen spoofing techniques
  - Relies on handcrafted features, limiting robustness
- Source: [ASVspoof 2019 Baseline](https://www.asvspoof.org/index.php)

## Summary Table

| Model        | Type                 | EER (%) | Real-Time Viability | Accuracy  | Interpretability |
|--------------|----------------------|---------|----------------------|-----------|------------------|
| RawNet2      | CNN + GRU on raw     | 1.5     | High                 | High      | Moderate         |
| AASIST       | CNN + Transformer    | 0.42    | Medium               | Very High | Moderate         |
| LFCC + GMM   | Traditional(LFCC+GMM)| 3.4    | Very High           | Low      | High             |

## **Final Choice for Implementation**

For the implementation phase, RawNet2 is selected due to its balance of:
- Strong performance
- Fast inference and real-time compatibility
- Simpler architecture for prototyping and further extension
- Direct operation on raw waveforms, reducing preprocessing overhead

The combination of end-to-end training, reduced dependency on handcrafted features, and real-world viability makes RawNet2 an ideal candidate for this project.
