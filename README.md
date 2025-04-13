# Momenta - Audio Deepfake Detection

##  Overview
This project is part of the take-home assessment for the Momenta Audio Deepfake Detection challenge. The goal is to research, implement, and analyze a model capable of distinguishing between real and AI-generated (deepfake) audio speech samples.

---

##  Research Summary
### Top 3 Audio Deepfake Detection Approaches Selected

#### 1. **RawNet2**
- **Key Innovation**: Uses raw audio waveforms directly as input without spectrogram conversion.
- **Performance**: High accuracy on ASVspoof2019 and other datasets.
- **Why Promising**: Lightweight, fewer preprocessing steps, suitable for near real-time detection.
- **Challenges**: Sensitive to noise and input length; may require padding or truncation.

#### 2. **AASIST (Audio Anti-Spoofing System)**
- **Key Innovation**: Attention-based feature learning over spectrogram inputs.
- **Performance**: SOTA results on ASVspoof2021.
- **Why Promising**: High generalization across multiple spoofing attacks.
- **Challenges**: Heavier and slower, difficult to use in real-time scenarios.

#### 3. **LCNN (Lightweight CNN)**
- **Key Innovation**: CNN on spectrograms with max-feature-map activation.
- **Performance**: Competitive accuracy with reduced model size.
- **Why Promising**: Lightweight and suitable for edge deployment.
- **Challenges**: Needs spectrogram preprocessing and tuning.

---

##  Chosen Model for Implementation: **RawNet2**

- **Reason for Selection**: Simpler input pipeline, good performance, faster to train and deploy.
- **Implementation**: Built in PyTorch; trained on a labeled dataset of real and fake audio samples.
- **Dataset Used**: Custom version of the [York University Audio Deepfake Dataset](https://www.eecs.yorku.ca/~bil/Datasets/for-original.tar.gz)

---

##  Results
- **Accuracy**: Achieved ~XX% test accuracy *(fill after running evaluate.py)*
- **Strengths**:
  - Efficient with raw waveform input
  - Relatively fast inference
- **Weaknesses**:
  - May require manual padding/truncation
  - Limited robustness to noisy environments

---

## 🔧 Usage Instructions

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Preprocess & Train
```bash
python train.py
```

### 3. Evaluate
```bash
python evaluate.py
```

### 4. Predict
```bash
python predict.py
```

---

##  Reflection Questions

1. **Challenges Faced**:
   - Working with raw waveform length mismatch
   - Data cleaning and label consistency

2. **Real-World Performance**:
   - May vary based on microphone quality, background noise, and speaker variability.

3. **Improvements with More Resources**:
   - Fine-tuning with a larger and more diverse dataset
   - Noise augmentation during training

4. **Production Deployment Strategy**:
   - Wrap the model in a FastAPI service
   - Batch process audio snippets in real-time buffers
   - Use caching and GPU acceleration for speed

---

## 📚 Dependencies
- torch
- torchaudio
- numpy
- scikit-learn
- tqdm



