# DSP + MLP Based Dereverberation  

This repository contains a deep learning + DSP project for **speech dereverberation**, the process of removing echo and reverberation from audio signals.  
The model is a custom-built **Multi-Layer Perceptron (MLP)** that operates on **framed audio segments** and reconstructs enhanced speech using **overlap-add** DSP techniques.  

The system is designed to improve **speech clarity and intelligibility** for real world reverberant recordings, and has been developed in the **Samsung Spatial Hackathon 2025**.  

---

## Key Features  

- **Custom MLP Architecture**  
  - Fully connected network for end-to-end speech frame enhancement.  
  - Input: 512-sample audio frames → Output: Enhanced frames.  

- **DSP Signal Framing + Reconstruction**  
  - Uses **framing** and **overlap-add** techniques for efficient time-domain reconstruction.  

- **Objective + Perceptual Evaluation**  
  - **MSE Loss** for frame-level training.  
  - **PESQ (Perceptual Evaluation of Speech Quality)** for perceptual quality measurement after every epoch.  

- **Efficient Data Pipeline**  
  - Custom PyTorch Dataset loads **parallel clean/reverberant audio pairs**.  
  - Handles resampling, mono conversion, and truncation to ensure alignment.  

---

## Technologies  

- **PyTorch** – Model training & optimization.  
- **Torchaudio** – Audio loading, saving, and resampling.  
- **PESQ** – Perceptual speech quality evaluation.  
- **Matplotlib** – Visualization of training loss curves.  

---

## Results  

Our DSP + MLP dereverberation system achieved an average **PESQ score of ~3.0**,which is considered a **good perceptual quality** for enhanced speech (scale: 1 = bad, 4.5 = excellent).  
This demonstrates that the model not only reduces reverberation but also improves speech clarity and intelligibility in realistic conditions.  

