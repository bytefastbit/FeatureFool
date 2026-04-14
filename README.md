# FeatureFool

**Adversarial video perturbation tool** designed to evade automated content moderation systems (e.g. TikTok, YouTube Shorts, Instagram Reels).

Using optical flow to detect high-motion frames and a **Projected Gradient Descent (PGD)** attack on the **I3D (Inception-v1)** model trained on Kinetics-400, FeatureFool generates a visually similar video that significantly reduces the confidence of violence/explosion/fireworks/shooting-related classes.

> ⚠️ This project is for **research and educational purposes only**. Use responsibly and in accordance with platform terms of service.

---

## Features

- Automatic selection of the most motion-rich frame using Farneback optical flow
- Real 16-frame temporal clip for accurate I3D gradient computation
- Strong **iterative PGD attack** (20+ steps) with L∞ norm constraint
- Guided ReLU backpropagation for cleaner saliency maps
- Preserves original audio and frame rate
- Works on any video containing explosions, blood, firearms, fights, etc.
- Easy to configure (EPS, steps, step size)

---

## How It Works

1. **Motion Analysis** – Finds the frame with maximum optical flow (most likely to trigger moderation).
2. **Target Class Detection** – Runs I3D on a 16-frame clip and identifies the top predicted class (e.g. class 121 = “extinguishing fire”).
3. **Adversarial Attack** – Performs PGD to minimize the logit of the target class.
4. **Video Reconstruction** – Applies the computed delta uniformly across all frames + re-adds original audio.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/bytefastbit/FeatureFool.git
cd FeatureFool

# 2. Create and activate environment (recommended: Anaconda)
conda create -n featurefool python=3.10
conda activate featurefool

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy scipy moviepy
