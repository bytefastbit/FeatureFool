import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from moviepy import VideoFileClip, AudioFileClip
from scipy.ndimage import gaussian_filter

# ========================= CONFIG =========================
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_featurefool.mp4"

EPS = 0.25          # ← EXTREEM: 0.25 (was 0.15). Probeer later 0.30 als het nog niet genoeg is
NUM_STEPS = 20      # ← Iterative PGD: 20 stappen = veel krachtiger
STEP_SIZE = 0.02    # ← Goede balans voor I3D (niet te groot, niet te klein)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================

# Load I3D model
from pytorch_i3d.pytorch_i3d import InceptionI3d

model = InceptionI3d(400, in_channels=3)

state_dict = torch.load(
    r"pytorch_i3d\models\rgb_imagenet.pt",
    map_location=DEVICE,
    weights_only=True  # ✅ FIX warning
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# Guided Backprop hook
def guided_relu_hook(module, grad_in, grad_out):
    return (torch.clamp(grad_in[0], min=0.0),)

for module in model.modules():
    if isinstance(module, torch.nn.ReLU):
        module.register_backward_hook(guided_relu_hook)

# ========================= STEP 1 =========================
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

frames = []
flow_magnitudes = []

ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("❌ Could not read video")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frames.append(prev_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
    flow_magnitudes.append(mag)

    frames.append(frame)
    prev_gray = gray

cap.release()

max_flow_idx = int(np.argmax(flow_magnitudes))
print(f"Selected max-motion frame index: {max_flow_idx}")

# ========================= STEP 2 =========================
# Betere clip: echte 16 frames rond max-motion
clip_length = 16
num_frames = len(frames)
center_idx = max_flow_idx + 1
start_idx = max(0, center_idx - clip_length // 2)
end_idx = min(num_frames, start_idx + clip_length)

clip_frames = frames[start_idx:end_idx]
while len(clip_frames) < clip_length:
    clip_frames.append(clip_frames[-1])

frame_tensors = []
for f in clip_frames:
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    frame_tensors.append(tensor)

clip = torch.stack(frame_tensors, dim=0)
clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE).detach().requires_grad_(True)

# Eerste forward → target class
model.zero_grad()
output = model(clip)
while output.dim() > 2:
    output = output.mean(dim=-1)
if output.dim() == 2:
    output = output[0]

target = output.argmax().item()
print(f"Attacking target class: {target} (initial logit = {output[target].item():.4f})")
print("→ Dit is class 'extinguishing fire' → perfect voor explosies!")

# ========================= PGD ATTACK (20 stappen) – GEFIXT =========================
clip_adv = clip.clone().detach().requires_grad_(True)   # verse leaf tensor
original_clip = clip.clone().detach()

for step in range(NUM_STEPS):
    model.zero_grad()
    output = model(clip_adv)
    while output.dim() > 2:
        output = output.mean(dim=-1)
    if output.dim() == 2:
        output = output[0]
    
    loss = output[target]          # minimaliseren = confidence omlaag
    loss.backward()
    
    # === DE FIX: no_grad + opnieuw leaf maken ===
    with torch.no_grad():
        grad_sign = torch.sign(clip_adv.grad)
        clip_adv = clip_adv - STEP_SIZE * grad_sign
        
        # Project binnen EPS-ball
        delta = torch.clamp(clip_adv - original_clip, -EPS, EPS)
        clip_adv = torch.clamp(original_clip + delta, 0.0, 1.0)
        
        # Belangrijk: maak opnieuw een leaf tensor voor de volgende stap
        clip_adv = clip_adv.detach().requires_grad_(True)
    
    print(f"PGD step {step+1}/{NUM_STEPS} - target logit = {output[target].item():.4f}")

# Finale forward na alle stappen (echte eindwaarde)
model.zero_grad()
output = model(clip_adv)
while output.dim() > 2:
    output = output.mean(dim=-1)
if output.dim() == 2:
    output = output[0]

print(f"✅ PGD finished! Final target logit = {output[target].item():.4f} (hoopvol sterk gedaald!)")

# ========================= FEATURE MAP (nu echte delta) =========================
delta = (clip_adv - original_clip)[0].mean(dim=(0, 1)).detach().cpu().numpy()  # [H, W]
delta = gaussian_filter(delta, sigma=1.0)
h, w = frames[0].shape[:2]
delta = cv2.resize(delta, (w, h))
delta = np.expand_dims(delta, axis=-1)

# ========================= STEP 3 =========================
perturbed_frames = []
for frame in frames:
    frame_float = frame.astype(np.float32) / 255.0
    adv_frame = np.clip(frame_float + delta, 0, 1)   # + delta (PGD heeft al de juiste richting)
    adv_frame = (adv_frame * 255).astype(np.uint8)
    perturbed_frames.append(adv_frame)

# ========================= STEP 4 =========================
height, width = perturbed_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter("temp_video.mp4", fourcc, fps, (width, height))

for f in perturbed_frames:
    out.write(f)

out.release()

# ========================= STEP 5 =========================
video_clip = VideoFileClip("temp_video.mp4")
audio_clip = AudioFileClip(INPUT_VIDEO)

final_clip = video_clip.with_audio(audio_clip)
final_clip.write_videofile(
    OUTPUT_VIDEO,
    codec="libx264",
    audio_codec="aac",
    temp_audiofile="temp_audio.m4a",
    remove_temp=True,
    logger=None
)

final_clip.close()
audio_clip.close()
video_clip.close()

os.remove("temp_video.mp4")

print("✅ Done! Output saved as", OUTPUT_VIDEO)