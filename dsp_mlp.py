import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pesq
import matplotlib.pyplot as plt

# PATHS (replace before running)

CLEAN_DIR = "path_to_clean_audio"
REVERB_DIR = "path_to_reverb_audio"

# DSP DATASET
class FrameAudioDataset(Dataset):
    def __init__(self, clean_dir, reverb_dir, frame_size=512, hop_size=256, target_sr=16000):
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.reverb_files = sorted([os.path.join(reverb_dir, f) for f in os.listdir(reverb_dir) if f.endswith('.wav')])
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.target_sr = target_sr
        assert len(self.clean_files) == len(self.reverb_files), "Number of files mismatch!"

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean, sr_c = torchaudio.load(self.clean_files[idx])
        reverb, sr_r = torchaudio.load(self.reverb_files[idx])

        # Mono
        clean = torch.mean(clean, dim=0)
        reverb = torch.mean(reverb, dim=0)

        # Resample if needed
        if sr_c != self.target_sr:
            clean = torchaudio.functional.resample(clean, sr_c, self.target_sr)
        if sr_r != self.target_sr:
            reverb = torchaudio.functional.resample(reverb, sr_r, self.target_sr)

        # Truncate to same length
        min_len = min(clean.size(0), reverb.size(0))
        clean = clean[:min_len]
        reverb = reverb[:min_len]

        return reverb, clean

# DSP FUNCTIONS
def frame_signal(signal, frame_size=512, hop_size=256):
    """Split 1D signal into overlapping frames"""
    return signal.unfold(0, frame_size, hop_size)

def overlap_add(frames, hop_size=256):
    """Reconstruct signal from frames"""
    frame_size = frames.size(1)
    total_len = (frames.size(0)-1) * hop_size + frame_size
    signal = torch.zeros(total_len)
    for i, frame in enumerate(frames):
        start = i * hop_size
        signal[start:start+frame_size] += frame
    return signal

# MLP MODEL
class DSPMLP(nn.Module):
    def __init__(self, frame_size=512):
        super(DSPMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(frame_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, frame_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# TRAINING
def train_model(clean_dir, reverb_dir, num_epochs=20, batch_size=2, frame_size=512, hop_size=256, target_sr=16000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FrameAudioDataset(clean_dir, reverb_dir, frame_size=frame_size, hop_size=hop_size, target_sr=target_sr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DSPMLP(frame_size=frame_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    all_losses = []
    print(" Training started...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for reverb_sig, clean_sig in dataloader:
            batch_loss = 0
            for r, c in zip(reverb_sig, clean_sig):
                r_frames = frame_signal(r, frame_size, hop_size).to(device).float()
                c_frames = frame_signal(c, frame_size, hop_size).to(device).float()

                optimizer.zero_grad()
                output_frames = model(r_frames)
                loss = criterion(output_frames, c_frames)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            epoch_loss += batch_loss / len(reverb_sig)

        avg_loss = epoch_loss / len(dataloader)
        all_losses.append(avg_loss)

        # PESQ on first sample
        model.eval()
        with torch.no_grad():
            r0, c0 = dataset[0]
            r0_frames = frame_signal(r0, frame_size, hop_size).to(device).float()
            enhanced_frames = model(r0_frames).cpu()
            enhanced_sig = overlap_add(enhanced_frames)
            min_len = min(len(c0), len(enhanced_sig))
            pesq_score = pesq.pesq(target_sr, c0[:min_len].numpy(), enhanced_sig[:min_len].numpy(), 'wb')
        model.train()

        # Print results after every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] -> Loss: {avg_loss:.6f}, PESQ: {pesq_score:.3f}")

    # Save model
    torch.save(model.state_dict(), "dsp_mlp_model.pth")
    print("Model saved as dsp_mlp_model.pth")

    # Save one example output
    torchaudio.save("dereverbed_example.wav", enhanced_sig.unsqueeze(0), target_sr)
    print("Dereverberated audio saved as dereverbed_example.wav")

    # Plot loss curve
    plt.figure()
    plt.plot(all_losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.show()

# RUN
if __name__ == "__main__":
    train_model(CLEAN_DIR, REVERB_DIR, num_epochs=25, batch_size=2)
