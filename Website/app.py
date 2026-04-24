import sys
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import mpld3
import librosa

# get file path from FastAPI
file_path = sys.argv[1]

audio, sample_rate = librosa.load(file_path, sr=None)
waveform = torch.tensor(audio).unsqueeze(0)

mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

mel_spec = mel_transform(waveform)
mel_spec_db = T.AmplitudeToDB()(mel_spec)

fig = plt.figure(figsize=(10, 4))
plt.imshow(mel_spec_db[0].numpy(), origin="lower", aspect="auto")
plt.colorbar(label="dB")
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.tight_layout()

# convert plot to HTML instead of opening a window
html = mpld3.fig_to_html(fig)

# IMPORTANT: print result so FastAPI can capture it
print(html)