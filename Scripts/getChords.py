import librosa
import numpy as np
import torch
import math
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

WINDOW_SEC = 1     # Default analysis window size in seconds
HOP_SEC = 1    # Default hop size in seconds (how often to make a prediction)

#Random seeds for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
AUDIO_FILE = "p.mp3"        #Change this to your audio file path

DEVICE = "cpu"

# Dataset sizes
TRAIN_SAMPLES_PER_KEY = 2000
VAL_SAMPLES_PER_KEY = 400

# Sequence config
SEQ_LEN = 64          # timesteps per example
CHUNK_MIN = 2         # min timesteps per event (chord or note)
CHUNK_MAX = 6         # max timesteps per event

# Model config
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
KEY_TO_INDEX = {f"{p}_major": i for i, p in enumerate(PITCH_CLASSES)}
KEY_TO_INDEX.update({f"{p}_minor": i+12 for i, p in enumerate(PITCH_CLASSES)})

def key_index_to_name(idx: int) -> str:
    if idx < 12:
        return f"{PITCH_CLASSES[idx]}_major"
    return f"{PITCH_CLASSES[idx-12]}_minor"


class KeyDetectionLSTM(nn.Module):
    """
    LSTM-based model for key detection from chroma features.
    Input: (B, T, 12) chroma sequences
    Output: (B, 24) logits for 24 keys (12 major + 12 minor)
    """
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=24, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, 12)
        out, _ = self.lstm(x)
        last = out[:, -1, :]   # (B, H)
        return self.fc(last)

# Load the trained model from disk
def load_trained_model():
    model = KeyDetectionLSTM().to(DEVICE)
    model.load_state_dict(torch.load("key_detector_synth.pth", map_location=DEVICE))
    model.eval()
    return model


# Convert raw audio to binary chroma features (12-dimensional, binarized)
def audio_to_binary_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512
    )  # (12, T)

    chroma = chroma.T  # (T, 12)

    # Normalize each frame
    chroma = chroma / (chroma.max(axis=1, keepdims=True) + 1e-6)

    # Binarize to match synthetic training data
    chroma = (chroma > 0.5).astype(np.float32)

    return chroma

# Predict key at regular intervals throughout the audio file
def predict_key_intervals(model, filepath, window_sec=10, hop_sec=5):
    y, sr = librosa.load(filepath, sr=22050)

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    predictions = []

    for start in range(0, len(y) - window_samples, hop_samples):
        segment = y[start:start + window_samples]

        # --- Convert segment to chroma ---
        chroma = librosa.feature.chroma_stft(
            y=segment,
            sr=sr,
            n_fft=2048,
            hop_length=512
        ).T  # (time, 12)

        # Normalize
        chroma = chroma / (chroma.max(axis=1, keepdims=True) + 1e-6)

        # Binarize to match training
        chroma = (chroma > 0.5).astype(np.float32)

        # Skip if too short
        if len(chroma) < SEQ_LEN:
            continue

        # Trim or pad to SEQ_LEN
        chroma = chroma[:SEQ_LEN]

        X = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(X)
            pred = logits.argmax(dim=1).item()

        time_sec = start / sr
        predictions.append((time_sec, pred))

    return predictions

note_timestamp = []
def print_results(results):
    for t, key_idx in results:
        note_timestamp.append(f"{t:6.1f}s: {key_index_to_name(key_idx)}") #Save the results into a list
        print(f"{t:6.1f}s → {key_index_to_name(key_idx)}")

def main():
    model = load_trained_model()

    results = predict_key_intervals(
        model,
        AUDIO_FILE,
        window_sec=1.5,
        hop_sec=.5
    )

    print_results(results)


main()


# Audio file processing and chord generation using Gemini API
from google import genai
from google.genai import types
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
audio_file = client.files.upload(file=AUDIO_FILE) #Upload the audio file to Gemini and get a reference ID for it

prompt = """
Rearrange the provided audio into a very beginner-friendly solo acoustic guitar version.

You are also given a timestamp-to-chord reference list.

This reference list is a secondary guidance signal:
- It provides useful suggestions for chord timing and progression.
- It should be considered carefully, but not treated as ground truth.
- The audio remains the primary source of truth.

Relative importance:
- Audio: high confidence
- Reference list: medium confidence
- General music knowledge: fallback only

How to use it:
- Use it to guide your expectations of when chord changes occur.
- Use it to confirm or support what you hear in the audio.
- If the audio is unclear, lean on the reference list more.
- If the audio clearly disagrees, prioritize the audio.

Balance rules:
- Do not ignore the reference list.
- Do not copy it mechanically.
- Your output should loosely align with its structure and timing, but be corrected by the audio where needed.

Instructions:
- Keep the song recognizable and preserve the general harmonic movement.
- Simplify for beginner acoustic guitar using easy playable chords whenever possible.
- Output only the chord changes that actually need to be played.
- Do not list repeated timestamps unless the chord changes.
- Do not add filler timestamps.
- Organize the output by song sections such as Intro, Verse, Pre-Chorus, Chorus, Bridge, and Outro whenever identifiable.
- Use timestamps in m:ss format.
- For each entry, write the beginner-friendly chord first, followed by the approximate original/heard chord in parentheses.
- If the original chord is unclear, make the best musical estimate from the audio and reference list.
- Keep the output concise and performance-ready.

Accuracy and musicality constraints:

- Each chord must last at least 2–4 seconds unless clearly required by the audio.
- Do not create rapid or 1-second chord changes.
- Never assign multiple chords to the same timestamp.
- Only include a new timestamp when the chord actually changes.

- Use a small, consistent set of chords (ideally 3–6 total).
- Keep the progression musically coherent and in a consistent key.
- Avoid random or unrelated chords.

- Prefer common beginner chord progressions (e.g., G–D–Em–C).
- Avoid unnecessary complexity or over-detection.

- Align chord changes with musical phrases, not arbitrary timestamps.
- Treat sections independently

Consistency and progression rules:

Before generating the final output, you must:

1. Identify the most likely key of the song.
2. Determine a small set of core chords (3–6 chords) that best represent the song.
3. Infer a repeating chord progression pattern (e.g., I–V–vi–IV).

Then:
- Use this progression as the foundation for all sections.
- Keep chord choices consistent across the entire song.
- Avoid changing chords unless clearly required.

- Do NOT re-evaluate chords independently at each timestamp.
- Base all chord decisions on the established progression.

- If a section repeats (e.g., Verse 1 and Verse 2), reuse the same chord progression unless the audio clearly changes.
- Normalize chord timing to a steady rhythm (roughly every 2–4 seconds or per musical phrase).

Output exactly this structure:

Chords You'll Need
- [list only the beginner-friendly guitar chords used]

Song Structure with Chords and Timestamps

**[Section Name]**
* 0:03: G (Ab Major)
* 0:06: D (Eb Major)

Only output these two sections and nothing else.
"""


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[note_timestamp, audio_file, prompt],
    config=types.GenerateContentConfig(temperature=0.05, top_p=0.8, top_k=20),
)

print(response.text)
chords_to_play = response.text

prompt2 = """
You are given:
1. An audio file of a song
2. A structured chord timeline with timestamps (generated previously)

Your task is to convert this into a clean, beginner-friendly guitar chord sheet aligned with lyrics, similar to standard chord/lyric sheets.

Instructions:
- Use the audio as the primary reference for timing and phrasing.
- Use the provided chord timeline as guidance, not strict truth.
- Align chords above the exact words where the chord changes occur.
- Do not include timestamps in the output.
- Only show chords when they change.
- Keep spacing clean and readable.
- Use simple chord names (G, Em, C, D, etc.).
- Keep the structure organized into sections (Intro, Verse, Pre-Chorus, Chorus, etc.).
- Do not overfill chords — only place them where musically necessary.

Formatting rules:
- Section headers must be in square brackets, e.g. [Verse 1]
- Chords must appear directly above the corresponding lyrics
- Do NOT repeat chords unnecessarily
- Keep everything minimal and clean

Example format:

[Intro]
G

[Verse 1]
        G              Em
I found a love for me
        C                D
Darling, just dive right in and follow my lead

        G              Em
Well, I found a girl beautiful and sweet
        C                   D
I never knew you were the someone waiting for me

[Pre-Chorus]
        G
Cause we were just kids when we...
"""
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[chords_to_play, audio_file, prompt2],
    config=types.GenerateContentConfig(temperature=0.05, top_p=0.8, top_k=20),
)

print(response.text) #Final Response with chords aligned to lyrics in a clean format