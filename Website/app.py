import os
import sys
import numpy as np
import torch
import torch.nn as nn
import librosa
from dotenv import load_dotenv

from google import genai
from google.genai import types

# =========================
# CONFIG
# =========================
DEVICE = "cpu"
PTH_FILE = "key_detector_synth.pth"
SEQ_LEN = 64

PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

load_dotenv(".env.local") 

def key_index_to_name(idx: int) -> str:
    if idx < 12:
        return f"{PITCH_CLASSES[idx]}_major"
    return f"{PITCH_CLASSES[idx - 12]}_minor"

# =========================
# MODEL
# Must match the trained .pth architecture
# =========================
class KeyDetectionLSTM(nn.Module):
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
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def load_trained_model():
    model = KeyDetectionLSTM().to(DEVICE)
    model.load_state_dict(torch.load(PTH_FILE, map_location=DEVICE))
    model.eval()
    return model

# =========================
# AUDIO -> BINARY CHROMA
# Matches your uploaded file's inference style
# =========================
def segment_to_binary_chroma(segment, sr):
    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=sr,
        n_fft=2048,
        hop_length=512
    ).T  # (time, 12)

    # normalize each frame
    chroma = chroma / (chroma.max(axis=1, keepdims=True) + 1e-6)

    # binarize to resemble synthetic training data
    chroma = (chroma > 0.5).astype(np.float32)
    return chroma

def predict_key_intervals(model, filepath, window_sec=1.5, hop_sec=0.5):
    y, sr = librosa.load(filepath, sr=22050, mono=True)

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    predictions = []

    for start in range(0, max(1, len(y) - window_samples + 1), hop_samples):
        segment = y[start:start + window_samples]
        if len(segment) < window_samples:
            break

        chroma = segment_to_binary_chroma(segment, sr)

        if len(chroma) == 0:
            continue

        # pad or trim to SEQ_LEN
        if len(chroma) < SEQ_LEN:
            pad_len = SEQ_LEN - len(chroma)
            pad = np.zeros((pad_len, 12), dtype=np.float32)
            chroma = np.concatenate([chroma, pad], axis=0)
        else:
            chroma = chroma[:SEQ_LEN]

        x = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1).item()

        time_sec = start / sr
        predictions.append((time_sec, pred))

    return predictions

def build_note_timestamps(results):
    note_timestamp = []
    for t, key_idx in results:
        note_timestamp.append(f"{t:6.1f}s: {key_index_to_name(key_idx)}")
    return "\n".join(note_timestamp)


# Audio file processing and chord generation using Gemini API
# =========================
# GEMINI
# =========================
PROMPT_1 = """
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

PROMPT_2 = """
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
"""

def run_gemini(audio_path, note_timestamp_text):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    audio_file = client.files.upload(file=audio_path)

    response1 = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[note_timestamp_text, audio_file, PROMPT_1],
        config=types.GenerateContentConfig(
            temperature=0.05,
            top_p=0.8,
            top_k=20,
        ),
    )
    chords_to_play = response1.text

    response2 = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[chords_to_play, audio_file, PROMPT_2],
        config=types.GenerateContentConfig(
            temperature=0.05,
            top_p=0.8,
            top_k=20,
        ),
    )

    return chords_to_play, response2.text

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <audio_file.mp3>")
        sys.exit(1)

    audio_file = sys.argv[1]

    model = load_trained_model()
    results = predict_key_intervals(
        model,
        audio_file,
        window_sec=1.5,
        hop_sec=0.5
    )

    note_timestamp_text = build_note_timestamps(results)

    final_sheet = run_gemini(audio_file, note_timestamp_text)

    print(final_sheet[1])

if __name__ == "__main__":
    main()