# Voice-Text-Hindi

A multi-mode speech-to-text pipeline for **Hindi**, built on
[Jivi AI's AudioX-North](https://huggingface.co/jiviai/audioX-north-v1) ASR model
(with an automatic fallback to a Whisper-based loader). It turns raw audio into
clean, normalized Hindi text, with speaker diarization and LLM-assisted dialogue
realignment for longer, multi-speaker recordings.

## What it does

The pipeline supports four modes, selected with `--mode`:

| Mode | Use case | Stack |
|---|---|---|
| `nemo` / `enterprise` | High-accuracy Hindi transcription | Jivi AI AudioX-North |
| `fir` | Long, multi-speaker statements | AudioX-North + pyannote diarization + local LLM dialogue realignment |
| `whisperx` | Subtitles with word-level timestamps | WhisperX |

### Pipeline stages

1. **Audio preprocessing** — resample to 16kHz mono PCM, loudness normalization,
   a voice-band bandpass filter, and duration-aware chunking so long recordings
   are split without cutting words.
2. **ASR** — Jivi AI AudioX-North for Hindi transcription (with a Whisper
   processor/model fallback if the pipeline API isn't available), or WhisperX
   for subtitle mode.
3. **Speaker diarization** (`fir` mode) — pyannote.audio assigns speaker turns,
   with automatic or manual speaker-count bounds.
4. **Text normalization** — an extensive, from-scratch Hindi normalization
   engine (all defined in [`src/pipeline.py`](src/pipeline.py)) that:
   - converts spoken number words to Devanagari numerals and resolves compound
     thousands/hundreds/years (e.g. "दो हज़ार पच्चीस" → "२०२५")
   - formats dates (multiple spoken styles + `DD/MM/YYYY` slash notation),
     times, ages, and heights
   - formats phone numbers, Aadhaar, PAN, driving license, bank account/IFSC,
     passport, and vehicle registration numbers into their standard patterns
   - normalizes currency amounts, pincodes, addresses, police ranks, and legal
     section references
   - applies Hindi sentence-level phonetic/punctuation cleanup
5. **Dialogue realignment** (`fir` mode, optional) — an LLM call (via a locally
   hosted [vLLM](https://github.com/vllm-project/vllm) server) that re-attributes
   turns to the correct speaker label and can draft a structured statement from
   the diarized transcript, with automatic fallback to the raw diarized text if
   the LLM service is unavailable.

## Project structure

```
Voice-Text-Hindi/
├── src/
│   └── pipeline.py   # CLI entry point, STT wrapper, diarization, and all Hindi text normalization
├── requirements.txt
└── LICENSE
```

## Setup

```bash
# System dependency used for audio decoding/resampling
sudo apt-get install ffmpeg

pip install -r requirements.txt
```

The Hindi STT model (`jiviai/audioX-north-v1`) is downloaded automatically from
Hugging Face on first use and cached under `~/.cache/huggingface`.

For `fir` mode's diarization step, accept the terms for
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
on Hugging Face and set an access token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

Dialogue realignment/drafting in `fir` mode is optional and talks to a local
vLLM server; if it isn't running, the pipeline falls back to the raw diarized
transcript. To enable it:

```bash
export VLLM_URL=http://localhost:8000/v1/chat/completions
export SPELL_CHECK_MODEL=google/gemma-2-27b-it   # whatever chat model vLLM is serving
```

**Note:** this pipeline requires a CUDA-capable GPU — it deliberately refuses to
run on CPU (`torch.cuda.is_available()` is checked at startup).

## Usage

```bash
# High-accuracy Hindi transcription
python src/pipeline.py --mode nemo --audio path/to/audio.wav

# Multi-speaker statement: diarization + dialogue realignment
python src/pipeline.py --mode fir --audio path/to/audio.wav --min-speakers 2 --max-speakers 4

# Subtitles with timestamps
python src/pipeline.py --mode whisperx --audio path/to/audio.wav
```

Key environment variables (all optional, with sensible defaults/auto-detection):

| Variable | Purpose |
|---|---|
| `AUDIO_FILE` | Default audio path when `--audio` isn't passed |
| `HINDI_STT_MODEL` | Hugging Face model id for the Hindi STT model |
| `HF_TOKEN` | Hugging Face token for gated diarization models |
| `PYANNOTE_MODEL` | Diarization model name/id |
| `VLLM_URL`, `VLLM_MODEL`, `SPELL_CHECK_MODEL` | Local LLM dialogue-realignment/drafting service |

## Notes

- No audio samples, model checkpoints, or generated transcripts are included in
  this repo — only the pipeline code.
- Built around [`jiviai/audioX-north-v1`](https://huggingface.co/jiviai/audioX-north-v1)
  for Hindi ASR, with pyannote.audio for diarization and WhisperX for the
  subtitle mode.
- See each dependency's own license/model card for usage terms.
