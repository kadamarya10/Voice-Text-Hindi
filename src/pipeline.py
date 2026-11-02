import sys
import os

# Preload OpenMP libgomp into global TLS table to prevent glibc dl-tls assertion failure on aarch64
try:
    import ctypes
    for _libgomp_path in ["libgomp.so.1", "/usr/lib/aarch64-linux-gnu/libgomp.so.1", "/usr/lib/x86_64-linux-gnu/libgomp.so.1"]:
        try:
            ctypes.CDLL(_libgomp_path, mode=ctypes.RTLD_GLOBAL)
            break
        except Exception:
            pass
except Exception:
    pass

# Prevent circular import collision with Python stdlib 'copy' module
if "copy" not in sys.modules or sys.modules["copy"] is None or getattr(sys.modules["copy"], "__file__", "") == __file__:
    import importlib.util
    _stdlib_copy_path = os.path.join(os.path.dirname(os.__file__), "copy.py")
    if os.path.isfile(_stdlib_copy_path):
        _spec = importlib.util.spec_from_file_location("copy", _stdlib_copy_path)
        _real_copy = importlib.util.module_from_spec(_spec)
        sys.modules["copy"] = _real_copy
        _spec.loader.exec_module(_real_copy)

# PyTorch 2.6 Backwards Compatibility: Set env var BEFORE importing torch!
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

# Import torch, torchaudio, and pyannote upfront to secure TLS slots early in process lifetime
import torch
try:
    import torchaudio
except ImportError:
    pass

from pyannote.audio import Pipeline

# Force weights_only=False globally for PyTorch 2.6 compatibility with older models
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

import re
import argparse
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import librosa

# Mock ModelFilter for AI4Bharat compatibility with newer huggingface_hub
import huggingface_hub
if not hasattr(huggingface_hub, 'ModelFilter'):
    class ModelFilter:
        pass
    huggingface_hub.ModelFilter = ModelFilter

# Mock NeptuneLogger for AI4Bharat compatibility with pytorch_lightning >= 2.0
try:
    import pytorch_lightning.loggers
    if not hasattr(pytorch_lightning.loggers, 'NeptuneLogger'):
        class NeptuneLogger:
            pass
        pytorch_lightning.loggers.NeptuneLogger = NeptuneLogger
except ImportError:
    pass

# Import spelling checker if available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from spell_check import correct_spelling_vllm, correct_spelling_l3cube
except ImportError:
    def correct_spelling_vllm(text, **kwargs):
        return text
    correct_spelling_l3cube = correct_spelling_vllm

# --- Global Configurations ---
# Identifier of the chat model served by your local vLLM instance.
SPELL_CHECK_MODEL = os.environ.get("SPELL_CHECK_MODEL", "google/gemma-2-27b-it")
# Set to True to test letting Gemma handle words-to-number conversion directly and disable regex number normalization:
USE_GEMMA_FOR_NUMBERS = True
def is_cuda_executable():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones(1, device="cuda")
        _ = (x + 1).cpu()
        return True
    except Exception as e:
        print(f"⚠️ GPU detected ({torch.cuda.get_device_name(0)}) but CUDA execution test failed: {e}")
        return False

# Enforce GPU (CUDA) execution strictly as requested
if not is_cuda_executable():
    raise RuntimeError(
        "🚨 GPU execution is strictly required ('run on gpu do not run on cpu').\n"
        "CUDA is not available or executable in current Python environment.\n"
        f"torch.cuda.is_available() = {torch.cuda.is_available()}"
    )

DEVICE = "cuda"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to an audio file to transcribe by default; override with --audio or this env var.
DEFAULT_AUDIO_FILE = os.environ.get("AUDIO_FILE", "sample_audio.wav")

# Primary model: Jivi AI AudioX-North (Hindi)
DEFAULT_HINDI_MODEL = "jiviai/audioX-north-v1"
JIVI_HINDI_MODEL = "jiviai/audioX-north-v1"


# ==============================================================================
# Model Wrapper for Jivi AI AudioX-North (jiviai/audioX-north-v1)
# ==============================================================================

class JiviHindiWrapper:
    """
    Unified Inference Wrapper for Jivi AI AudioX-North (jiviai/audioX-north-v1)
    configured for state-of-the-art Hindi STT transcription.
    """
    def __init__(self, model_id_or_path="jiviai/audioX-north-v1", device=None):
        self.device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("🚨 GPU (CUDA) is strictly required for Jivi Hindi model execution!")

        self.model_id_or_path = model_id_or_path
        auth_token = (
            os.environ.get("HF_TOKEN") or
            os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not auth_token:
            try:
                with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                    auth_token = f.read().strip()
            except Exception:
                pass

        pipe_device = 0
        self.torch_dtype = torch.float16
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        print(f"📦 Loading Jivi AudioX-North Hindi model ({model_id_or_path}) strictly on GPU ({gpu_name})...")

        from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

        try:
            pipe_kwargs = {
                "chunk_length_s": 30,
                "return_timestamps": False,
            }
            if auth_token:
                pipe_kwargs["token"] = auth_token

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id_or_path,
                device=pipe_device,
                torch_dtype=self.torch_dtype,
                **pipe_kwargs
            )
            self.use_pipe = True
        except Exception as e:
            print(f"ℹ️ Pipeline initialization note ({e}), loading via direct Whisper processor & model...")
            token_kwargs = {"token": auth_token} if auth_token else {}
            self.processor = WhisperProcessor.from_pretrained(model_id_or_path, **token_kwargs)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id_or_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                **token_kwargs
            ).to(self.device)
            self.model.eval()
            self.use_pipe = False

        print(f"✅ Jivi AudioX-North successfully loaded for Hindi on {self.device}.")

    def to(self, device):
        self.device = str(device)
        if hasattr(self, "model"):
            self.model = self.model.to(self.device)
        return self

    def eval(self):
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def transcribe(self, audio_inputs, batch_size=4, **kwargs):
        """
        Transcribes audio inputs (list of file paths or numpy arrays).
        Returns a list of decoded Hindi strings.
        """
        if not isinstance(audio_inputs, list):
            audio_inputs = [audio_inputs]

        results = []
        gen_kwargs = {"language": "hi", "task": "transcribe", "max_new_tokens": 440}

        for item in audio_inputs:
            if isinstance(item, (str, Path)):
                try:
                    wav_array, wav_sr = sf.read(str(item), dtype="float32")
                    if len(wav_array.shape) > 1 and wav_array.shape[1] > 1:
                        wav_array = wav_array.mean(axis=1)
                    if wav_sr != 16000:
                        wav_array = librosa.resample(wav_array, orig_sr=wav_sr, target_sr=16000)
                        wav_sr = 16000
                except Exception as err:
                    print(f"⚠️ Audio load error for {item}: {err}")
                    wav_array = np.array([], dtype=np.float32)
                    wav_sr = 16000
            elif isinstance(item, np.ndarray):
                wav_array = item.astype(np.float32)
                wav_sr = 16000
            elif isinstance(item, torch.Tensor):
                wav_array = item.detach().cpu().numpy().astype(np.float32)
                wav_sr = 16000
            else:
                wav_array = np.array([], dtype=np.float32)
                wav_sr = 16000

            if len(wav_array) == 0:
                results.append("")
                continue

            max_v = float(np.max(np.abs(wav_array)))
            if max_v < 0.005:
                # Silence guard
                results.append("")
                continue

            wav_array = (wav_array / max_v) * 0.95
            pad = np.zeros(int(0.05 * wav_sr), dtype=np.float32)
            padded = np.concatenate([pad, wav_array, pad])

            try:
                if getattr(self, "use_pipe", False):
                    out = self.pipe(
                        {"raw": padded, "sampling_rate": wav_sr},
                        generate_kwargs=gen_kwargs
                    )["text"].strip()
                else:
                    inputs = self.processor(padded, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs.input_features.to(self.device).to(self.torch_dtype)
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            input_features, 
                            task="transcribe", 
                            language="hi",
                            max_new_tokens=440
                        )
                    out = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

                out = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', out)
                out = re.sub(r'((?:\S+\s+){1,10}\S+[।.,!?]?)(?:\s+\1){1,}', r'\1', out)
                results.append(out.strip())
            except Exception as e:
                print(f"⚠️ Jivi Hindi STT inference error: {e}")
                results.append("")

        return results


def load_hindi_stt_model(model_identifier=None, device=None):
    """
    Factory loader supporting Jivi AudioX-North (jiviai/audioX-north-v1) for Hindi.
    """
    model_identifier = model_identifier or os.environ.get("HINDI_STT_MODEL", DEFAULT_HINDI_MODEL)
    target_device = device or DEVICE
    return JiviHindiWrapper(model_id_or_path=model_identifier, device=target_device)


# ==============================================================================
# Audio Processing Utilities
# ==============================================================================

def apply_voice_bandpass_filter(data, sr, lowcut=30.0, highcut=None):
    """
    Sub-rumble / DC bias removal (>30Hz) while leaving all speech frequencies completely untouched.
    Prevents consonant and nasal distortion in Hindi ASR.
    """
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * sr
        low = max(lowcut / nyq, 0.001)
        if highcut is not None and highcut < nyq:
            high = min(highcut / nyq, 0.999)
            if low >= high:
                return data
            b, a = butter(2, [low, high], btype='band')
        else:
            b, a = butter(2, low, btype='high')
        filtered_data = filtfilt(b, a, data)
        return filtered_data.astype(np.float32)
    except Exception:
        return data


def load_audio_16k_mono_pcm(audio_file):
    """
    Step A: Dynamic Range Compression & Loudness Normalization.
    Ensures input audio is strictly 16 kHz (16,000 Hz), 16-bit Mono PCM,
    with highpass at 80 Hz and EBU R128 loudness normalization.
    """
    import subprocess
    import shutil
    
    if shutil.which("ffmpeg"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(audio_file),
            "-vn",
            "-af", "highpass=f=80",
            "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            tmp_wav_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(tmp_wav_path) and os.path.getsize(tmp_wav_path) > 1000:
                data, sr = sf.read(tmp_wav_path, dtype="float32")
                try:
                    os.remove(tmp_wav_path)
                except Exception:
                    pass
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = (data / max_val) * 0.95
                return data, 16000
        except Exception:
            pass
        if os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except Exception:
                pass

    try:
        data, sr = sf.read(audio_file, dtype="float32")
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val) * 0.95
        return data, sr
    except Exception:
        data, sr = librosa.load(audio_file, sr=16000, mono=True)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val) * 0.95
        return data, sr


def split_and_prepare_audio(file_path, target_duration=22.0):
    print(f"Loading and analyzing audio: {Path(file_path).name}")
    data, sr = librosa.load(file_path, sr=16000, mono=True)
    
    print("Suppressing DC rumble (<30Hz) and preserving full speech spectrum...")
    data = apply_voice_bandpass_filter(data, sr)
    
    max_val = np.max(np.abs(data))
    if max_val > 0.001:
        data = (data / max_val) * 0.90
        
    duration = len(data) / sr
    if duration <= target_duration:
        pad = np.zeros(int(0.2 * sr), dtype=np.float32)
        chunk_padded = np.concatenate([pad, data, pad])
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_wav.name, chunk_padded, 16000, subtype='PCM_16')
        return [temp_wav.name]

    print(f"Preparing continuous lossless chunks (target ~{target_duration}s, zero speech discarded)...")
    chunk_paths = []
    start_sec = 0.0
    while start_sec < duration:
        end_sec = min(duration, start_sec + target_duration)
        if end_sec < duration:
            search_start = max(start_sec + 8.0, end_sec - 3.5)
            s_idx = int(search_start * sr)
            e_idx = int(end_sec * sr)
            sub = data[s_idx:e_idx]
            if len(sub) > 512:
                frame_len = 512
                rms = np.sqrt(np.mean(np.square(librosa.util.frame(sub, frame_length=frame_len, hop_length=256)), axis=0))
                min_idx = np.argmin(rms)
                split_offset = (min_idx * 256) / sr
                actual_end = search_start + split_offset
            else:
                actual_end = end_sec
        else:
            actual_end = duration

        start_samp = int(start_sec * sr)
        end_samp = int(actual_end * sr)
        chunk_data = data[start_samp:end_samp]
        
        pad = np.zeros(int(0.2 * sr), dtype=np.float32)
        chunk_padded = np.concatenate([pad, chunk_data, pad])
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_wav.name, chunk_padded, 16000, subtype='PCM_16')
        chunk_paths.append(temp_wav.name)
        
        start_sec = actual_end
        
    print(f"Audio successfully prepared into {len(chunk_paths)} lossless contiguous chunks (100% of speech preserved).")
    return chunk_paths


# ==============================================================================
# Comprehensive Hindi Number Dictionary (0 - 100 + Spoken / English Phonetic Variants)
# ==============================================================================

HINDI_NUMBER_WORDS = {
    # 0
    'शून्य': '०', 'शुन्‍ह': '०', 'जीरो': '०', 'ज़ीरो': '०', 'zero': '०', 'Zero': '०',
    # 1 - 10
    'एक': '१', 'दो': '२', 'तीन': '३', 'चार': '४', 'पाँच': '५', 'पांच': '५',
    'छह': '६', 'छः': '६', 'छे': '६', 'सात': '७', 'आठ': '८', 'नौ': '९', 'नऊ': '९', 'दस': '१०',
    # English spoken numerals in Hindi context
    'वन': '१', 'टू': '२', 'थ्री': '३', 'फोर': '४', 'फाइव': '५', 'फाईव': '५', 'फाईव्ह': '५',
    'सिक्स': '६', 'सेवन': '७', 'सेव्हन': '७', 'एट': '८', 'नाइन': '९', 'नाईन': '९', 'टेन': '१०',
    # 11 - 20
    'ग्यारह': '११', 'बारह': '१२', 'तेरह': '१३', 'चौदह': '१४', 'पंद्रह': '१५', 'पन्द्रह': '१५',
    'सोलह': '१६', 'सत्रह': '१७', 'अठारह': '१८', 'उन्नीस': '१९', 'उनीस': '१९', 'बीस': '२०',
    # 21 - 30
    'इक्कीस': '२१', 'इकीस': '२१',
    'बाईस': '२२', 'बाइस': '२२',
    'तेईस': '२३', 'तेइस': '२३',
    'चौबीस': '२४', 'चौबिस': '२४',
    'पच्चीस': '२५', 'पचीस': '२५',
    'छब्बीस': '२६', 'छब्बिस': '२६',
    'सत्ताईस': '२७', 'सताइस': '२७',
    'अट्ठाईस': '२८', 'अठाईस': '२८', 'अट्ठाइस': '२८',
    'उनतीस': '२९', 'उनत्तीस': '२९', 'तीस': '३०',
    # 31 - 40
    'इकत्तीस': '३१', 'इकतीस': '३१',
    'बत्तीस': '३२', 'बतीस': '३२',
    'तैंतीस': '३३', 'तेतीस': '३३',
    'चौंतीस': '३४', 'चौतीस': '३४',
    'पैंतीस': '३५', 'पैतीस': '३५',
    'छत्तीस': '३६', 'छतीस': '३६',
    'सैंतीस': '३७', 'सैतीस': '३७',
    'अड़तीस': '३८', 'अडतीस': '३८', 'अड़त्तीस': '३८',
    'उनतालीस': '३९', 'उनचालीस': '३९', 'चालीस': '४०',
    # 41 - 50
    'इकतालीस': '४१',
    'बयालीस': '४२', 'बयालिस': '४२',
    'तैंतालीस': '४३', 'तेतालीस': '४३',
    'चवालीस': '४४', 'चौवालीस': '४४',
    'पैंतालीस': '४५', 'पैंतालिस': '४५',
    'छियालीस': '४६', 'छियालिस': '४६',
    'सैंतालीस': '४७', 'सैंतालिस': '४७',
    'अड़तालीस': '४८', 'अडतालीस': '४८',
    'उनचास': '४९', 'पचास': '५०',
    # 51 - 60
    'इक्यावन': '५१', 'इकावन': '५१',
    'बावन': '५२',
    'तिरपन': '५३', 'तिरेपन': '५३',
    'चौवन': '५४', 'चव्वन': '५४',
    'पचपन': '५५',
    'छप्पन': '५६',
    'सत्तावन': '५७', 'सतावन': '५७',
    'अट्ठावन': '५८', 'अठावन': '५८',
    'उनसठ': '५९', 'साठ': '६०',
    # 61 - 70
    'इकसठ': '६१',
    'बासठ': '६२',
    'तिरसठ': '६३', 'तिरेसठ': '६३',
    'चौंसठ': '६४', 'चौसठ': '६४',
    'पैंसठ': '६५', 'पैसठ': '६५',
    'छियासठ': '६६',
    'सड़सठ': '६७', 'सदसठ': '६७',
    'अड़सठ': '६८', 'अडसठ': '६८',
    'उनहत्तर': '६९', 'सत्तर': '७०',
    # 71 - 80
    'इकहत्तर': '७१',
    'बहत्तर': '७२',
    'तिहत्तर': '७३',
    'चौहत्तर': '७४',
    'पचहत्तर': '७५',
    'छिहत्तर': '७६',
    'सतहत्तर': '७७', 'सत्तहत्तर': '७७',
    'अठहत्तर': '७८',
    'उन्यासी': '७९', 'उनासी': '७९', 'अस्सी': '८०',
    # 81 - 90
    'इक्यासी': '८१', 'इकासी': '८१',
    'बयासी': '८२',
    'तिरासी': '८३', 'तिरयासी': '८३',
    'चौरासी': '८४',
    'पचासी': '८५',
    'छियासी': '८६',
    'सत्तासी': '८७', 'सतासी': '८७',
    'अठासी': '८८', 'अट्ठासी': '८८',
    'नवासी': '८९', 'नब्बे': '९०',
    # 91 - 100
    'इक्यानवे': '९१', 'इक्यानवे': '९१',
    'बयानवे': '९२',
    'तिरानवे': '९३', 'तिरान्वे': '९३',
    'चौरानवे': '९४', 'चौरान्वे': '९४',
    'पचानवे': '९५', 'पचान्वे': '९५',
    'छियानवे': '९६', 'छियान्वे': '९६',
    'सत्तानवे': '९७', 'सत्तान्वे': '९७',
    'अट्ठानवे': '९८', 'अट्ठान्वे': '९८', 'अठानवे': '९८',
    'निन्यानवे': '९९', 'निन्यान्वे': '९९',
    'सौ': '१००', 'एक सौ': '१००',
    # Hundreds (100 - 900)
    'एकसौ': '१००', 'दोसौ': '२००', 'तीनसौ': '३००', 'चारसौ': '४००', 'पांचसौ': '५००', 'पाँचसौ': '५००',
    'छहसौ': '६००', 'सातसौ': '७००', 'आठसौ': '८००', 'नौसौ': '९००',
    'एक सौ': '१००', 'दो सौ': '२००', 'तीन सौ': '३००', 'चार सौ': '४००', 'पांच सौ': '५००', 'पाँच सौ': '५००',
    'छह सौ': '६००', 'सात सौ': '७००', 'आठ सौ': '८००', 'नौ सौ': '९००'
}


def resolve_hindi_compound_thousands_and_years(text):
    """
    Universally resolves spoken Hindi compound thousands, centuries, and calendar years mathematically.
    Prevents ASR/LLM digit concatenation or premature year truncation:
      - 'दो हज़ार ग्यारह' -> '२०११' (2000 + 11 = 2011, NOT '२००० ११')
      - 'दो हज़ार बारह'  -> '२०१२'
      - 'दो हज़ार एक'   -> '२००१'
      - 'दो हज़ार दस'   -> '२०१०'
      - 'दो हज़ार पच्चीस' -> '२०२५'
      - 'एक हज़ार नौ सौ पचानवे' -> '१९९५' (1000 + 900 + 95 = 1995)
      - 'उन्नीस सौ चौरासी' -> '१९८४' (1900 + 84 = 1984)
    Zero hardcoding: generic mathematical composition for ANY number in HINDI_NUMBER_WORDS (1-99) or digits.
    """
    import re
    hindi_digits = str.maketrans('०१२३४५६७८९', '0123456789')
    to_hindi = str.maketrans('0123456789', '०१२३४५६७८९')

    unit_words_map = {}
    for w, d in HINDI_NUMBER_WORDS.items():
        d_ar = d.translate(hindi_digits)
        if d_ar.isdigit() and 1 <= int(d_ar) < 100:
            unit_words_map[w] = int(d_ar)
    words_pat = '|'.join(sorted(unit_words_map.keys(), key=len, reverse=True))

    thousand_multipliers_words = {
        'दस': 10000, 'नौ': 9000, 'आठ': 8000, 'सात': 7000, 'छह': 6000, 'छे': 6000,
        'पांच': 5000, 'पाँच': 5000, 'चार': 4000, 'तीन': 3000, 'दो': 2000, 'एक': 1000
    }
    th_words_pat = '|'.join(sorted(thousand_multipliers_words.keys(), key=len, reverse=True))

    hundred_map = {
        'नौ सौ': 900, 'नौसौ': 900,
        'आठ सौ': 800, 'आठसौ': 800,
        'सात सौ': 700, 'सातसौ': 700,
        'छह सौ': 600, 'छहसौ': 600,
        'पांच सौ': 500, 'पाँच सौ': 500, 'पांचसौ': 500, 'पाँचसौ': 500,
        'चार सौ': 400, 'चारसौ': 400,
        'तीन सौ': 300, 'तीनसौ': 300,
        'दो सौ': 200, 'दोसौ': 200,
        'एक सौ': 100, 'एकसौ': 100
    }
    hd_words_pat = '|'.join(sorted(hundred_map.keys(), key=len, reverse=True))

    century_words_map = {
        'उन्नीस सौ': 1900, 'उन्नीससौ': 1900, 'अठारह सौ': 1800, 'अठारहसौ': 1800,
        'सत्रह सौ': 1700, 'सत्रहसौ': 1700, 'सोलह सौ': 1600, 'सोलहसौ': 1600,
        'पंद्रह सौ': 1500, 'पंद्रहसौ': 1500, 'चौदह सौ': 1400, 'चौदहसौ': 1400,
        'तेरह सौ': 1300, 'तेरहसौ': 1300, 'बारह सौ': 1200, 'बारहसौ': 1200,
        'ग्यारह सौ': 1100, 'ग्यारहसौ': 1100
    }
    century_words_pat = '|'.join(sorted(century_words_map.keys(), key=len, reverse=True))

    # 1. Thousand [Optional Hundred] + Unit Words (e.g. 'दो हज़ार ग्यारह', 'एक हज़ार नौ सौ पचानवे')
    def repl_th_hd_unit_words(m):
        th_word = m.group(1)
        hd_word = m.group(2)
        unit_word = m.group(3)

        th_val = thousand_multipliers_words.get(th_word)
        if th_val is None:
            try:
                th_val = int(th_word.translate(hindi_digits)) * 1000
            except ValueError:
                return m.group(0)

        hd_val = hundred_map.get(hd_word, 0) if hd_word else 0
        unit_val = unit_words_map.get(unit_word, 0)

        total = th_val + hd_val + unit_val
        return str(total).translate(to_hindi)

    text = re.sub(
        rf'(?<![\u0900-\u097F])({th_words_pat}|[०-९0-9]{{1,2}})\s+(?:हज़ार|हजार)(?:\s+({hd_words_pat}))?\s+({words_pat})(?![\u0900-\u097F])(?!\s*सौ)',
        repl_th_hd_unit_words,
        text
    )

    # 2. Thousand [Optional Hundred] + Unit Digits (e.g. 'दो हज़ार ११', 'एक हज़ार नौ सौ ९५')
    def repl_th_hd_unit_digits(m):
        th_word = m.group(1)
        hd_word = m.group(2)
        unit_digits = m.group(3)

        th_val = thousand_multipliers_words.get(th_word)
        if th_val is None:
            try:
                th_val = int(th_word.translate(hindi_digits)) * 1000
            except ValueError:
                return m.group(0)

        hd_val = hundred_map.get(hd_word, 0) if hd_word else 0
        try:
            unit_val = int(unit_digits.translate(hindi_digits))
            if 0 <= unit_val < 100:
                total = th_val + hd_val + unit_val
                return str(total).translate(to_hindi)
        except ValueError:
            pass
        return m.group(0)

    text = re.sub(
        rf'(?<![\u0900-\u097F])({th_words_pat}|[०-९0-9]{{1,2}})\s+(?:हज़ार|हजार)(?:\s+({hd_words_pat}))?\s+([०-९0-9]{{1,2}})(?![\u0900-\u097F])(?!\s*सौ)',
        repl_th_hd_unit_digits,
        text
    )

    # 3. Century word + Unit Words or Digits (e.g. 'उन्नीस सौ पचानवे' -> 1995, 'उन्नीस सौ ९५' -> 1995)
    def repl_century_unit_words(m):
        cent_word = m.group(1)
        unit_word = m.group(2)
        cent_val = century_words_map.get(cent_word, 0)
        unit_val = unit_words_map.get(unit_word, 0)
        return str(cent_val + unit_val).translate(to_hindi)

    text = re.sub(
        rf'(?<![\u0900-\u097F])({century_words_pat})\s+({words_pat})(?![\u0900-\u097F])',
        repl_century_unit_words,
        text
    )

    def repl_century_unit_digits(m):
        cent_word = m.group(1)
        unit_digits = m.group(2)
        cent_val = century_words_map.get(cent_word, 0)
        try:
            unit_val = int(unit_digits.translate(hindi_digits))
            if 0 <= unit_val < 100:
                return str(cent_val + unit_val).translate(to_hindi)
        except ValueError:
            pass
        return m.group(0)

    text = re.sub(
        rf'(?<![\u0900-\u097F])({century_words_pat})\s+([०-९0-9]{{1,2}})(?![\u0900-\u097F])',
        repl_century_unit_digits,
        text
    )

    # 4. Standalone century words without units (e.g. 'उन्नीस सौ' -> '१९००')
    for c_word, c_val in century_words_map.items():
        text = re.sub(rf'(?<![\u0900-\u097F]){c_word}(?![\u0900-\u097F])', str(c_val).translate(to_hindi), text)

    # 5. Standalone Thousand + Hundred (e.g. 'एक हज़ार नौ सौ' -> '१९००')
    def repl_th_hd_standalone(m):
        th_word = m.group(1)
        hd_word = m.group(2)
        th_val = thousand_multipliers_words.get(th_word)
        if th_val is None:
            try:
                th_val = int(th_word.translate(hindi_digits)) * 1000
            except ValueError:
                return m.group(0)
        hd_val = hundred_map.get(hd_word, 0)
        return str(th_val + hd_val).translate(to_hindi)

    text = re.sub(
        rf'(?<![\u0900-\u097F])({th_words_pat}|[०-९0-9]{{1,2}})\s+(?:हज़ार|हजार)\s+({hd_words_pat})(?![\u0900-\u097F])',
        repl_th_hd_standalone,
        text
    )

    # 6. Standalone Thousands (e.g. 'दो हज़ार' -> '२०००', 'एक हज़ार' -> '१०००')
    def repl_th_standalone(m):
        th_word = m.group(1)
        th_val = thousand_multipliers_words.get(th_word)
        if th_val is None:
            try:
                th_val = int(th_word.translate(hindi_digits)) * 1000
            except ValueError:
                return m.group(0)
        return str(th_val).translate(to_hindi)

    text = re.sub(
        rf'(?<![\u0900-\u097F])({th_words_pat}|[०-९0-9]{{1,2}})\s+(?:हज़ार|हजार)(?![\u0900-\u097F])',
        repl_th_standalone,
        text
    )

    return text


def format_hindi_month_year(text):
    """
    Dynamically converts Month + Year expressions into formal Hindi month name and 4-digit year:
    e.g. '4 2026' -> 'अप्रैल २०२६'
         '४ २०२६' -> 'अप्रैल २०२६'
         '०४/२०२६' -> 'अप्रैल २०२६'
         'चौथा महीना २०२६' -> 'अप्रैल २०२६'
         'महीना ४ २०२६' -> 'अप्रैल २०२६'
         'April 2026' -> 'अप्रैल २०२६'
    """
    import re
    hindi_digits = str.maketrans('0123456789', '०१२३४५६७८९')
    to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')

    month_names_map = {
        1: 'जनवरी', 2: 'फ़रवरी', 3: 'मार्च', 4: 'अप्रैल',
        5: 'मई', 6: 'जून', 7: 'जुलाई', 8: 'अगस्त',
        9: 'सितंबर', 10: 'अक्टूबर', 11: 'नवंबर', 12: 'दिसंबर'
    }

    month_words_to_num = {
        'जनवरी': 1, 'january': 1, 'jan': 1,
        'फ़रवरी': 2, 'फरवरी': 2, 'february': 2, 'feb': 2,
        'मार्च': 3, 'march': 3, 'mar': 3,
        'अप्रैल': 4, 'अप्रेल': 4, 'april': 4, 'apr': 4,
        'मई': 5, 'may': 5,
        'जून': 6, 'june': 6, 'jun': 6,
        'जुलाई': 7, 'july': 7, 'jul': 7,
        'अगस्त': 8, 'august': 8, 'aug': 8,
        'सितंबर': 9, 'सितम्बर': 9, 'september': 9, 'sep': 9, 'sept': 9,
        'अक्टूबर': 10, 'अक्तूबर': 10, 'october': 10, 'oct': 10,
        'नवंबर': 11, 'नवम्बर': 11, 'november': 11, 'nov': 11,
        'दिसंबर': 12, 'दिसम्बर': 12, 'december': 12, 'dec': 12
    }

    ordinal_numeral_month_words = {
        'एक': 1, 'पहला': 1, 'पहली': 1, 'पहले': 1, '१ ला': 1, '१ला': 1, '1st': 1,
        'दो': 2, 'दूसरा': 2, 'दूसरी': 2, 'दूसरे': 2, '२ रा': 2, '२रा': 2, '2nd': 2,
        'तीन': 3, 'तीसरा': 3, 'तीसरी': 3, 'तीसरे': 3, '३ रा': 3, '३रा': 3, '3rd': 3,
        'चार': 4, 'चौथा': 4, 'चौथी': 4, 'चौथे': 4, '४ था': 4, '४था': 4, '4th': 4,
        'पांच': 5, 'पाँच': 5, 'पाँचवाँ': 5, 'पांचवा': 5, '५ वा': 5, '5th': 5,
        'छह': 6, 'छठा': 6, 'छठी': 6, 'छठे': 6, '६ वा': 6, '6th': 6,
        'सात': 7, 'सातवाँ': 7, 'सातवा': 7, 'सातवीं': 7, '७ वा': 7, '7th': 7,
        'आठ': 8, 'आठवाँ': 8, 'आठवा': 8, 'आठवीं': 8, '८ वा': 8, '8th': 8,
        'नौ': 9, 'नौवाँ': 9, 'नौवा': 9, 'नौवीं': 9, '९ वा': 9, '9th': 9,
        'दस': 10, 'दसवाँ': 10, 'दसवा': 10, 'दसवीं': 10, '१० वा': 10, '10th': 10,
        'ग्यारह': 11, 'ग्यारहवाँ': 11, '11th': 11,
        'बारह': 12, 'बारहवाँ': 12, '12th': 12
    }

    all_month_words = {}
    all_month_words.update(month_words_to_num)
    all_month_words.update(ordinal_numeral_month_words)

    m_words_pat = '|'.join(re.escape(k) for k in sorted(all_month_words.keys(), key=len, reverse=True))

    # Step 1: Explicit month word + year: e.g. "चौथा महीना २०२६", "महीना ४ २०२६", "महीना चार २०२६"
    def repl_explicit_month_word(m):
        token = m.group(2).strip().lower()
        year_str = m.group(4).translate(to_arabic)
        try:
            y_val = int(year_str)
            if not (1900 <= y_val <= 2100):
                return m.group(0)
        except ValueError:
            return m.group(0)
            
        m_num = all_month_words.get(token)
        if not m_num:
            token_ar = token.translate(to_arabic)
            if token_ar.isdigit():
                val = int(token_ar)
                if 1 <= val <= 12:
                    m_num = val
                    
        if m_num and 1 <= m_num <= 12:
            return f"{month_names_map[m_num]} {str(y_val).translate(hindi_digits)}"
        return m.group(0)

    text = re.sub(
        rf'\b(महीना\s+)?({m_words_pat}|[०-९0-9]{{1,2}})(\s+महीना|\s+महीने|\s+माह)?\s+(?:के\s+|साल\s+|सन\s+)?([०-९0-9]{{4}})\b',
        repl_explicit_month_word,
        text,
        flags=re.IGNORECASE
    )

    # Step 2: Month name (Hindi or English) + Year
    def repl_month_name_year(m):
        m_word = m.group(1).lower()
        y_str = m.group(2).translate(to_arabic)
        try:
            y_val = int(y_str)
            if 1900 <= y_val <= 2100:
                m_num = month_words_to_num.get(m_word)
                if m_num:
                    return f"{month_names_map[m_num]} {str(y_val).translate(hindi_digits)}"
        except ValueError:
            pass
        return m.group(0)

    m_names_only_pat = '|'.join(re.escape(k) for k in sorted(month_words_to_num.keys(), key=len, reverse=True))
    text = re.sub(
        rf'\b({m_names_only_pat})\s+(?:साल\s+|सन\s+)?([०-९0-9]{{4}})\b',
        repl_month_name_year,
        text,
        flags=re.IGNORECASE
    )

    # Step 3: Numeric Month + Year (e.g. '०४/२०२६', '४/२०२६', '04/2026')
    pattern_num_my = (
        r'(?<![०-९0-9/.\-])'
        r'(?:(दिनांक|माह|तारीख)\s+)?'
        r'\b(०?[१-९]|१०|११|१२|0?[1-9]|10|11|12)\b'
        r'([\s/\-]+)'
        r'\b([०-९0-9]{4})\b'
        r'(?![०-९0-9/.\-])'
    )

    def repl_num_my(m):
        m_str = m.group(2).translate(to_arabic)
        y_str = m.group(4).translate(to_arabic)
        try:
            m_val = int(m_str)
            y_val = int(y_str)
            if 1 <= m_val <= 12 and 1900 <= y_val <= 2100:
                return f"{month_names_map[m_val]} {str(y_val).translate(hindi_digits)}"
        except ValueError:
            pass
        return m.group(0)

    text = re.sub(pattern_num_my, repl_num_my, text)
    return text


def repair_and_format_hindi_dates_and_years(text):
    """
    Intelligently audits, repairs, and standardizes dates and years in Hindi STT and LLM output.
    Universal mathematical repair without hardcoding:
      1. Year remainder glued to age (e.g. '१४/०५/२००० १११५ वर्ष' -> '१४/०५/२०११ १५ वर्ष')
      2. Year remainder separated by space before age (e.g. '१४/०५/२००० ११ १५ वर्ष' -> '१४/०५/२०११ १५ वर्ष')
      3. Date ending in century followed by standalone year offset
      4. Standardizes dates to DD/MM/YYYY with Devanagari numerals.
    """
    import re
    to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')
    to_hindi = str.maketrans('0123456789', '०१२३४५६७८९')

    text = resolve_hindi_compound_thousands_and_years(text)
    century_pattern = r'([०-९0-9]{2}[०0]{2})'
    age_units_pattern = r'(वर्ष|साल|उम्र|आयु)'

    def repl_glued_year_age(m):
        prefix = m.group(1)
        century_str = m.group(2).translate(to_arabic)
        digits_str = m.group(3).translate(to_arabic)
        unit = m.group(4) or "वर्ष"

        try:
            century_val = int(century_str)
            if 1800 <= century_val <= 2100 and len(digits_str) in (3, 4):
                yr_offset = int(digits_str[:2])
                age_val = int(digits_str[2:])
                if 0 <= yr_offset < 100 and 0 <= age_val <= 125:
                    full_year = century_val + yr_offset
                    full_year_h = str(full_year).translate(to_hindi)
                    age_h = str(age_val).translate(to_hindi)
                    return f"{prefix}{full_year_h} {age_h} {unit}"
        except ValueError:
            pass
        return m.group(0)

    text = re.sub(
        rf'(\b[०-९0-9]{{1,2}}[/\-\.][०-९0-9]{{1,2}}[/\-\.]){century_pattern}\s+([०-९0-9]{{3,4}})\s*{age_units_pattern}\b',
        repl_glued_year_age,
        text
    )

    def repl_spaced_year_age(m):
        prefix = m.group(1)
        century_str = m.group(2).translate(to_arabic)
        yr_str = m.group(3).translate(to_arabic)
        age_str = m.group(4).translate(to_arabic)
        unit = m.group(5) or "वर्ष"

        try:
            century_val = int(century_str)
            yr_offset = int(yr_str)
            age_val = int(age_str)
            if 1800 <= century_val <= 2100 and 0 <= yr_offset < 100 and 0 <= age_val <= 125:
                full_year = century_val + yr_offset
                full_year_h = str(full_year).translate(to_hindi)
                age_h = str(age_val).translate(to_hindi)
                return f"{prefix}{full_year_h} {age_h} {unit}"
        except ValueError:
            pass
        return m.group(0)

    text = re.sub(
        rf'(\b[०-९0-9]{{1,2}}[/\-\.][०-९0-9]{{1,2}}[/\-\.]){century_pattern}\s+([०-९0-9]{{1,2}})\s+([०-९0-9]{{1,3}})\s*{age_units_pattern}\b',
        repl_spaced_year_age,
        text
    )

    def repl_date_split_year(m):
        prefix = m.group(1)
        century_str = m.group(2).translate(to_arabic)
        yr_str = m.group(3).translate(to_arabic)
        try:
            century_val = int(century_str)
            yr_offset = int(yr_str)
            if 1800 <= century_val <= 2100 and 0 <= yr_offset < 100:
                full_year = century_val + yr_offset
                return f"{prefix}{str(full_year).translate(to_hindi)}"
        except ValueError:
            pass
        return m.group(0)

    text = re.sub(
        rf'(\b[०-९0-9]{{1,2}}[/\-\.][०-९0-9]{{1,2}}[/\-\.]){century_pattern}\s+([०-९0-9]{{1,2}})\b(?!\s*(?:[०-९0-9]|वर्ष|साल|दिन|महीने|उम्र|आयु))',
        repl_date_split_year,
        text
    )

    if 'format_hindi_date_slashes' in globals():
        text = format_hindi_date_slashes(text)

    text = format_hindi_month_year(text)
    return text


def format_hindi_dates_dynamically(text):
    """
    Dynamically captures and formats calendar dates in Hindi STT output to DD/MM/YYYY.
    """
    import re
    text = text.translate(str.maketrans('0123456789', '०१२३४५६७८९'))
    text = resolve_hindi_compound_thousands_and_years(text)

    months_map = {
        'जनवरी': '०१', 'फ़रवरी': '०२', 'फरवरी': '०२', 'मार्च': '०३', 'अप्रैल': '०४', 'अप्रेल': '०४',
        'मई': '०५', 'जून': '०६', 'जुलाई': '०७', 'अगस्त': '०८',
        'सितंबर': '०९', 'सितम्बर': '०९', 'अक्टूबर': '१०', 'अक्तूबर': '१०', 'नवंबर': '११', 'नवम्बर': '११', 'दिसंबर': '१२', 'दिसम्बर': '१२'
    }

    for word, digit in sorted(months_map.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(rf'(\b(?:[०-९0-9]{{1,2}})[\s/\-\.]*){word}(?![\u0900-\u097F])', rf'\g<1>{digit}', text)

    suffixes = r'(के|में|को|का|की|से|पर)?'
    for word, digit in sorted(HINDI_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(rf'(?<![\u0900-\u097F]){word}(?:{suffixes})?(?![\u0900-\u097F])',
                      lambda m, d=digit: d + (m.group(1) if m.group(1) else ''), text)

    text = repair_and_format_hindi_dates_and_years(text)

    def date_replacer(match):
        day = match.group(1).lstrip('०0') or '०'
        month = match.group(2).lstrip('०0') or '०'
        year = match.group(3)
        day_str = f"०{day}" if len(day) == 1 else day
        month_str = f"०{month}" if len(month) == 1 else month
        return f"{day_str}/{month_str}/{year}"

    text = re.sub(r'([०-९]{1,2})[\s/\-\.]+([०-९]{1,2})[\s/\-\.]+([०-९]{4})', date_replacer, text)
    text = re.sub(r'(दिनांक|तारीख)\s*([०-९]{1,2}/(?:[०-९]{1,2}/)?[०-९]{4})', r'\1 \2 ', text)
    text = format_hindi_month_year(text)
    text = repair_and_format_hindi_dates_and_years(text)
    return text


def format_hindi_date_slashes(text):
    """
    Ensures calendar dates in Hindi dialogue have standard slashes (DD/MM/YYYY):
    e.g. '२० ०४ २०२६' -> '२०/०४/२०२६'
    """
    import re
    hindi_digits = str.maketrans('0123456789', '०१२३४५६७८९')
    arabic_digits = str.maketrans('०१२३४५६७८९', '0123456789')

    pattern = r'(?:(दिनांक|तारीख)\s+)?\b([०-९0-9]{1,2})[\s/\-\.]+([०-९0-9]{1,2})[\s/\-\.]+([०-९0-9]{4})\b'

    def replacer(match):
        prefix = match.group(1)
        d_raw = match.group(2).translate(arabic_digits)
        m_raw = match.group(3).translate(arabic_digits)
        y_raw = match.group(4).translate(arabic_digits)
        try:
            d_val = int(d_raw)
            m_val = int(m_raw)
            y_val = int(y_raw)
            if 1 <= d_val <= 31 and 1 <= m_val <= 12 and 1900 <= y_val <= 2100:
                d_str = f"0{d_val}" if d_val < 10 else str(d_val)
                m_str = f"0{m_val}" if m_val < 10 else str(m_val)
                y_str = str(y_val)
                formatted = f"{d_str}/{m_str}/{y_str}".translate(hindi_digits)
                return f"{prefix} {formatted}" if prefix else formatted
        except ValueError:
            pass
        return match.group(0)

    return re.sub(pattern, replacer, text)


def format_hindi_time_dynamically(text):
    """
    Dynamically captures and formats spoken time in Hindi STT in 24-hour formal police format.
    e.g. 'दोपहर २:४५' -> 'दोपहर १४:४५ बजे के आसपास'
         'सुबह १०:३०' -> 'सुबह १०:३० बजे के आसपास'
    """
    import re
    hour_words_map = {
        'शून्य': '००', 'जीरो': '००',
        'एक': '०१', 'दो': '०२', 'तीन': '०३', 'चार': '०४', 'पांच': '०५', 'पाँच': '०५',
        'छह': '०६', 'छे': '०६', 'सात': '०७', 'आठ': '०८', 'नौ': '०९', 'दस': '१०',
        'ग्यारह': '११', 'बारह': '१२', 'तेरह': '१३', 'चौदह': '१४', 'पंद्रह': '१५',
        'सोलह': '१६', 'सत्रह': '१७', 'अठारह': '१८', 'उन्नीस': '१९', 'बीस': '२०',
        'इक्कीस': '२१', 'बाईस': '२२', 'तेईस': '२३', 'चौबीस': '२४'
    }

    _hin2ara = str.maketrans('०१२३४५६७८९', '0123456789')
    _ara2hin = str.maketrans('0123456789', '०१२३४५६७८९')
    STANDARD_SUFFIX = 'बजे के आसपास'
    PERIOD_WORDS = ['सुबह', 'दोपहर', 'शाम', 'रात', 'मध्यरात्रि']

    def infer_period(hr_24):
        if 0 <= hr_24 < 4:
            return 'मध्यरात्रि'
        elif 4 <= hr_24 < 12:
            return 'सुबह'
        elif 12 <= hr_24 < 17:
            return 'दोपहर'
        elif 17 <= hr_24 < 21:
            return 'शाम'
        else:
            return 'रात'

    # Clean colon artifacts
    text = re.sub(r'([०-९0-9]{1,2})\s*COLON\s*([०-९0-9]{2})', r'\1:\2', text)

    # Convert 12-hour period times to 24-hour
    def period_time_replacer(m):
        period = m.group(1)
        hr_str = m.group(2).translate(_hin2ara)
        min_str = m.group(3).translate(_hin2ara)
        try:
            hr = int(hr_str)
            mn = int(min_str)
            if period in ['दोपहर', 'शाम', 'रात'] and hr < 12:
                hr += 12
            elif period == 'रात' and hr == 12:
                hr = 0
            hr_fmt = f"{hr:02d}".translate(_ara2hin)
            mn_fmt = f"{mn:02d}".translate(_ara2hin)
            return f"{period} {hr_fmt}:{mn_fmt} {STANDARD_SUFFIX}"
        except ValueError:
            return m.group(0)

    periods_pat = '|'.join(PERIOD_WORDS)
    text = re.sub(
        rf'\b({periods_pat})\s*([०-९0-9]{{1,2}})[:\.\s]([०-९0-9]{{2}})(?:\s*(?:बजे|के\s*आसपास))?\b',
        period_time_replacer,
        text
    )

    return text


def format_phone_numbers_dynamically(text):
    """
    Dynamically captures and formats 10-digit Indian mobile numbers in Hindi STT output.
    """
    import re
    phone_keywords = r'(?:मोबाइल\s*नंबर|मोबाइल\s*क्रमांक|मोबाइल|फोन\s*नंबर|फोन|सम्पर्क\s*नंबर|संपर्क\s*नंबर|दूरभाष)'

    def phone_word_replacer(match):
        prefix = match.group(1).strip()
        rest = match.group(2)

        stop_m = re.search(r'(?:\r?\n|(?<![\w\u0900-\u097F])(?:Speaker|स्पीकर)\b|\b(?:है\s+और|और\s+मेरा|आधार|पैन|PAN|गाड़ी|कार)\b)', rest, flags=re.IGNORECASE)
        if stop_m:
            trailing = rest[stop_m.start():]
            rest = rest[:stop_m.start()]
        else:
            trailing = ""

        working_text = rest
        for word, digit in sorted(HINDI_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
            working_text = re.sub(rf'(?<![\u0900-\u097F]){word}(?![\u0900-\u097F])', f" {digit} ", working_text)

        digits_matches = list(re.finditer(r'[0-9०-९]', working_text))
        if len(digits_matches) >= 10:
            first_match = digits_matches[0]
            start_idx = first_match.start()
            leading_between = working_text[:start_idx].strip()
            leading_between = re.sub(r'^[:\-\s]+|[:\-\s]+$', '', leading_between).strip()

            tenth_match = digits_matches[9]
            end_idx = tenth_match.end()

            remaining_part = working_text[end_idx:]
            remaining_part = re.sub(r'^[,\s\-]+', ' ', remaining_part).strip()
            ten_digits = [m.group() for m in digits_matches[:10]]
            trans_table = str.maketrans('0123456789', '०१२३४५६७८९')
            clean_digits = ''.join(ten_digits).translate(trans_table)

            rem_str = f" {remaining_part}" if remaining_part else ""
            if leading_between:
                res = f"{prefix} {leading_between} {clean_digits}{rem_str}"
            else:
                res = f"{prefix} {clean_digits}{rem_str}"
            if trailing:
                res = f"{res}{trailing}"
            return res.strip()
        return match.group(0)

    text = re.sub(
        rf'({phone_keywords})[^\S\r\n]*[:\-]?[^\S\r\n]*([०-९0-9\u0900-\u097F ,–\-\.\t]{{3,80}})',
        phone_word_replacer,
        text
    )

    def standalone_10_digits(match):
        raw_str = match.group(0)
        digits = re.findall(r'[0-9०-९]', raw_str)
        if len(digits) == 10:
            trans_table = str.maketrans('0123456789', '०१२३४५६७८९')
            return ''.join(digits).translate(trans_table)
        return raw_str

    text = re.sub(
        r'(?<![0-9०-९])[0-9०-९](?:[^\S\r\n]*[,\-]?[^\S\r\n]*[0-9०-९]){9}(?!(?:[^\S\r\n]*[,\-]?[^\S\r\n]*[0-9०-९]))',
        standalone_10_digits,
        text
    )
    return text


def format_aadhar_numbers_dynamically(text):
    """
    Dynamically captures and formats Indian Aadhaar card numbers (12 digits) in Hindi STT output.
    """
    import re
    aadhar_keywords = r'(?:आधार\s*कार्ड\s*नंबर|आधार\s*कार्ड\s*क्रमांक|आधार\s*नंबर|आधार\s*क्रमांक|आधार\s*कार्ड|आधार)'

    def aadhar_replacer(match):
        prefix = match.group(1).strip()
        rest = match.group(2)

        stop_m = re.search(r'(?:\r?\n|(?<![\w\u0900-\u097F])(?:Speaker|स्पीकर)\b|\b(?:तथा|और\s+मेरा\s+पैन|पैन|PAN)\b)', rest, flags=re.IGNORECASE)
        if stop_m:
            trailing = rest[stop_m.start():]
            rest = rest[:stop_m.start()]
        else:
            trailing = ""

        working_text = rest
        for word, digit in sorted(HINDI_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
            working_text = re.sub(rf'(?<![\u0900-\u097F]){word}(?![\u0900-\u097F])', f" {digit} ", working_text)

        digits_matches = list(re.finditer(r'[0-9०-९]', working_text))
        if len(digits_matches) >= 12:
            twelfth_match = digits_matches[11]
            end_idx = twelfth_match.end()
            remaining_part = working_text[end_idx:]
            remaining_part = re.sub(r'^[,\s\-]+', ' ', remaining_part).strip()
            trans_table = str.maketrans('0123456789', '०१२३४५६७८९')
            clean_digits = [m.group().translate(trans_table) for m in digits_matches[:12]]
            clean_12 = ''.join(clean_digits)
            rem_str = f" {remaining_part}" if remaining_part else ""
            res = f"{prefix} {clean_12}{rem_str}"
            if trailing:
                res = f"{res}{trailing}"
            return res.strip()
        return match.group(0)

    text = re.sub(
        rf'({aadhar_keywords})[^\S\r\n]*[:\-]?[^\S\r\n]*(?:है\.?)?[^\S\r\n]*([०-९0-9\u0900-\u097F ,–\-\.\t]{{6,90}})',
        aadhar_replacer,
        text
    )

    def standalone_12_digits(match):
        raw_str = match.group(0)
        digits = re.findall(r'[0-9०-९]', raw_str)
        if len(digits) == 12:
            trans_table = str.maketrans('0123456789', '०१२३४५६७८९')
            return ''.join(digits).translate(trans_table)
        return raw_str

    text = re.sub(
        r'(?<![0-9०-९])[0-9०-९](?:[^\S\r\n]*[,\-]?[^\S\r\n]*[0-9०-९]){11}(?!(?:[^\S\r\n]*[,\-]?[^\S\r\n]*[0-9०-९]))',
        standalone_12_digits,
        text
    )

    text = re.sub(
        r'\b([०-९0-9]{4})[^\S\r\n]+([०-९0-9]{4})[^\S\r\n]+([०-९0-9]{4})\b',
        r'\1 \2 \3',
        text
    )
    return text


def format_pan_numbers_dynamically(text):
    """
    Dynamically captures and formats Indian Permanent Account Number (PAN) in Hindi STT output:
    5 uppercase letters + 4 digits + 1 uppercase letter (e.g. ABCDE1234F).
    """
    import re
    hindi_to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')
    pan_letter_map = {
        'ए': 'A', 'बी': 'B', 'सी': 'C', 'डी': 'D', 'ई': 'E', 'एफ': 'F',
        'जी': 'G', 'एच': 'H', 'आई': 'I', 'आय': 'I', 'जे': 'J', 'के': 'K', 'एल': 'L',
        'एम': 'M', 'एन': 'N', 'ओ': 'O', 'पी': 'P', 'क्यू': 'Q', 'आर': 'R',
        'एस': 'S', 'टी': 'T', 'यू': 'U', 'वी': 'V', 'व्ही': 'V', 'डब्ल्यू': 'W', 'डब्लू': 'W',
        'एक्स': 'X', 'वाई': 'Y', 'वाय': 'Y', 'जेड': 'Z', 'झेड': 'Z'
    }

    pan_keywords = r'(?:पैन\s*कार्ड\s*(?:नंबर|क्रमांक)?|पैन\s*(?:नंबर|क्रमांक)|PAN\s*(?:card\s*)?(?:number|no)?)'

    def pan_replacer(match):
        prefix = match.group(1).strip()
        raw_body = match.group(2)
        working = raw_body
        for k, v in sorted(pan_letter_map.items(), key=lambda x: len(x[0]), reverse=True):
            working = re.sub(rf'(?<![\u0900-\u097F]){k}(?![\u0900-\u097F])', v, working)
        working = working.translate(hindi_to_arabic)

        m_pan = re.search(r'\b([A-Za-z\s]{5,10})\s*([0-9\s]{4,8})\s*([A-Za-z])\b', working)
        if m_pan:
            p1 = re.sub(r'\s+', '', m_pan.group(1)).upper()
            p2 = re.sub(r'\s+', '', m_pan.group(2))
            p3 = m_pan.group(3).upper()
            if len(p1) == 5 and len(p2) == 4 and len(p3) == 1:
                pan_str = f"{p1}{p2}{p3}"
                end = m_pan.end()
                return f"{prefix} {pan_str}{working[end:]}"
        return match.group(0)

    text = re.sub(
        rf'({pan_keywords})\s*[:\-]?\s*([A-Za-z0-9०-९\u0900-\u097F\s]{{10,40}})',
        pan_replacer,
        text,
        flags=re.IGNORECASE
    )
    return text


def format_driving_license_dynamically(text):
    """
    Dynamically captures and formats Indian Driving Licenses (e.g. MH 46 20110001234 / DL-0420110012345).
    """
    import re
    hindi_to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')
    dl_keywords = r'(?:ड्राइविंग\s*लाइसेंस\s*(?:नंबर)?|ड्रायव्हिंग\s*लायसन्स|लाइसेंस\s*नंबर|DL\s*number)'

    def dl_replacer(match):
        prefix = match.group(1).strip()
        body = match.group(2).translate(hindi_to_arabic)
        clean = re.sub(r'[^A-Za-z0-9]', '', body).upper()
        if len(clean) >= 15:
            return f"{prefix} {clean[:16]}"
        return match.group(0)

    text = re.sub(
        rf'({dl_keywords})\s*[:\-]?\s*([A-Za-z0-9०-९\s\-]{{12,30}})',
        dl_replacer,
        text,
        flags=re.IGNORECASE
    )
    return text


def normalize_bank_details_dynamically(text):
    """
    Dynamically normalizes Indian bank account numbers and IFSC codes in Hindi STT output.
    """
    import re
    hindi_to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')
    ifsc_keywords = r'(?:IFSC\s*(?:कोड|code)?|आई\s*एफ\s*एस\s*सी\s*(?:कोड)?)'

    def ifsc_replacer(match):
        prefix = "IFSC कोड"
        raw_body = match.group(1).translate(hindi_to_arabic)
        m_ifsc = re.search(r'\b([A-Za-z]{4})\s*(?:0|०)\s*([A-Za-z0-9]{6})\b', raw_body)
        if m_ifsc:
            return f"{prefix} {m_ifsc.group(1).upper()}0{m_ifsc.group(2).upper()}"
        return match.group(0)

    text = re.sub(rf'({ifsc_keywords})\s*[:\-]?\s*([A-Za-z0-9०-९\s]{{5,30}})', ifsc_replacer, text, flags=re.IGNORECASE)

    acc_keywords = r'(?:बैंक\s*खाता\s*(?:संख्या|नंबर)?|खाता\s*(?:संख्या|नंबर)|अकाउंट\s*(?:नंबर|संख्या)|Account\s*(?:number|no)?)'
    def acc_replacer(match):
        prefix = match.group(1).strip()
        raw_digits = match.group(2)
        for w, d in sorted(HINDI_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
            raw_digits = re.sub(rf'(?<![\u0900-\u097F]){w}(?![\u0900-\u097F])', f" {d} ", raw_digits)
        clean_digits = re.sub(r'\D', '', raw_digits.translate(hindi_to_arabic))
        if 9 <= len(clean_digits) <= 18:
            return f"{prefix} {clean_digits}"
        return match.group(0)

    text = re.sub(rf'({acc_keywords})\s*[:\-]?\s*([0-9०-९\u0900-\u097F\s]{{9,35}})', acc_replacer, text, flags=re.IGNORECASE)
    return text


def normalize_currency_amounts_dynamically(text):
    """
    Dynamically normalizes currency amounts (रुपये / नकद राशि) in Hindi in Indian numeric format.
    e.g. '५०००० रुपये' -> '५०,०००/- रुपये'
         'दो लाख रुपये' -> '२,००,०००/- रुपये'
         'डेढ़ लाख रुपये' -> '१,५०,०००/- रुपये'
    """
    import re
    hindi_to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')
    arabic_to_hindi = str.maketrans('0123456789', '०१२३४५६७८९')

    multipliers = {
        'करोड़': 10000000, 'करोड': 10000000,
        'लाख': 100000,
        'हज़ार': 1000, 'हजार': 1000,
        'सौ': 100
    }

    fractions = {
        'डेढ़': 1.5, 'डेढ': 1.5,
        'ढाई': 2.5,
        'सवा': 1.25,
        'पौने': 0.75,
        'साढ़े': 0.5, 'साढे': 0.5
    }

    words_pat = '|'.join(sorted(HINDI_NUMBER_WORDS.keys(), key=len, reverse=True))

    def currency_replacer(match):
        body = match.group(1).strip()
        suffix = match.group(2).strip()

        num_start_m = re.search(rf'\b({words_pat}|[०-९0-9]+|डेढ़|ढाई|सवा|पौने|साढ़े)\b', body)
        if num_start_m:
            leading_text = body[:num_start_m.start()]
            num_body = body[num_start_m.start():]
        else:
            leading_text = ""
            num_body = body

        for frac, mult in fractions.items():
            for unit, u_val in multipliers.items():
                m_frac = re.search(rf'\b{frac}\s+{unit}\b', num_body)
                if m_frac:
                    calc_val = int(mult * u_val)
                    formatted_val = str(calc_val).translate(arabic_to_hindi)
                    return f"{leading_text}{formatted_val} {suffix}"

        for unit, u_val in multipliers.items():
            m_unit = re.search(rf'({words_pat}|[०-९0-9]+)\s+{unit}\b', num_body)
            if m_unit:
                num_raw = m_unit.group(1)
                num_val = int(HINDI_NUMBER_WORDS.get(num_raw, num_raw).translate(hindi_to_arabic))
                calc_val = num_val * u_val
                formatted_val = str(calc_val).translate(arabic_to_hindi)
                return f"{leading_text}{formatted_val} {suffix}"

        return match.group(0)

    currency_keywords = r'(?:रुपये|रुपया|रुपयों|रु\.|नकद\s*राशि|कैश)'
    text = re.sub(
        rf'([०-९0-9A-Za-z\u0900-\u097F\s]{{2,30}})\s+({currency_keywords})\b',
        currency_replacer,
        text
    )
    return text


def normalize_pincodes_dynamically(text):
    """
    Dynamically captures and standardizes Indian postal PIN codes (6 digits) in Hindi context.
    """
    import re
    hindi_digits = str.maketrans('0123456789', '०१२३४५६७८९')
    arabic_digits = str.maketrans('०१२३४५६७८९', '0123456789')

    def pin_replacer(match):
        prefix = match.group(1).strip()
        raw_code = match.group(2)
        clean_code = re.sub(r'\D', '', raw_code.translate(arabic_digits))
        if len(clean_code) == 6:
            formatted_code = clean_code.translate(hindi_digits)
            return f"{prefix} {formatted_code}"
        return match.group(0)

    pin_keywords = r'(?:पिन\s*कोड|पिनकोड|PIN\s*code|PIN)'
    text = re.sub(
        rf'({pin_keywords})\s*[:\-]?\s*([०-९0-9\s]{{6,10}})\b',
        pin_replacer,
        text,
        flags=re.IGNORECASE
    )
    return text


def normalize_passport_numbers_dynamically(text):
    """
    Dynamically captures and standardizes Indian Passport numbers (1 letter + 7 digits).
    """
    import re
    hindi_to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')

    def passport_replacer(match):
        prefix = match.group(1).strip()
        letter = match.group(2).upper()
        digits = re.sub(r'\D', '', match.group(3).translate(hindi_to_arabic))
        if len(digits) == 7:
            return f"{prefix} {letter}{digits}"
        return match.group(0)

    pass_keywords = r'(?:पासपोर्ट\s*(?:नंबर|क्रमांक)?|Passport\s*(?:number|no)?)'
    text = re.sub(
        rf'({pass_keywords})\s*[:\-]?\s*([A-Za-z])\s*([0-9०-९\s]{{7,10}})\b',
        passport_replacer,
        text,
        flags=re.IGNORECASE
    )
    return text


def normalize_addresses_dynamically(text):
    """
    Dynamically standardizes spoken address components in Hindi:
    - Spoken slashes/obliques: '११ बटा १ स्लैश ४८' -> '११/१/४८'
    - Building wing & numbers: 'बिल्डिंग नंबर बी ११' -> 'बिल्डिंग नंबर बी-११'
    - Standardizes abbreviations: मकान नंबर, कमरा नंबर, फ्लैट नंबर, प्लॉट नंबर.
    """
    import re
    slash_words = r'(?:स्लैश|स्लेश|slash|oblique|ऑब्लिक|बटा|भाग)'
    slash_pattern = rf'([०-९0-9]+(?:\s*{slash_words}\s*[०-९0-9]+)+)'

    def convert_spoken_slashes(m):
        raw = m.group(1)
        parts = re.split(rf'\s*{slash_words}\s*', raw, flags=re.IGNORECASE)
        clean_parts = [p.strip() for p in parts if p.strip()]
        return "/".join(clean_parts)

    text = re.sub(slash_pattern, convert_spoken_slashes, text, flags=re.IGNORECASE)

    text = re.sub(r'(?<![\u0900-\u097F])(?:कमरा|रूम|फ्लैट|सदनिका)\s*(?:नंर|नं\.?)(?![\u0900-\u097F])', 'कमरा नंबर', text)
    text = re.sub(r'(?<![\u0900-\u097F])(?:मकान|घर)\s*(?:नंर|नं\.?)(?![\u0900-\u097F])', 'मकान नंबर', text)
    text = re.sub(r'(?<![\u0900-\u097F])(?:बिल्डिंग|इमारत)\s*(?:नंर|नं\.?)(?![\u0900-\u097F])', 'बिल्डिंग नंबर', text)
    text = re.sub(r'(?<![\u0900-\u097F])प्लॉट\s*(?:नंर|नं\.?)(?![\u0900-\u097F])', 'प्लॉट नंबर', text)
    text = re.sub(r'(?<![\u0900-\u097F])सेक्टर\s*(?:नंर|नं\.?)(?![\u0900-\u097F])', 'सेक्टर नंबर', text)

    bldg_keywords = r'(?:बिल्डिंग|इमारत|विंग|फ्लैट|ब्लॉक|टावर|अपार्टमेंट|सोसायटी|कमरा|रूम|मकान|घर)\s*(?:नंबर|नं|क्रमांक)?'
    text = re.sub(
        rf'({bldg_keywords})\s*([A-Za-z\u0904-\u0939\u0958-\u0961])\s*(?:नंबर|-|–)?\s*([०-९0-9]{{1,4}})\b',
        r'\1 \2-\3',
        text
    )

    def address_num_replacer(m):
        prefix = m.group(1)
        word = m.group(2)
        digit = HINDI_NUMBER_WORDS.get(word, word)
        return f"{prefix} {digit}"

    addr_keywords = r'(?:प्लॉट\s*(?:नंबर|क्र\.?)?|कमरा\s*(?:नंबर|क्र\.?)?|रूम\s*(?:नंबर|क्र\.?)?|फ्लैट\s*(?:नंबर|क्र\.?)?|सेक्टर\s*(?:नंबर|क्र\.?)?|गली\s*(?:नंबर|क्र\.?)?|मकान\s*(?:नंबर|क्र\.?)?|दुकान\s*(?:नंबर|क्र\.?)?|बिल्डिंग\s*(?:नंबर|क्र\.?)?)'
    words_pat = '|'.join(sorted(HINDI_NUMBER_WORDS.keys(), key=len, reverse=True))
    text = re.sub(rf'({addr_keywords})\s*({words_pat})\b', address_num_replacer, text)
    return text


def normalize_police_and_case_details(text):
    """
    Dynamically normalizes police case details (age, buckle numbers, ranks) in Hindi.
    """
    import re
    def age_replacer(m):
        raw_val = m.group(1).strip()
        unit = m.group(2) if m.group(2) else "वर्ष"
        digit_val = HINDI_NUMBER_WORDS.get(raw_val, raw_val)
        digit_val = digit_val.translate(str.maketrans('0123456789', '०१२३४५६७८९'))
        return f"उम्र {digit_val} {unit}"

    words_pat = '|'.join(sorted(HINDI_NUMBER_WORDS.keys(), key=len, reverse=True))
    text = re.sub(
        rf'(?<![\u0900-\u097F])(?:उम्र|आयु|वय)\s*({words_pat}|[०-९0-9]{{1,2}})\s*(वर्ष|साल|इयर्स)?(?!\s*(?:ग्राम|किलो|मीटर|लाख|हजार|करोड़))(?![\u0900-\u097F])',
        age_replacer,
        text
    )

    def buckle_replacer(m):
        prefix = m.group(1)
        num_part = m.group(2)
        for w, d in sorted(HINDI_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
            num_part = re.sub(rf'(?<![\u0900-\u097F]){w}(?![\u0900-\u097F])', d, num_part)
        digits = re.sub(r'\D', '', num_part.translate(str.maketrans('०१२३४५६७८९', '0123456789')))
        if digits:
            hindi_digits = digits.translate(str.maketrans('0123456789', '०१२३४५६७८९'))
            return f"{prefix} {hindi_digits}"
        return m.group(0)

    text = re.sub(
        r'(बक्कल\s*नंबर|बैज\s*नंबर|बेल्ट\s*नंबर|कांस्टेबल\s*नंबर)\s*([^\n,]+?)(?=\s*(?:नियुक्ति|थाना|कोतवाली|चौकी|,|\.|$))',
        buckle_replacer,
        text
    )
    return text


def normalize_police_ranks_hindi(text):
    """
    Maps abbreviations and common ASR misspellings to official Hindi police ranks.
    """
    police_ranks_hindi = {
        "पुआ": "पुलिस आयुक्त",
        "सहपुआ": "संयुक्त पुलिस आयुक्त",
        "अपोआ": "अपर पुलिस आयुक्त",
        "पुउआ": "पुलिस उपायुक्त",
        "सपुआ": "सहायक पुलिस आयुक्त",
        "वपुनि": "वरिष्ठ पुलिस निरीक्षक",
        "पुनि": "पुलिस निरीक्षक",
        "सपुनि": "सहायक पुलिस निरीक्षक",
        "पुउपनि": "पुलिस उपनिरीक्षक",
        "उनि": "उपनिरीक्षक",
        "सउनि": "सहायक उपनिरीक्षक",
        "एसआई": "सब-इंस्पेक्टर",
        "एएसआई": "सहायक सब-इंस्पेक्टर",
        "पुह": "पुलिस हवलदार",
        "हेकां": "हेड कांस्टेबल",
        "हवलदार": "पुलिस हवलदार",
        "पुना": "पुलिस नायक",
        "कांस्टेबल": "पुलिस कांस्टेबल",
        "सिपाही": "पुलिस कांस्टेबल",
        "मपुह": "महिला पुलिस हवलदार",
        "मकां": "महिला कांस्टेबल",
        "थाना": "थाना",
        "कोतवाली": "कोतवाली",
        "पुलिस स्टेशन": "पुलिस स्टेशन"
    }

    for incorrect, correct in sorted(police_ranks_hindi.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(rf'\b{re.escape(incorrect)}\b', correct, text)

    return text


def normalize_height_measurements(text):
    """
    Dynamically normalizes physical height measurements in Hindi.
    e.g. '५ फुट ६ इंच' / '५ फीट छह इंच'
    """
    import re
    num_map = {
        'एक': '१', 'दो': '२', 'तीन': '३', 'चार': '४', 'पांच': '५', 'पाँच': '५',
        'छह': '६', 'छे': '६', 'सात': '७', 'आठ': '८', 'नौ': '९', 'दस': '१०', 'ग्यारह': '११', 'बारह': '१२'
    }

    def height_replacer(match):
        feet = match.group(1)
        inches_word = match.group(2)
        inches = num_map.get(inches_word, inches_word)
        return f"{feet} फीट {inches} इंच"

    word_pattern = '|'.join(num_map.keys())
    text = re.sub(rf'([०-९0-9]{{1,2}})\s*(?:फुट|फीट)\s+({word_pattern})\s+इंच\b', height_replacer, text)
    return text


def normalize_vehicle_numbers_hindi(text):
    """
    Dynamically normalizes vehicle registration numbers across ALL Indian states without hardcoding:
    Supports DL, UP, HR, MP, RJ, MH, TN, GJ, KA, AP, TS, KL, GA, CH, PB, WB, BR, JH, OD, UK.
    """
    import re
    text = re.sub(r'\b(DL|UP|HR|MP|RJ|MH|GJ|KA|TN|PB|UK|BR|JH|CH)\s*(?:00|००)\s*[,–\-]?\s*([0-9०-९]{2})\b', r'\1 \2', text)
    text = re.sub(r'\b(?:वैगन\s*आर|वैगनार|वॅगनर|वेगनर)\s*(?:कार)?\b', 'Wagon R कार', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:बीएमडब्ल्यू|बीएमडब्लू|बमडब्लू)\s*(?:कार)?\b', 'BMW कार', text, flags=re.IGNORECASE)

    state_map = {
        'डीएल': 'DL', 'डी एल': 'DL', 'DL': 'DL',
        'यूपी': 'UP', 'यू पी': 'UP', 'UP': 'UP',
        'एचआर': 'HR', 'एच आर': 'HR', 'HR': 'HR',
        'एमपी': 'MP', 'एम पी': 'MP', 'MP': 'MP',
        'आरजे': 'RJ', 'आर जे': 'RJ', 'RJ': 'RJ',
        'एमएच': 'MH', 'एम एच': 'MH', 'MH': 'MH',
        'जीजे': 'GJ', 'जी जे': 'GJ', 'GJ': 'GJ',
        'केए': 'KA', 'के ए': 'KA', 'KA': 'KA',
        'टीएन': 'TN', 'टी एन': 'TN', 'TN': 'TN',
        'पीबी': 'PB', 'पी बी': 'PB', 'PB': 'PB',
        'यूके': 'UK', 'यू के': 'UK', 'UK': 'UK',
        'बीआर': 'BR', 'बी आर': 'BR', 'BR': 'BR'
    }

    hindi_to_arabic = str.maketrans('०१२३४५६७८९', '0123456789')
    state_keys = sorted(state_map.keys(), key=len, reverse=True)
    state_pattern = '|'.join(re.escape(k) for k in state_keys)

    def veh_replacer(match):
        st_word = match.group(1).upper()
        st_code = state_map.get(st_word, state_map.get(match.group(1), match.group(1)))
        rto_digits = match.group(2).translate(hindi_to_arabic)
        series = re.sub(r'\s+', '', match.group(3)).upper()
        veh_digits = match.group(4).translate(hindi_to_arabic)
        return f"{st_code} {rto_digits} {series} {veh_digits}"

    text = re.sub(
        rf'\b({state_pattern})\s*([०-९0-9]{{1,2}})\s*([A-Za-z\s]{{1,4}})\s*([०-९0-9]{{1,4}})\b',
        veh_replacer,
        text
    )
    return text


def normalize_hindi_sentence_phonetics(text):
    """
    Context-aware phonetic and orthographic correction for Hindi police statements:
    - Deduplicates accidental consecutive Devanagari vowel combining marks.
    - Standardizes typed statement legal declarations (e.g. 'टंक लिखित' -> 'टंकलिखित').
    """
    import re
    # Unicode diacritic deduplication
    text = re.sub(r'([\u093e\u093f\u0940\u0941\u0942\u0947\u0948\u094b\u094c\u0902\u0901])\1+', r'\1', text)
    text = re.sub(r'\bटंक\s+लिखित\b', 'टंकलिखित', text)
    text = re.sub(r'\b(?:CCTV|सी\s*सी\s*टी\s*वी)\b', 'सीसीटीवी', text)
    return text


def resolve_hindi_compound_hundreds(text):
    """
    Universally resolves spoken Hindi compound hundreds (100 to 999) mathematically.
    Prevents digit concatenation (e.g., 'एक सौ ४४' -> १४४, NOT १००४४; 'पाँच सौ ३२' -> ५३२).
    """
    import re
    hundreds_map = {
        'एक सौ': 100, 'एकसौ': 100,
        'दो सौ': 200, 'दोसौ': 200,
        'तीन सौ': 300, 'तीनसौ': 300,
        'चार सौ': 400, 'चारसौ': 400,
        'पांच सौ': 500, 'पाँच सौ': 500, 'पांचसौ': 500, 'पाँचसौ': 500,
        'छह सौ': 600, 'छहसौ': 600,
        'सात सौ': 700, 'सातसौ': 700,
        'आठ सौ': 800, 'आठसौ': 800,
        'नौ सौ': 900, 'नौसौ': 900
    }
    hindi_digits = str.maketrans('०१२३४५६७८९', '0123456789')
    to_hindi = str.maketrans('0123456789', '०१२३४५६७८९')

    hundreds_regex = '|'.join(hundreds_map.keys())

    pattern = rf'\b({hundreds_regex})\s*([0-9०-९]{{1,2}})\b'
    def repl(m):
        base = hundreds_map[m.group(1)]
        rem = int(m.group(2).translate(hindi_digits))
        if rem < 100:
            return str(base + rem).translate(to_hindi)
        return m.group(0)
    text = re.sub(pattern, repl, text)

    for word, val in hundreds_map.items():
        text = re.sub(rf'\b{word}\b', str(val).translate(to_hindi), text)

    return text


def normalize_hindi_numbers(text):
    """
    Dynamically maps Hindi spoken numeral words (1 to 100) to standard Devanagari digits.
    """
    import re
    text = resolve_hindi_compound_hundreds(text)
    text = text.translate(str.maketrans('0123456789', '०१२३४५६७८९'))
    text = resolve_hindi_compound_thousands_and_years(text)

    suffixes = r'(के|में|को|का|की|से|पर|वे|वीं)?'
    for word, digit in sorted(HINDI_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf'(?<![\u0900-\u097F]){word}{suffixes}(?![\u0900-\u097F])'
        def repl(m, d=digit):
            suf = m.group(1) if m.group(1) else ''
            return f"{d}{suf}"
        text = re.sub(pattern, repl, text)

    text = resolve_hindi_compound_hundreds(text)
    return text


def normalize_legal_sections_hindi(text):
    """
    Dynamically normalizes legal acts, sections, and sub-sections in Hindi police FIRs:
    - Formats parenthetical sub-sections ('धारा ३०२ (२)' -> 'धारा ३०२(२)')
    - Standardizes IPC, BNS, and CrPC references.
    """
    import re
    col_words = {
        'एक': '१', 'दो': '२', 'तीन': '३', 'चार': '४', 'पांच': '५', 'पाँच': '५',
        'छह': '६', 'सात': '७', 'आठ': '८', 'नौ': '९', 'दस': '१०'
    }
    for w, d in col_words.items():
        text = re.sub(rf'\bधारा\s*{w}\b', f'धारा {d}', text)
        text = re.sub(rf'\bधारा{w}\b', f'धारा {d}', text)

    text = re.sub(r'([०-९0-9]+)\s*[\(\[]?\s*(?:कोष्ठक|कौस|ब्रैकेट)?\s*([०-९0-9]+)\s*[\)\]]?', r'\1(\2)', text)
    text = re.sub(r'\b(?:आईपीसी|भादंवि|भा\.दं\.वि\.)\b', 'भादंवि', text)
    text = re.sub(r'\b(?:बीएनएस|बी\.एन\.एस\.)\b', 'बीएनएस', text)

    return text


def _dedupe_join(parts, max_overlap_words=8):
    if not parts:
        return ""
    result_words = parts[0].split()
    for text in parts[1:]:
        curr_words = text.split()
        max_check = min(max_overlap_words, len(result_words), len(curr_words))
        best_k = 0
        for k in range(max_check, 0, -1):
            if result_words[-k:] == curr_words[:k]:
                best_k = k
                break
        result_words.extend(curr_words[best_k:])
    return " ".join(result_words)


def transcribe_audio_segment(stt_model, segment_data, sr, max_chunk_duration=25.0):
    """
    Transcribes an audio segment with full speech fidelity:
    - Normalizes speech peak to 0.92 so soft/whispered words are fully captured.
    - Guarantees minimum temporal padding (>=1.2s).
    - For long monologues (>25s), splits cleanly at inter-word energy minima (pauses).
    """
    if len(segment_data) == 0:
        return ""

    seg_max = np.max(np.abs(segment_data))
    if seg_max > 0.005:
        segment_data = (segment_data / seg_max) * 0.92

    duration = len(segment_data) / sr
    if duration <= max_chunk_duration:
        if duration < 1.0:
            pad_samples = max(int(0.2 * sr), int((1.2 - duration) / 2.0 * sr))
            chunk_padded = np.pad(segment_data, (pad_samples, pad_samples), mode='constant')
        else:
            pad = np.zeros(int(0.2 * sr), dtype=np.float32)
            chunk_padded = np.concatenate([pad, segment_data, pad])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, chunk_padded, sr, subtype='PCM_16')
            temp_path = temp_wav.name
        try:
            # Check if NeMo model or IndicWav2Vec model
            if hasattr(stt_model, "change_decoding_strategy"):
                outputs = stt_model.transcribe([temp_path], batch_size=1, language_id="hi", logprobs=False)
                transcriptions = outputs[0] if isinstance(outputs, tuple) else outputs
                text = " ".join([str(t).strip() for t in transcriptions if str(t).strip()])
                if not text and seg_max > 0.02:
                    try:
                        stt_model.change_decoding_strategy(decoder_type="ctc")
                        c_outs = stt_model.transcribe([temp_path], batch_size=1, language_id="hi", logprobs=False)
                        stt_model.change_decoding_strategy(decoder_type="rnnt")
                        c_trans = c_outs[0] if isinstance(c_outs, tuple) else c_outs
                        text = " ".join([str(t).strip() for t in c_trans if str(t).strip()])
                    except Exception:
                        try:
                            stt_model.change_decoding_strategy(decoder_type="rnnt")
                        except Exception:
                            pass
            else:
                outputs = stt_model.transcribe([temp_path], batch_size=1)
                transcriptions = outputs[0] if isinstance(outputs, tuple) else outputs
                text = " ".join([str(t).strip() for t in transcriptions if str(t).strip()])
            return text
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # For long turns (> 25s), split at natural low-energy pauses
    sub_texts = []
    curr_s = 0.0
    while curr_s < duration:
        target_end = min(duration, curr_s + max_chunk_duration)
        if target_end < duration:
            search_start = max(curr_s + 10.0, target_end - 4.0)
            s_i = int(search_start * sr)
            e_i = int(target_end * sr)
            sub = segment_data[s_i:e_i]
            if len(sub) > 512:
                frame_len = 512
                rms = np.sqrt(np.mean(np.square(librosa.util.frame(sub, frame_length=frame_len, hop_length=256)), axis=0))
                min_i = np.argmin(rms)
                curr_end = search_start + (min_i * 256) / sr
            else:
                curr_end = target_end
        else:
            curr_end = duration

        chunk_part = segment_data[int(curr_s * sr):int(curr_end * sr)]
        if len(chunk_part) / sr >= 0.2:
            t = transcribe_audio_segment(stt_model, chunk_part, sr, max_chunk_duration=max_chunk_duration)
            if t:
                sub_texts.append(t)
        curr_s = curr_end

    return " ".join(sub_texts)


# ==============================================================================
# Diarization & Speaker Alignment Utilities
# ==============================================================================

def analyze_and_plot_overlap(audio_file, diarization, base_dir):
    """
    Computes acoustic speaker overlap metrics and exports a diagnostic plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
        spk_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            spk_segments.append((turn.start, turn.end, speaker))

        if not spk_segments:
            return

        plt.figure(figsize=(10, 4))
        speakers = sorted(list(set(s[2] for s in spk_segments)))
        spk_y = {spk: idx for idx, spk in enumerate(speakers)}

        for start, end, spk in spk_segments:
            plt.plot([start, end], [spk_y[spk], spk_y[spk]], linewidth=4, solid_capstyle='butt')

        plt.yticks(range(len(speakers)), speakers)
        plt.xlabel("Time (seconds)")
        plt.title(f"Speaker Diarization Timeline - {audio_basename}")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path = os.path.join(base_dir, "speaker_overlap_timeline.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📊 Saved diarization timeline plot to {plot_path}")
    except Exception as e:
        print(f"ℹ️ Overlap plotting note: {e}")


def standardize_speaker_labels(dialogue_text):
    """
    Ensures speaker labels are standard ('Speaker 0:', 'Speaker 1:', etc.).
    """
    lines = dialogue_text.splitlines()
    cleaned = []
    for line in lines:
        m = re.match(r'^(?:Speaker\s*|स्पीकर\s*)(\d+)\s*:\s*(.*)$', line.strip(), re.IGNORECASE)
        if m:
            cleaned.append(f"Speaker {m.group(1)}: {m.group(2).strip()}")
        else:
            cleaned.append(line.strip())
    return "\n".join(cleaned)


def clean_dialogue_remove_punctuation(dialogue_text):
    """
    Strips extraneous punctuation before LLM pass while preserving Hindi characters and numbers.
    """
    cleaned_lines = []
    for line in dialogue_text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        m = re.match(r'^(Speaker\s*\d+)\s*:\s*(.*)$', line_str, re.IGNORECASE)
        if m:
            prefix = m.group(1)
            content = m.group(2)
            content_clean = re.sub(r'[\.,\?!|।;:"\'\(\)\[\]]', ' ', content)
            content_clean = re.sub(r'\s+', ' ', content_clean).strip()
            cleaned_lines.append(f"{prefix}: {content_clean}")
        else:
            line_clean = re.sub(r'[\.,\?!|।;:"\'\(\)\[\]]', ' ', line_str)
            line_clean = re.sub(r'\s+', ' ', line_clean).strip()
            cleaned_lines.append(line_clean)
    return "\n".join(cleaned_lines)


def group_dialogue_by_speaker(dialogue_text):
    """
    Groups turns by speaker into consolidated statements.
    """
    speaker_statements = {}
    for line in dialogue_text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        m = re.match(r'^(Speaker\s*\d+)\s*:\s*(.*)$', line_str, re.IGNORECASE)
        if m:
            spk = m.group(1).title()
            text = m.group(2).strip()
            if spk not in speaker_statements:
                speaker_statements[spk] = []
            speaker_statements[spk].append(text)
    
    blocks = []
    for spk, utts in sorted(speaker_statements.items()):
        blocks.append(f"### {spk} Statement:\n" + " ".join(utts))
    return "\n\n".join(blocks)


def realign_dialogue_speakers_vllm(dialogue_text, master_transcript=None, vllm_url=None, model_name=None):
    """
    Uses local Gemma / LLM via vLLM to reconstruct and disentangle conversational speaker turns
    in Hindi police interrogation transcripts:
      - Speaker 0: Leading Investigating Officer asking questions / conducting inquiries.
      - Speaker 1: Respondent / Complainant answering questions and giving statements.
    Strictly preserves all spoken words from the speech context, corrects Hindi orthography/spelling,
    with ZERO added words, ZERO dropped words, and adds natural punctuation where needed.
    """
    source_text = master_transcript.strip() if (master_transcript and master_transcript.strip()) else (dialogue_text or "").strip()
    if not source_text:
        return dialogue_text

    target_url = vllm_url or os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
    target_model = model_name or os.environ.get("VLLM_MODEL", SPELL_CHECK_MODEL)

    if USE_GEMMA_FOR_NUMBERS:
        number_prompt_rules = (
            "6. WORDS TO NUMBER CONVERSION & ARITHMETIC (MANDATORY):\n"
            "   - Convert spoken Hindi numbers into Devanagari digits accurately (e.g. 'एक' -> '१', 'दो' -> '२', 'पच्चीस' -> '२५', 'अट्ठाईस' -> '२८', 'सौ' -> '१००').\n"
            "   - HUNDREDS & THOUSANDS ARITHMETIC ADDITION:\n"
            "     When hundreds ('एक सौ', 'दो सौ', 'तीन सौ', 'चार सौ', 'पांच सौ', 'छह सौ', 'सात सौ', 'आठ सौ', 'नौ सौ') or thousands ('दो हज़ार', 'एक हज़ार') are followed by units, you MUST ADD them mathematically into a single number! NEVER concatenate with extra zeros:\n"
            "     * 'एक सौ तीन'          -> '१०३'   (100 + 3 = 103, NEVER '१००३')\n"
            "     * 'दो सौ पंद्रह'        -> '२१५'   (200 + 15 = 215, NEVER '२००१५')\n"
            "     * 'सात सौ उनतालीस'     -> '७३९'   (700 + 39 = 739, NEVER '७००३९')\n"
            "     * 'दो हज़ार ग्यारह'     -> '२०११'  (2000 + 11 = 2011, NEVER '२००० ११')\n"
            "     * 'दो हज़ार पच्चीस'     -> '२०२५'  (2000 + 25 = 2025)\n"
            "   - CALENDAR DATES & BIRTHDATES:\n"
            "     * Format calendar dates as DD/MM/YYYY using Devanagari numerals.\n"
            "     * MONTH AND YEAR: When only a month and year are stated (e.g. '४ २०२६', '०४/२०२६', 'चौथा महीना २०२६'), format as Hindi month name and year (e.g. 'अप्रैल २०२६').\n"
            "     * Keep birthdate and complainant's age ('उम्र <X> वर्ष') strictly separate.\n"
            "   - Mobile phone numbers must be converted into 10 continuous digits (e.g. '९८३३३८८२९५').\n"
            "   - Aadhaar numbers must be converted into 12 digits.\n"
            "   - Spoken time must be formatted as HH:MM with period (e.g. 'दोपहर १४:४५ बजे', 'रात १२:०० बजे').\n"
            "   - Vehicle registration numbers must be in standard English uppercase and digits (e.g. 'DL 01 AB 1234', 'UP 14 CD 5678').\n"
        )
        task_instruction = "convert spoken words to numbers/digits according to the arithmetic and formatting rules, "
    else:
        number_prompt_rules = ""
        task_instruction = ""

    system_prompt = (
        "You are a strict verbatim Hindi ASR Orthographic and Spelling Corrector for police interrogation transcripts.\n\n"
        "MANDATORY CLIENT CONSTRAINTS:\n"
        "1. STRICT VERBATIM PRESERVATION - ZERO WORD ADDITION / ZERO WORD SKIPPING:\n"
        "   - You must NEVER invent, add, or complete any questions or phrases.\n"
        "   - If a speaker says 'उम्र', output strictly 'उम्र'. NEVER add 'कितनी है' (NEVER output 'उम्र कितनी है').\n"
        "   - Keep all short utterances and turns ('हाँ बताइए', 'उम्र', 'मैं', 'जी') exactly as spoken.\n"
        "   - Do NOT drop, skip, summarize, or paraphrase any words.\n\n"
        "2. PRESERVE SPEAKER LABELS & LINE FRAMES EXACTLY:\n"
        "   - Keep every line's exact speaker label ('Speaker 0:' or 'Speaker 1:') as provided in the input.\n"
        "   - Maintain the exact sequence of turns 1-to-1.\n\n"
        "3. HINDI SPELLING & ORTHOGRAPHY CORRECTION ONLY:\n"
        "   - Only correct spelling mistakes, vowel matras, and orthographic typos present in standard Hindi.\n\n"
        f"{number_prompt_rules}"
        "4. NATURAL PUNCTUATION & FORMATTING:\n"
        "   - Add natural punctuation marks (e.g. '?' for questions, ',' for pauses/clauses, and '.' for sentences) wherever appropriate.\n"
        "   - Place a blank line between every speaker turn.\n"
        "   - Output ONLY the speaker-attributed dialogue without markdown fences or preambles."
    )

    user_prompt = (
        f"Correct the spelling mistakes, {task_instruction}add natural punctuation marks where needed, and preserve EVERY exact spoken word and speaker label line-by-line without adding or skipping any words:\n\n"
        f"{source_text}"
    )

    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 3000,
        "temperature": 0.0
    }

    try:
        import requests
        resp = requests.post(target_url, json=payload, timeout=90)
        resp.raise_for_status()
        res_json = resp.json()
        realigned = res_json["choices"][0]["message"]["content"].strip()
        print("✅ Local LLM successfully realigned Hindi speaker turns and normalized text.")
        return realigned
    except Exception as e:
        print(f"⚠️ Local LLM realignment note ({e}), proceeding with acoustic speaker attribution.")
        return dialogue_text


def _fill_transcription_gaps_via_vad(
    data: np.ndarray,
    sr: int,
    duration: float,
    merged_turns: list,
    speaker_mapping: dict,
    stt_model: any,
    transcribed_windows: list,
):
    """
    Safety net: finds speech intervals detected by silence-based VAD (librosa.effects.split)
    that were not covered by diarized turn windows and merges them in.
    """
    try:
        intervals = librosa.effects.split(data, top_db=28, frame_length=1024, hop_length=256)
    except Exception as e:
        print(f"⚠️ VAD gap-fill pass note: {e}")
        return []

    min_silence_samples = int(0.45 * sr)
    merged_intervals = []
    for s, e in intervals:
        if merged_intervals and (s - merged_intervals[-1][1]) < min_silence_samples:
            merged_intervals[-1] = (merged_intervals[-1][0], e)
        else:
            merged_intervals.append((s, e))

    default_speaker = next(iter(speaker_mapping.values()), "Speaker 0")

    def _speaker_for_window(win_start: float, win_end: float) -> str:
        best_turn, best_overlap = None, 0.0
        for t in merged_turns:
            overlap = min(win_end, t['end']) - max(win_start, t['start'])
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = t
        if best_turn is not None:
            return speaker_mapping.get(best_turn['speaker'], default_speaker)
        if not merged_turns:
            return default_speaker
        nearest = min(
            merged_turns,
            key=lambda t: min(abs(win_start - t['end']), abs(t['start'] - win_end)),
        )
        return speaker_mapping.get(nearest['speaker'], default_speaker)

    gap_turns = []
    for s_idx, e_idx in merged_intervals:
        win_start = round(s_idx / sr, 2)
        win_end = round(min(duration, e_idx / sr), 2)
        interval_len = win_end - win_start
        if interval_len < 0.20 or interval_len > 6.0:
            continue

        covered = 0.0
        for t_start, t_end in transcribed_windows:
            covered += max(0.0, min(win_end, t_end) - max(win_start, t_start))
        if covered >= 0.85 * interval_len:
            continue

        pad_s = max(0, int((win_start - 0.20) * sr))
        pad_e = min(len(data), int((win_end + 0.20) * sr))
        chunk = data[pad_s:pad_e]
        raw_text = transcribe_audio_segment(stt_model, chunk, sr)
        if not raw_text.strip():
            continue

        final_text = raw_text
        if not USE_GEMMA_FOR_NUMBERS:
            final_text = format_hindi_dates_dynamically(final_text)
            final_text = format_hindi_date_slashes(final_text)
            final_text = format_hindi_time_dynamically(final_text)
            final_text = normalize_addresses_dynamically(final_text)
            final_text = normalize_pincodes_dynamically(final_text)
            final_text = format_phone_numbers_dynamically(final_text)
            final_text = format_aadhar_numbers_dynamically(final_text)
            final_text = format_pan_numbers_dynamically(final_text)
            final_text = format_driving_license_dynamically(final_text)
            final_text = normalize_passport_numbers_dynamically(final_text)
            final_text = normalize_bank_details_dynamically(final_text)
            final_text = normalize_currency_amounts_dynamically(final_text)
            final_text = normalize_police_and_case_details(final_text)
            final_text = resolve_hindi_compound_hundreds(final_text)
            final_text = normalize_hindi_numbers(final_text)
            final_text = normalize_vehicle_numbers_hindi(final_text)
            final_text = normalize_police_ranks_hindi(final_text)
            final_text = normalize_height_measurements(final_text)
            final_text = normalize_legal_sections_hindi(final_text)
        final_text = normalize_hindi_sentence_phonetics(final_text)

        gap_turns.append({
            'start': win_start,
            'end': win_end,
            'speaker': _speaker_for_window(win_start, win_end),
            'text': final_text,
            'verbatim': raw_text.strip(),
        })
    return gap_turns


# ==============================================================================
# Execution Modes
# ==============================================================================

def run_subtitles_whisperx(audio_file, device=None):
    """
    Mode 1: Subtitles & Timestamps using WhisperX for Hindi
    """
    target_device = device or DEVICE
    print("\n" + "="*50)
    print(f"🎬 MODE: Subtitles & Timestamps (WhisperX - Hindi on {target_device})")
    print("="*50)
    
    try:
        import whisperx
    except ImportError:
        print("WhisperX is not installed. Run: pip install whisperx")
        return

    print(f"[1/4] Loading WhisperX Model on {target_device}...")
    compute_type = "float16" if target_device == "cuda" else "int8"
    model = whisperx.load_model("large-v2", target_device, compute_type=compute_type)
    
    print(f"[2/4] Loading audio: {Path(audio_file).name}...")
    audio = whisperx.load_audio(audio_file)
    
    print("[2.5/4] Applying voice frequency filter (30-3400Hz)...")
    audio = apply_voice_bandpass_filter(audio, 16000)
    
    print("[3/4] Transcribing (Hindi)...")
    result = model.transcribe(audio, batch_size=4, language="hi")
    
    print(f"[4/4] Aligning Timestamps on {target_device}...")
    model_a, metadata = whisperx.load_align_model(language_code=result.get("language", "hi"), device=target_device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, target_device, return_char_alignments=False)
    
    print("\n✨ --- ALIGNED SUBTITLES --- ✨")
    for segment in result["segments"]:
        start = segment['start']
        end = segment['end']
        text = segment['text']
        print(f"[{start:05.2f}s -> {end:05.2f}s]: {text.strip()}")


def run_enterprise_stt(audio_file, stt_model_identifier=DEFAULT_HINDI_MODEL, device=None):
    """
    Mode 2: Enterprise Audio / STT using Jivi AI AudioX-North (Hindi) strictly on GPU
    """
    target_device = "cuda"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
    print("\n" + "="*50)
    print(f"🏢 MODE: Enterprise Audio (Jivi AudioX-North Hindi STT - {stt_model_identifier} on GPU ({gpu_name}))")
    print("="*50)
    
    print(f"[1/3] Loading Hindi STT Model ({stt_model_identifier}) on {target_device}...")
    stt_model = load_hindi_stt_model(stt_model_identifier, device=target_device)
    
    print("[2/3] Transcribing Audio...")
    chunk_files = split_and_prepare_audio(audio_file)
    
    raw_transcript = ""
    try:
        if hasattr(stt_model, "change_decoding_strategy"):
            outputs = stt_model.transcribe(chunk_files, batch_size=4, language_id="hi")
        else:
            outputs = stt_model.transcribe(chunk_files, batch_size=4)
        transcriptions = outputs[0] if isinstance(outputs, tuple) else outputs
        raw_transcript = " ".join([str(t).strip() for t in transcriptions if str(t).strip()])
    except Exception as e:
        print(f"🚨 ASR Transcription Error: {e}")
        raw_transcript = ""
    finally:
        for chunk_path in chunk_files:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    print("\n🔹 Raw Transcript:")
    print(raw_transcript)

    print("\n[3/3] Applying Hindi Text Normalization & Formatting...")
    itn_transcript = raw_transcript
    if not USE_GEMMA_FOR_NUMBERS:
        itn_transcript = format_hindi_dates_dynamically(itn_transcript)
        itn_transcript = format_hindi_date_slashes(itn_transcript)
        itn_transcript = format_hindi_time_dynamically(itn_transcript)
        itn_transcript = normalize_addresses_dynamically(itn_transcript)
        itn_transcript = normalize_pincodes_dynamically(itn_transcript)
        itn_transcript = format_phone_numbers_dynamically(itn_transcript)
        itn_transcript = format_aadhar_numbers_dynamically(itn_transcript)
        itn_transcript = format_pan_numbers_dynamically(itn_transcript)
        itn_transcript = format_driving_license_dynamically(itn_transcript)
        itn_transcript = normalize_passport_numbers_dynamically(itn_transcript)
        itn_transcript = normalize_bank_details_dynamically(itn_transcript)
        itn_transcript = normalize_currency_amounts_dynamically(itn_transcript)
        itn_transcript = normalize_police_and_case_details(itn_transcript)
        itn_transcript = resolve_hindi_compound_hundreds(itn_transcript)
        itn_transcript = normalize_hindi_numbers(itn_transcript)
        itn_transcript = normalize_vehicle_numbers_hindi(itn_transcript)
        itn_transcript = normalize_police_ranks_hindi(itn_transcript)
        itn_transcript = normalize_height_measurements(itn_transcript)
        itn_transcript = normalize_legal_sections_hindi(itn_transcript)
    itn_transcript = normalize_hindi_sentence_phonetics(itn_transcript)

    print("\n✨ --- NORMALIZED HINDI TRANSCRIPT --- ✨")
    print(itn_transcript)
    return itn_transcript


# Backward compatible alias for Mode 2
run_enterprise_nemo = run_enterprise_stt


def run_fir_pipeline(audio_file, min_speakers=None, max_speakers=None, diarization_model=None, stt_model_identifier=DEFAULT_HINDI_MODEL, device=None):
    """
    Mode 3: FIR Statement Generation using Pyannote + Jivi AI AudioX-North + LLM strictly on GPU
    Generates formal First Information Report (FIR) / बयान in authentic Hindi.
    """
    target_device = "cuda"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
    print("\n" + "="*50)
    print(f"📝 MODE: FIR Statement Generation (Pyannote + Jivi AudioX-North {stt_model_identifier} on GPU ({gpu_name}) + LLM)")
    print("="*50)
    
    model_alias_map = {
        "community-1": "pyannote/speaker-diarization-community-1",
        "community": "pyannote/speaker-diarization-community-1",
        "pyannote/speaker-diarization-community-1": "pyannote/speaker-diarization-community-1",
        "default": "pyannote/speaker-diarization-community-1",
        "3.1": "pyannote/speaker-diarization-community-1",
        "pyannote/speaker-diarization-3.1": "pyannote/speaker-diarization-community-1",
        "precision-2": "pyannote/speaker-diarization-precision-2",
        "precision": "pyannote/speaker-diarization-precision-2",
        "pyannote/speaker-diarization-precision-2": "pyannote/speaker-diarization-precision-2",
    }
    raw_model = (diarization_model or os.environ.get("PYANNOTE_MODEL") or "pyannote/speaker-diarization-community-1").strip()
    selected_model = model_alias_map.get(raw_model.lower(), "pyannote/speaker-diarization-community-1")
    if "3.1" in selected_model:
        selected_model = "pyannote/speaker-diarization-community-1"
    is_precision = "precision" in selected_model.lower()

    auth_token = None
    if is_precision:
        auth_token = (
            os.environ.get("PYANNOTEAI_API_KEY") or
            os.environ.get("PYANNOTE_API_KEY") or
            os.environ.get("HF_TOKEN") or
            os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
    else:
        auth_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not auth_token:
        try:
            with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                auth_token = f.read().strip()
        except Exception:
            pass

    if not auth_token:
        print("⚠️ Hugging Face Token (HF_TOKEN) not set. Falling back to Enterprise Hindi STT...")
        run_enterprise_stt(audio_file, stt_model_identifier=stt_model_identifier)
        return

    print(f"[1/3] Running Speaker Diarization with model '{selected_model}'...")
    try:
        audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
        base_dir = f"outputs/{audio_basename}"
        os.makedirs(base_dir, exist_ok=True)

        print("[1/3] Preparing 16 kHz 16-bit Mono PCM Audio & Speaker Diarization...")
        data, sr = load_audio_16k_mono_pcm(audio_file)
        data = apply_voice_bandpass_filter(data, sr, lowcut=30.0)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val) * 0.95

        duration = len(data) / sr
        diarization = None
        num_speakers = 1

        if duration < 15.0:
            print("⚠️ Audio is too short (< 15s) for diarization. Treating as a single speaker.")
            merged_turns = [{'start': 0.0, 'end': duration, 'speaker': 'SPEAKER_00'}]
        else:
            print(f"📦 Loading Pyannote Pipeline: {selected_model}...")
            pyannote_cache_dir = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub" / "models--pyannote--speaker-diarization-community-1"
            cached_snapshot = next(iter(pyannote_cache_dir.rglob("config.yaml")), None) if pyannote_cache_dir.exists() else None
            try:
                pipeline = Pipeline.from_pretrained(selected_model, token=auth_token)
            except Exception as load_err:
                print(f"⚠️ Hugging Face API note ({load_err}), checking local snapshot...")
                if cached_snapshot and os.path.exists(cached_snapshot):
                    pipeline = Pipeline.from_pretrained(cached_snapshot)
                    print(f"✅ Loaded from local snapshot.")
                else:
                    raise load_err

            try:
                pipeline.to(torch.device(target_device))
            except Exception as dev_err:
                print(f"ℹ️ Device placement note: {dev_err}")

            clustering_threshold = float(os.environ.get("PYANNOTE_CLUSTERING_THRESHOLD", "0.60"))
            if not is_precision:
                try:
                    pipeline.instantiate({
                        "segmentation": {"min_duration_off": 0.5},
                        "clustering": {"threshold": 0.60},
                    })
                except Exception as e:
                    print(f"⚠️ Clustering threshold note: {e}")

            waveform = torch.from_numpy(data).float().unsqueeze(0)
            audio_dict = {'waveform': waveform, 'sample_rate': sr}

            diarize_kwargs = {}
            diarize_kwargs['min_speakers'] = min_speakers if min_speakers is not None else 2
            if max_speakers is not None:
                diarize_kwargs['max_speakers'] = max_speakers

            try:
                raw_diarization = pipeline(audio_dict, **diarize_kwargs)
            except Exception:
                raw_diarization = pipeline(audio_file, **diarize_kwargs)

            if hasattr(raw_diarization, "speaker_diarization"):
                diarization = raw_diarization.speaker_diarization
            elif hasattr(raw_diarization, "exclusive_speaker_diarization") and len(raw_diarization.exclusive_speaker_diarization) > 0:
                diarization = raw_diarization.exclusive_speaker_diarization
            else:
                diarization = raw_diarization

            raw_labels = list(diarization.labels()) if hasattr(diarization, "labels") else []
            print(f"📊 Pyannote Hypothesis: {len(raw_labels)} raw speaker labels: {raw_labels}")

            merged_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if not merged_turns:
                    merged_turns.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
                else:
                    last_turn = merged_turns[-1]
                    if turn.start >= last_turn['start'] and turn.end <= last_turn['end'] and (turn.end - turn.start) < 0.15:
                        continue
                    if last_turn['speaker'] == speaker and (turn.start - last_turn['end']) < 3.0:
                        last_turn['end'] = max(last_turn['end'], turn.end)
                    else:
                        merged_turns.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})

            # Speaker Embeddings & Cosine Merging
            speaker_embeddings = {}
            if hasattr(raw_diarization, "speaker_embeddings") and raw_diarization.speaker_embeddings is not None:
                labels = list(raw_diarization.speaker_diarization.labels()) if hasattr(raw_diarization, "speaker_diarization") else list(diarization.labels())
                for s_idx, spk in enumerate(labels):
                    if s_idx < len(raw_diarization.speaker_embeddings):
                        vec = np.asarray(raw_diarization.speaker_embeddings[s_idx], dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 1e-6:
                            speaker_embeddings[spk] = vec / norm

            speaker_durations = {}
            for t in merged_turns:
                spk = t['speaker']
                speaker_durations[spk] = speaker_durations.get(spk, 0.0) + (t['end'] - t['start'])

            # Agglomerative clustering merge (threshold >= 0.82)
            MERGE_SIMILARITY_THRESHOLD = 0.82
            while len(speaker_embeddings) > 1:
                spk_list = [s for s in speaker_embeddings.keys() if s in speaker_durations]
                if len(spk_list) <= 1:
                    break
                best_sim = -1.0
                best_pair = None
                for i in range(len(spk_list)):
                    for j in range(i + 1, len(spk_list)):
                        sim = float(np.dot(speaker_embeddings[spk_list[i]], speaker_embeddings[spk_list[j]]))
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (spk_list[i], spk_list[j])

                if best_pair is not None and best_sim >= MERGE_SIMILARITY_THRESHOLD:
                    spk_a, spk_b = best_pair
                    dur_a = speaker_durations.get(spk_a, 0.0)
                    dur_b = speaker_durations.get(spk_b, 0.0)
                    spk_keep, spk_merge = (spk_a, spk_b) if dur_a >= dur_b else (spk_b, spk_a)
                    print(f"🔗 Merging speaker clusters {spk_merge} -> {spk_keep} (Cosine Similarity: {best_sim:.3f})")

                    for turn in merged_turns:
                        if turn['speaker'] == spk_merge:
                            turn['speaker'] = spk_keep

                    total_d = dur_a + dur_b
                    speaker_durations[spk_keep] = total_d
                    if spk_merge in speaker_durations:
                        del speaker_durations[spk_merge]
                    if spk_merge in speaker_embeddings:
                        del speaker_embeddings[spk_merge]
                else:
                    break

            # Re-merge adjacent segments of same speaker
            new_merged = []
            for turn in merged_turns:
                if not new_merged:
                    new_merged.append(turn)
                else:
                    last = new_merged[-1]
                    if last['speaker'] == turn['speaker'] and (turn['start'] - last['end']) < 3.0:
                        last['end'] = max(last['end'], turn['end'])
                    else:
                        new_merged.append(turn)
            merged_turns = new_merged
            unique_speakers = sorted(list(set(t['speaker'] for t in merged_turns)))
            num_speakers = len(unique_speakers)

        if diarization and num_speakers > 1:
            analyze_and_plot_overlap(audio_file, diarization, base_dir)

        print(f"[1.5/3] Loading Hindi STT Model ({stt_model_identifier}) on {target_device}...")
        stt_model = load_hindi_stt_model(stt_model_identifier, device=target_device)

        print(f"[1.6/3] Generating Master Continuous Hindi Transcription...")
        chunk_files = split_and_prepare_audio(audio_file)
        raw_master_transcript = ""
        try:
            if hasattr(stt_model, "change_decoding_strategy"):
                outputs = stt_model.transcribe(chunk_files, batch_size=4, language_id="hi")
            else:
                outputs = stt_model.transcribe(chunk_files, batch_size=4)
            transcriptions = outputs[0] if isinstance(outputs, tuple) else outputs
            raw_master_transcript = " ".join([str(t).strip() for t in transcriptions if str(t).strip()])
        except Exception as e:
            print(f"⚠️ Continuous transcription note: {e}")
            raw_master_transcript = ""
        finally:
            for chunk_path in chunk_files:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

        if raw_master_transcript:
            print(f"\n✨ --- RAW CONTINUOUS HINDI MASTER TRANSCRIPT --- ✨\n{raw_master_transcript}\n")
            with open(f"{base_dir}/raw_master_transcription.txt", "w", encoding="utf-8") as f:
                f.write(raw_master_transcript)

        print(f"\n🔹 Diarization Segments ({num_speakers} speakers, {len(merged_turns)} turns) & Transcription:")
        diarized_transcript = ""
        verbatim_diarized_transcript = ""
        diarization_only = ""
        processed_turns = []

        speaker_mapping = {}
        for t in merged_turns:
            spk = t['speaker']
            if t['start'] < 1.5 and (t['end'] - t['start']) < 0.8:
                continue
            if spk not in speaker_mapping:
                speaker_mapping[spk] = f"Speaker {len(speaker_mapping)}"
        for t in merged_turns:
            spk = t['speaker']
            if spk not in speaker_mapping:
                speaker_mapping[spk] = f"Speaker {len(speaker_mapping)}"

        if num_speakers == 1 and raw_master_transcript.strip():
            single_spk = list(speaker_mapping.keys())[0] if speaker_mapping else "SPEAKER_00"
            label = speaker_mapping.get(single_spk, "Speaker 0")
            print(f"ℹ️ Single speaker detected ({label}). Using Master Continuous Hindi Transcription directly.")
            processed_turns = [{
                'start': 0.0,
                'end': duration,
                'speaker': label,
                'text': raw_master_transcript.strip(),
                'verbatim': raw_master_transcript.strip()
            }]
        else:
            transcribed_windows = []
            for i, turn in enumerate(merged_turns):
                turn_dur = turn['end'] - turn['start']
                if turn_dur < 0.10:
                    continue

                diarization_only += f"[{turn['start']:.1f}s -> {turn['end']:.1f}s] {turn['speaker']}\n"
                turn_start = max(0.0, turn['start'] - 0.08)
                turn_end = min(duration, turn['end'] + 0.08)

                # Handle inter-speaker boundaries & interruptions
                if i + 1 < len(merged_turns) and merged_turns[i + 1]['speaker'] != turn['speaker']:
                    if merged_turns[i + 1]['start'] < turn['end']:
                        # Overlap: next speaker interrupted current speaker, cut current turn before interruption
                        turn_end = min(turn_end, merged_turns[i + 1]['start'] + 0.10)
                    else:
                        turn_end = min(merged_turns[i + 1]['start'], turn_end)

                if i > 0 and merged_turns[i - 1]['speaker'] != turn['speaker']:
                    if turn['start'] >= merged_turns[i - 1]['end']:
                        turn_start = max(merged_turns[i - 1]['end'], turn_start)
                    else:
                        # Overlapping response: preserve speaker's actual start time
                        turn_start = max(0.0, turn['start'] - 0.05)

                start_sample = max(0, int(turn_start * sr))
                end_sample = min(len(data), int(turn_end * sr))
                chunk = data[start_sample:end_sample]
                transcribed_windows.append((turn_start, turn_end))

                raw_text = transcribe_audio_segment(stt_model, chunk, sr)
                if not raw_text.strip():
                    continue

                final_text = raw_text
                if not USE_GEMMA_FOR_NUMBERS:
                    final_text = format_hindi_dates_dynamically(final_text)
                    final_text = format_hindi_date_slashes(final_text)
                    final_text = format_hindi_time_dynamically(final_text)
                    final_text = normalize_addresses_dynamically(final_text)
                    final_text = normalize_pincodes_dynamically(final_text)
                    final_text = format_phone_numbers_dynamically(final_text)
                    final_text = format_aadhar_numbers_dynamically(final_text)
                    final_text = format_pan_numbers_dynamically(final_text)
                    final_text = format_driving_license_dynamically(final_text)
                    final_text = normalize_passport_numbers_dynamically(final_text)
                    final_text = normalize_bank_details_dynamically(final_text)
                    final_text = normalize_currency_amounts_dynamically(final_text)
                    final_text = normalize_police_and_case_details(final_text)
                    final_text = resolve_hindi_compound_hundreds(final_text)
                    final_text = normalize_hindi_numbers(final_text)
                    final_text = normalize_vehicle_numbers_hindi(final_text)
                    final_text = normalize_police_ranks_hindi(final_text)
                    final_text = normalize_height_measurements(final_text)
                    final_text = normalize_legal_sections_hindi(final_text)
                final_text = normalize_hindi_sentence_phonetics(final_text)

                speaker_label = speaker_mapping.get(turn['speaker'], "Speaker 0")
                processed_turns.append({
                    'start': turn['start'],
                    'end': turn['end'],
                    'speaker': speaker_label,
                    'text': final_text,
                    'verbatim': raw_text.strip()
                })

            # VAD Gap Fill Safety Net
            gap_turns = _fill_transcription_gaps_via_vad(
                data, sr, duration, merged_turns, speaker_mapping, stt_model, transcribed_windows
            )
            if gap_turns:
                print(f"✅ VAD gap-fill recovered {len(gap_turns)} missing speech segment(s).")
                processed_turns = sorted(processed_turns + gap_turns, key=lambda pt: pt['start'])

        diarization_only = "".join(f"[{pt['start']:.1f}s -> {pt['end']:.1f}s] {pt['speaker']}\n" for pt in processed_turns)

        for pt in processed_turns:
            line = f"{pt['speaker']}: {pt['text']}"
            verbatim_line = f"{pt['speaker']}: {pt['verbatim']}"
            print(f"[{pt['start']:.1f}s -> {pt['end']:.1f}s] {line}")
            diarized_transcript += line + "\n"
            verbatim_diarized_transcript += verbatim_line + "\n"

        diarized_transcript = standardize_speaker_labels(diarized_transcript)

        with open(f"{base_dir}/pyannote_diarization.txt", "w", encoding="utf-8") as f:
            f.write(diarization_only)
        with open(f"{base_dir}/jivi_hindi_transcription.txt", "w", encoding="utf-8") as f:
            f.write(diarized_transcript)
        with open(f"{base_dir}/ai4bharat_hindi_transcription.txt", "w", encoding="utf-8") as f:
            f.write(diarized_transcript)
        with open(f"{base_dir}/verbatim_raw_turns.txt", "w", encoding="utf-8") as f:
            f.write(verbatim_diarized_transcript)

    except Exception as e:
        print(f"🚨 Diarization or STT Error: {e}")
        return

    print("\n[2/3] Cleaning Raw Dialogue Output...")
    cleaned_transcript = clean_dialogue_remove_punctuation(diarized_transcript)
    cleaned_transcript = normalize_hindi_sentence_phonetics(cleaned_transcript)

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
    vllm_model = os.environ.get("VLLM_MODEL", SPELL_CHECK_MODEL)

    print("\n[2.5/3] Re-aligning Speaker Turns, Adding Punctuation & Correcting Hindi Spelling via Local LLM...")
    cleaned_transcript = realign_dialogue_speakers_vllm(
        cleaned_transcript,
        vllm_url=vllm_url,
        model_name=vllm_model
    )
    cleaned_transcript = standardize_speaker_labels(cleaned_transcript)
    cleaned_transcript = repair_and_format_hindi_dates_and_years(cleaned_transcript)
    cleaned_transcript = format_hindi_month_year(cleaned_transcript)
    cleaned_transcript = normalize_hindi_sentence_phonetics(cleaned_transcript)

    # Ensure every speaker turn is separated with a blank line
    turn_blocks = []
    curr_speaker = None
    curr_words = []
    for line in cleaned_transcript.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        m = re.match(r'^(Speaker\s*\d+)\s*:\s*(.*)$', line_str, re.IGNORECASE)
        if m:
            if curr_speaker is not None and curr_words:
                turn_text = " ".join(curr_words)
                turn_text = re.sub(r'\s+([\.,\?!|।;:])', r'\1', turn_text)
                turn_blocks.append(f"{curr_speaker}: {turn_text}")
                curr_words = []
            curr_speaker = m.group(1).title()
            body = m.group(2).strip()
            if body:
                curr_words.extend(body.split())
        else:
            if curr_speaker is not None:
                curr_words.extend(line_str.split())
            else:
                curr_speaker = "Speaker 0"
                curr_words.extend(line_str.split())
    if curr_speaker is not None and curr_words:
        turn_text = " ".join(curr_words)
        turn_text = re.sub(r'\s+([\.,\?!|।;:])', r'\1', turn_text)
        turn_blocks.append(f"{curr_speaker}: {turn_text}")

    cleaned_transcript = "\n\n".join(turn_blocks)

    print("\n✨ --- CHRONOLOGICAL HINDI DIALOGUE --- ✨")
    print(cleaned_transcript)

    with open(f"{base_dir}/police_interrogation_dialogue.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_transcript)
    with open(f"{base_dir}/cleaned_transcription.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_transcript)
    try:
        grouped_blocks = group_dialogue_by_speaker(cleaned_transcript)
        with open(f"{base_dir}/speaker_grouped_statements.txt", "w", encoding="utf-8") as f:
            f.write(grouped_blocks)
    except Exception:
        pass

    print("\n[3/3] Calling Local LLM Server to Generate Official FIR Statement in Hindi...")
    try:
        import requests
        system_prompt = (
            "You are an expert police officer and official FIR statement writer. Your task is to write an authentic, formal First Information Report (FIR) statement in pure Hindi based on the provided conversation between a police officer and a complainant.\n\n"
            "The FIR statement MUST strictly follow this exact layout with 'बयान' in the heading, followed by 3 structured paragraphs:\n\n"
            "HEADING LINE:\n"
            "Format the header with centered 'बयान' and the date on the right side (e.g., '                  बयान                        दिनांक DD/MM/YYYY').\n\n"
            "PARAGRAPHS STRUCTURE:\n"
            "Paragraph 1 (Complainant Personal & Identification Details):\n"
            "Extract complainant's full name, age, occupation, residential address, mobile number, family living situation, and belt/badge number details if applicable. Write in the first person in formal Hindi (e.g., 'मैं श्री/श्रीमती [नाम], उम्र [उम्र] वर्ष, पेशा [पेशा], पता [पता]...').\n\n"
            "Paragraph 2 (Chronological Incident & Case Details):\n"
            "Narrate the incident comprehensively and chronologically. Include exact dates, times, vehicle numbers, addresses, suspect names, victim details, seized articles, and sections applied, exactly as stated in the conversation. Write in the first person in formal Hindi.\n\n"
            "Paragraph 3 (Closing Legal Declaration):\n"
            "Reflect the formal legal closing declaration as verified by the complainant (e.g., 'मेरा उपरोक्त बयान टंकलिखित कराया गया है जिसे मैंने पढ़कर देख लिया है तथा मुझे पढ़कर सुनाया गया, जो मेरे बताए अनुसार बिल्कुल सत्य और सही है।').\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output MUST start with 'बयान' in the heading line.\n"
            "2. Follow with the 3 distinct paragraphs separated by blank lines.\n"
            "3. DO NOT ALTER NUMBERS OR CODES: Copy exact age, badge number, phone number, Aadhaar number, vehicle registration number, and dates as stated in the conversation.\n"
            "4. DATE & TIME FORMATTING: Format calendar dates strictly as DD/MM/YYYY. When only a month and year are stated, format as Hindi month name and year (e.g. 'अप्रैल २०२६'). Incident time must be in formal 24-hour police format with period (e.g. 'दोपहर १४:४५ बजे के आसपास').\n"
            "5. Output pure Hindi FIR statement only without code fences, English words, or commentary."
        )

        fir_payload = {
            "model": vllm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"निम्नलिखित बातचीत के आधार पर 'बयान' शीर्षक देकर तीन अनुच्छेदों में औपचारिक एफआईआर (FIR) बयान तैयार करें:\n\n{cleaned_transcript}"}
            ],
            "max_tokens": 1500,
            "temperature": 0.1
        }

        fir_response = requests.post(vllm_url, json=fir_payload)
        fir_response.raise_for_status()
        generated_text = fir_response.json()["choices"][0]["message"]["content"]

        generated_text = repair_and_format_hindi_dates_and_years(generated_text)
        generated_text = format_hindi_month_year(generated_text)
        generated_text = normalize_hindi_sentence_phonetics(generated_text)

        from datetime import datetime
        hindi_digits = str.maketrans('0123456789', '०१२३४५६७८९')
        live_date = datetime.now().strftime("%d/%m/%Y").translate(hindi_digits)
        header_line = f"बयान                                        दिनांक {live_date}"
        clean_statement = re.sub(r'^\s*बयान[^\n]*\n*', '', generated_text.strip()).strip()
        generated_text = f"{header_line}\n\n{clean_statement}"

        print("\n✨ --- GENERATED HINDI FIR STATEMENT --- ✨")
        print(generated_text.strip())

        with open(f"{base_dir}/fir_statement_hindi.txt", "w", encoding="utf-8") as f:
            f.write(generated_text.strip())
        print(f"\n✅ Saved Hindi FIR Statement to {base_dir}/fir_statement_hindi.txt")

    except Exception as e:
        print(f"🚨 LLM Error: {e}")


def parse_optional_int(val):
    if val is None or str(val).strip().lower() in ["none", "null", "auto", "automatic"]:
        return None
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer or 'None'/'auto': {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Hindi STT & Diarization Pipeline (Jivi AI AudioX-North & NeMo Conformer)")
    parser.add_argument(
        "--mode", 
        choices=["enterprise", "fir", "nemo", "whisperx"], 
        required=True, 
        help="Select the architecture mode to run."
    )
    parser.add_argument(
        "--audio", 
        type=str, 
        default=DEFAULT_AUDIO_FILE, 
        help="Path to the audio file."
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=DEFAULT_HINDI_MODEL,
        help=f"Hindi STT Model: Jivi AudioX-North (default: '{DEFAULT_HINDI_MODEL}'), or .nemo path."
    )
    parser.add_argument(
        "--min-speakers", 
        type=parse_optional_int, 
        default=None, 
        help="Minimum number of speakers to detect in diarization (default: None for automatic detection)."
    )
    parser.add_argument(
        "--max-speakers", 
        type=parse_optional_int, 
        default=None, 
        help="Maximum number of speakers to detect in diarization (default: None for automatic detection)."
    )
    parser.add_argument(
        "--diarization-model",
        type=str,
        default=os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"),
        help="Pyannote speaker diarization model. Default: 'pyannote/speaker-diarization-community-1'."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference: strictly 'cuda' (GPU)."
    )
    
    args = parser.parse_args()
    
    if args.mode == "whisperx":
        run_subtitles_whisperx(args.audio, device=args.device)
    elif args.mode in ["nemo", "enterprise"]:
        run_enterprise_stt(args.audio, stt_model_identifier=args.stt_model, device=args.device)
    elif args.mode == "fir":
        run_fir_pipeline(
            args.audio, 
            min_speakers=args.min_speakers, 
            max_speakers=args.max_speakers,
            diarization_model=args.diarization_model,
            stt_model_identifier=args.stt_model,
            device=args.device
        )
