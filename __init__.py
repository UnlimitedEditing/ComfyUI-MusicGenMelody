"""ComfyUI-MusicGenMelody: picture -> song glue for Graydient.

Two nodes, plain `transformers` (no audiocraft, no compiled deps):

  ImageToMusicBrief      image URL/filename -> Qwen2.5-VL -> a one-line text-to-music prompt
  MusicGenMelodyGenerate prompt (+ optional melody audio) -> facebook/musicgen-melody -> AUDIO

Weights are expected under {ComfyUI}/models/ (staged by Graydient concept_mapping):
  musicgen/musicgen-melody/...            (facebook/musicgen-melody)
  vlm/Qwen2.5-VL-3B-Instruct/...          (Qwen/Qwen2.5-VL-3B-Instruct)
If the local folder is missing the nodes fall back to the Hugging Face id (slower: the
download then counts against the run-time budget).

musicgen-melody weights are CC-BY-NC 4.0 (non-commercial).
"""
import io
import os
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np

try:
    import folder_paths
    _MODELS_DIR = folder_paths.models_dir
    _INPUT_DIR = folder_paths.get_input_directory()
except Exception:  # allows importing outside ComfyUI for tests
    _MODELS_DIR = os.path.join(os.getcwd(), "models")
    _INPUT_DIR = os.getcwd()

MELODY_SR = 32000
FRAME_RATE = 50  # EnCodec tokens per second of audio
MAX_SECONDS = 30  # musicgen-melody was trained on 30 s windows
# With the KV cache on, transformers (4.51.3, 5.9.0, 5.22.2 all tested) ignores the melody
# (input_features): output is bit-identical with and without it. use_cache=False honours it
# but recomputes the whole sequence every step, so melody mode is capped much lower.
MELODY_MAX_SECONDS = 15

_UA = {"User-Agent": "Mozilla/5.0 (ComfyUI-MusicGenMelody)"}


def _local_or_id(subdir, local_name, hf_id):
    p = os.path.join(_MODELS_DIR, subdir, local_name)
    if os.path.isfile(os.path.join(p, "config.json")):
        return p
    print(f"[MusicGenMelody] {p} not staged, falling back to hub id {hf_id}")
    return hf_id


def _fetch_bytes(url):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _resolve_source(url, filename):
    """Return bytes for a URL, or for a file already in the input dir; None if neither."""
    url = (url or "").strip()
    filename = (filename or "").strip()
    if url:
        return _fetch_bytes(url)
    if filename:
        for cand in (filename, os.path.join(_INPUT_DIR, filename),
                     os.path.join(_INPUT_DIR, os.path.basename(filename))):
            if os.path.isfile(cand):
                with open(cand, "rb") as f:
                    return f.read()
        raise FileNotFoundError(f"[MusicGenMelody] file not found: {filename}")
    return None


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("[MusicGenMelody] ffmpeg not found; cannot decode the melody clip")


def _decode_mono_32k(data, max_seconds=MAX_SECONDS):
    """Any audio/video container -> float32 mono @32 kHz, first `max_seconds` only."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tf.write(data)
        path = tf.name
    try:
        cmd = [_ffmpeg_exe(), "-v", "error", "-i", path, "-t", str(max_seconds),
               "-vn", "-ac", "1", "-ar", str(MELODY_SR), "-f", "f32le", "-"]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    arr = np.frombuffer(out, dtype=np.float32).copy()
    if arr.size < MELODY_SR // 2:
        raise RuntimeError("[MusicGenMelody] melody clip is empty or shorter than 0.5 s")
    return arr


# --------------------------------------------------------------------------------------
# Node 1: image -> music brief
# --------------------------------------------------------------------------------------

BRIEF_SYSTEM = (
    "You are a film music supervisor. You look at a picture and write ONE prompt for a "
    "text-to-music model, as a single comma-separated phrase, not labelled fields. Cover "
    "genre, mood, main instruments, production feel and tempo in BPM, chosen to suit the "
    "scene. Example: 'slow ambient piano and warm strings, soft pads, melancholic and "
    "spacious, 70 BPM'. Instrumental only: never mention vocals, singing, lyrics, artists, "
    "songs or brands. English, at most 30 words, no quotes, no labels like 'Genre:', no "
    "preamble."
)
BRIEF_USER = "Write the music brief for this picture."


class ImageToMusicBrief:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"default": ""}),
                "image_filename": ("STRING", {"default": ""}),
                "style_hint": ("STRING", {"multiline": True, "default": ""}),
                "max_new_tokens": ("INT", {"default": 96, "min": 16, "max": 512}),
                "model_dir": ("STRING", {"default": "Qwen/Qwen2.5-VL-3B-Instruct"}),
            },
            "optional": {
                # Empty = the built-in film-music-supervisor instruction (unchanged behaviour). Non-empty replaces it,
                # so each workflow can ask the vision model for something different.
                "instruction": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("music_prompt", "vlm_brief")
    FUNCTION = "run"
    CATEGORY = "MusicGenMelody"

    def run(self, image_url, image_filename, style_hint, max_new_tokens, model_dir, instruction=""):
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLM
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as VLM

        data = _resolve_source(image_url, image_filename)
        if data is None:
            raise RuntimeError("[MusicGenMelody] no image: set init_image_url or init_image_filename")
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((896, 896))  # bound the vision-token count

        if model_dir and not os.path.isdir(model_dir) and "/" in model_dir:
            model_dir = _local_or_id("vlm", model_dir.split("/")[-1], model_dir)
        processor = AutoProcessor.from_pretrained(model_dir)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = VLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)

        messages = [
            {"role": "system", "content": (instruction or "").strip() or BRIEF_SYSTEM},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": BRIEF_USER}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        brief = processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        brief = " ".join(brief.replace('"', "").split())

        del model, processor, inputs, ids
        torch.cuda.empty_cache()

        hint = (style_hint or "").strip()
        prompt = f"{hint}. {brief}" if hint else brief
        print(f"[MusicGenMelody] brief: {prompt}")
        return (prompt, brief)


# --------------------------------------------------------------------------------------
# Node 2: musicgen-melody
# --------------------------------------------------------------------------------------

class MusicGenMelodyGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "calm cinematic piano and strings, slow, 70 BPM"}),
                "duration_seconds": ("INT", {"default": 20, "min": 2, "max": MAX_SECONDS}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 250, "min": 0, "max": 2048}),
                "melody_url": ("STRING", {"default": ""}),
                "melody_filename": ("STRING", {"default": ""}),
                "model_dir": ("STRING", {"default": "facebook/musicgen-melody"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "run"
    CATEGORY = "MusicGenMelody"

    def run(self, prompt, duration_seconds, guidance_scale, seed, temperature, top_k,
            melody_url, melody_filename, model_dir):
        import torch
        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

        if model_dir == "facebook/musicgen-melody":
            model_dir = _local_or_id("musicgen", "musicgen-melody", model_dir)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # NOT float16: the T5 text encoder overflows in fp16 and the whole clip comes out
        # NaN (measured locally, GTX 1660 Ti). bf16 and fp32 are both clean.
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(model_dir)
        model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype).to(dev)

        melody = None
        data = _resolve_source(melody_url, melody_filename)
        if data is not None:
            melody = _decode_mono_32k(data)
        mode = "melody-guided" if melody is not None else "text-only"

        kw = dict(text=[prompt], padding=True, return_tensors="pt")
        if melody is not None:
            kw.update(audio=melody, sampling_rate=MELODY_SR)
        inputs = processor(**kw).to(dev)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype)

        torch.manual_seed(int(seed))
        cap = MELODY_MAX_SECONDS if melody is not None else MAX_SECONDS
        if duration_seconds > cap:
            print(f"[MusicGenMelody] {mode}: duration capped {duration_seconds}s -> {cap}s")
        n_tokens = int(min(duration_seconds, cap) * FRAME_RATE)
        with torch.no_grad():
            wav = model.generate(
                **inputs, do_sample=True, guidance_scale=float(guidance_scale),
                temperature=float(temperature), top_k=int(top_k) or None,
                max_new_tokens=n_tokens,
                use_cache=melody is None)  # see MELODY_MAX_SECONDS note
        sr = int(model.config.audio_encoder.sampling_rate)
        wav = wav.detach().float().cpu()  # (batch, channels, samples)
        wav = wav / max(1.0, float(wav.abs().max()))  # avoid clipping on save

        del model, processor, inputs
        torch.cuda.empty_cache()

        info = f"{mode}, {wav.shape[-1] / sr:.1f}s @ {sr} Hz, prompt: {prompt}"
        print(f"[MusicGenMelody] {info}")
        return ({"waveform": wav, "sample_rate": sr}, info)


NODE_CLASS_MAPPINGS = {
    "ImageToMusicBrief": ImageToMusicBrief,
    "MusicGenMelodyGenerate": MusicGenMelodyGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToMusicBrief": "Image to Music Brief (Qwen2.5-VL)",
    "MusicGenMelodyGenerate": "MusicGen Melody Generate",
}
