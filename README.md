# ComfyUI-MusicGenMelody

Picture -> song glue for Graydient, plain `transformers` (no audiocraft).

- `ImageToMusicBrief`: image -> Qwen2.5-VL-3B -> one-line text-to-music prompt.
- `MusicGenMelodyGenerate`: prompt (+ optional melody clip) -> `facebook/musicgen-melody` -> AUDIO.

Weights are looked up under `{ComfyUI}/models/musicgen/musicgen-melody` and
`{ComfyUI}/models/vlm/Qwen2.5-VL-3B-Instruct`, falling back to the Hugging Face ids.

Notes (measured):
- Use bf16/fp32, never fp16 (T5 overflow gives NaN audio).
- transformers 5.9.0 ignores MusicGen conditioning; 4.51.3 and 5.22.2 work.
- With the KV cache on, the melody input is ignored on all tested versions, so melody mode
  runs with `use_cache=False` and is capped at 15 s.

The musicgen-melody weights are CC-BY-NC 4.0 (non-commercial).
