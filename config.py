from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    # ────────────────────────────────────────────────────────────────
    # Transcription / ASR
    # ────────────────────────────────────────────────────────────────
    whisper_model: str = "medium"              # faster-whisper model size
    device: str = "auto"                       # "cuda" | "cpu" | "auto"
    word_timestamps: bool = True
    force_language: Optional[str] = None       # e.g. "ur", "hi", "en"

    # Language-forcing behaviour (used only during TRANSCRIBE)
    enable_conditional_force_non_en: bool = True
    english_codes: Tuple[str, ...] = ("en",)
    force_non_english_to: str = "ur"

    # Audio extraction
    audio_profile: str = "speech_wav"          # "opus_copy" | "music_wav" | "speech_wav"

    # ────────────────────────────────────────────────────────────────
    # OpenAI LLM settings (ENV VARS are preferred; these are defaults)
    # ────────────────────────────────────────────────────────────────
    openai_model: str = "gpt-4o-mini"          # used if OPENAI_MODEL is not set
    temperature = 1
    openai_base_url: Optional[str] = None      # optional (proxy/gateway)
    openai_org: Optional[str] = None           # optional

    # Planner knobs
    num_clips: int = 5
    target_duration: int = 35                  # advisory; min/max are enforced
    min_duration: int = 30
    max_duration: int = 40
    planner_segment_text_max_chars: int = 1800
    planner_prompt_text_max_chars: int = 120_000

    # Rendering / encoding
    portrait_width: int = 1080
    portrait_height: int = 1920
    video_bitrate_kbps: int = 6_000
    crf: int = 18
    pix_fmt: str = "yuv420p"
    acodec: str = "aac"
    audio_bitrate_kbps: int = 192
    
    enable_active_crop = True


    # Captioning (placeholders kept for compatibility)
    caption_font: str = "Arial"
    caption_font_emph: str = "Arial Black"
    chunk_min_words: int = 3
    chunk_max_words: int = 4

    # Misc
    verbose: bool = True

# Singleton default
DEFAULT = Config()
