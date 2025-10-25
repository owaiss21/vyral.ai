from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # allow single-process coexistence
os.environ.setdefault("OMP_NUM_THREADS", "1")          # reduce thread contention
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")  
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
from datetime import datetime

import streamlit as st

# Project modules
from config import DEFAULT as CFG
from modules.downloader import download_youtube_video
from modules.extract_audio import extract_audio
from modules.transcribe import transcribe_audio
from modules.transliteration import transliterate_transcript
from modules.llm_clipper import propose_plan_file
from modules.video_editing import render_from_paths

st.set_page_config(page_title="Shorts Generation Agent", layout="wide")


# -----------------------------------------------------------------------------
# Session helpers
# -----------------------------------------------------------------------------
def _ensure_session():
    # Workspace is fixed for Basic/Advanced
    st.session_state["workspace"] = "workdir"


_ensure_session()


# -----------------------------------------------------------------------------
# Small UI helpers
# -----------------------------------------------------------------------------
def _badge_row(title: str, steps: Dict[str, bool], progress: Optional[Dict[str, int]] = None) -> None:
    cols = st.columns(6, gap="small")
    labels = ["Download", "Extract", "ASR", "Transliteration", "Planning", "Render"]
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for i, name in enumerate(labels):
            cols[i].markdown(f"{name}\n\n`{'done' if steps.get(name) else 'pending'}`")
            # Small progress bar under each badge (0–100)
            p = 0 if progress is None else int(progress.get(name, 0))
            cols[i].progress(p)


def _topbar():
    left, _ = st.columns([0.72, 0.28])
    with left:
        st.title("Shorts Generation Agent")
    # Right side intentionally empty.


# -----------------------------------------------------------------------------
# BASIC MODE (YouTube-only)
# -----------------------------------------------------------------------------
def _pipeline_basic_ui():
    st.subheader("Basic Mode")
    with st.container(border=True):
        # Layout: left inputs, right Start button
        left, right = st.columns([0.78, 0.22])
        with left:
            url = st.text_input("YouTube URL", key="basic_url")
            row = st.columns([0.5, 0.5])
            with row[0]:
                num_clips = st.number_input(
                    "How many shorts?", min_value=1, max_value=10000, value=5, step=1, key="basic_num_clips"
                )
            with row[1]:
                target_duration = st.number_input(
                    "Target duration per clip (sec)",
                    min_value=10,
                    max_value=120,
                    value=int(getattr(CFG, "target_duration", 35)),
                    key="basic_target_duration",
                )
        with right:
            st.write("")
            st.write("")
            start_btn = st.button("Start", use_container_width=True, type="primary", key="basic_start_btn")

    if start_btn:
        if not url.strip():
            st.warning("Please enter a YouTube URL.")
            return
        _run_pipeline_from_stage(
            stage="Download",
            workdir=st.session_state.workspace,  # fixed to 'workdir'
            url=url.strip(),
            video_path="",
            transcript_json="",
            translit_json="",
            plan_json="",
            num_clips=num_clips,
            target_duration=target_duration,
        )


# -----------------------------------------------------------------------------
# ADVANCED MODE (start from any stage)
# -----------------------------------------------------------------------------
def _advanced_mode_ui():
    st.subheader("Advanced Mode")
    with st.container(border=True):
        stage = st.selectbox(
            "Start from", ["Download", "Extract", "Transcribe", "Transliterate", "Plan", "Render"], index=0, key="adv_stage_select"
        )

        # Infer required fields by stage
        url = ""
        video_path = ""
        transcript_json = ""
        translit_json = ""
        plan_json = ""
        num_clips = 5
        target_duration = int(getattr(CFG, "target_duration", 35))

        if stage == "Download":
            url = st.text_input("YouTube URL", key="adv_url")
            row = st.columns([0.5, 0.5])
            with row[0]:
                num_clips = st.number_input(
                    "How many shorts?", min_value=1, max_value=10000, value=5, step=1, key="adv_num_dl"
                )
            with row[1]:
                target_duration = st.number_input(
                    "Target duration per clip (sec)", min_value=10, max_value=120, value=target_duration, key="adv_tdur_dl"
                )
        elif stage == "Extract":
            video_path = st.text_input("Video file path (MP4/MOV)", key="adv_video_ex")
            row = st.columns([0.5, 0.5])
            with row[0]:
                num_clips = st.number_input(
                    "How many shorts?", min_value=1, max_value=10000, value=5, step=1, key="adv_num_ex"
                )
            with row[1]:
                target_duration = st.number_input(
                    "Target duration per clip (sec)", min_value=10, max_value=120, value=target_duration, key="adv_tdur_ex"
                )
        elif stage == "Transcribe":
            video_path = st.text_input("Video file path (MP4/MOV)", key="adv_video_tr")
            row = st.columns([0.5, 0.5])
            with row[0]:
                num_clips = st.number_input(
                    "How many shorts?", min_value=1, max_value=10000, value=5, step=1, key="adv_num_tr"
                )
            with row[1]:
                target_duration = st.number_input(
                    "Target duration per clip (sec)", min_value=10, max_value=120, value=target_duration, key="adv_tdur_tr"
                )
        elif stage == "Transliterate":
            transcript_json = st.text_input("Transcript JSON path", key="adv_transcript_tl")
            video_path = st.text_input("Video file path (for upcoming render)", key="adv_video_tl")
            row = st.columns([0.5, 0.5])
            with row[0]:
                num_clips = st.number_input(
                    "How many shorts?", min_value=1, max_value=10000, value=5, step=1, key="adv_num_tl"
                )
            with row[1]:
                target_duration = st.number_input(
                    "Target duration per clip (sec)", min_value=10, max_value=120, value=target_duration, key="adv_tdur_tl"
                )
        elif stage == "Plan":
            translit_json = st.text_input("Transliteration JSON path (*_roman.json)", key="adv_translit_pl")
            video_path = st.text_input("Video file path (for render)", key="adv_video_pl")
            row = st.columns([0.5, 0.5])
            with row[0]:
                num_clips = st.number_input(
                    "How many shorts?", min_value=1, max_value=10000, value=5, step=1, key="adv_num_pl"
                )
            with row[1]:
                target_duration = st.number_input(
                    "Target duration (sec)", min_value=10, max_value=120, value=target_duration, key="adv_tdur_pl"
                )
        elif stage == "Render":
            translit_json = st.text_input("Transliteration JSON path (*_roman.json)", key="adv_translit_rd")
            plan_json = st.text_input("Plan JSON path", key="adv_plan_rd")
            video_path = st.text_input("Video file path (MP4/MOV)", key="adv_video_rd")

        cols = st.columns([0.8, 0.2])
        with cols[1]:
            adv_start = st.button("Start", use_container_width=True, type="primary", key="adv_start_btn")

    if adv_start:
        _run_pipeline_from_stage(
            stage=stage,
            workdir=st.session_state.workspace,  # fixed to 'workdir'
            url=url.strip(),
            video_path=video_path.strip(),
            transcript_json=transcript_json.strip(),
            translit_json=translit_json.strip(),
            plan_json=plan_json.strip(),
            num_clips=num_clips,
            target_duration=target_duration,
        )


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def _run_pipeline_from_stage(
    *,
    stage: str,
    workdir: str,
    url: str = "",
    video_path: str = "",
    transcript_json: str = "",
    translit_json: str = "",
    plan_json: str = "",
    num_clips: int = 5,
    target_duration: int = 35,
) -> None:
    # Directories (derived from fixed workdir)
    wd = Path(workdir)
    raw_videos = wd / "raw_videos"
    raw_audios = wd / "raw_audios"
    trans_dir = wd / "transcriptions"
    translit_dir = wd / "transliteration"
    plans_dir = wd / "llm_shorts"
    shorts_dir = wd / "finalized_shorts"
    for d in (raw_videos, raw_audios, trans_dir, translit_dir, plans_dir, shorts_dir):
        d.mkdir(parents=True, exist_ok=True)

    step_names = ["Download", "Extract", "ASR", "Transliteration", "Planning", "Render"]
    steps_done = {name: False for name in step_names}
    steps_prog = {name: 0 for name in step_names}  # 0..100

    board = st.empty()

    def _refresh(title="Progress"):
        with board.container():
            _badge_row(title, steps_done, steps_prog)

    _refresh()

    vp = video_path
    ap = ""
    tr_json = transcript_json
    tl_json = translit_json
    pl_json = plan_json

    try:
        # START FROM SELECTED STAGE AND CONTINUE
        if stage == "Download":
            # Download
            steps_prog["Download"] = 10; _refresh()
            vp = download_youtube_video(url, str(raw_videos))
            steps_done["Download"] = True; steps_prog["Download"] = 100; _refresh()

            # Extract
            steps_prog["Extract"] = 10; _refresh()
            ap = extract_audio(vp, str(raw_audios), profile=getattr(CFG, "audio_profile", "speech_wav"))
            steps_done["Extract"] = True; steps_prog["Extract"] = 100; _refresh()

            # ASR
            steps_prog["ASR"] = 10; _refresh()
            tr_json, _ = transcribe_audio(
                ap, str(trans_dir),
                model_size=getattr(CFG, "whisper_model", "medium"),
                device=getattr(CFG, "device", "auto"),
                word_timestamps=getattr(CFG, "word_timestamps", True),
                force_language=getattr(CFG, "force_language", None),
                conditional_force_non_en=getattr(CFG, "conditional_force_non_en", None),
                english_codes=getattr(CFG, "english_codes", ("en",)),
                verbose=getattr(CFG, "verbose", True),
            )
            steps_done["ASR"] = True; steps_prog["ASR"] = 100; _refresh()

            # Transliteration
            steps_prog["Transliteration"] = 10; _refresh()
            tl_json = transliterate_transcript(
                tr_json, out_dir=str(translit_dir),
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                batch_size=int(getattr(CFG, "transliteration_batch_size", 12)),
            )
            steps_done["Transliteration"] = True; steps_prog["Transliteration"] = 100; _refresh()

            # Planning
            steps_prog["Planning"] = 10; _refresh()
            pl_json = propose_plan_file(
                tl_json, str(plans_dir),
                num_clips=num_clips, target_duration=target_duration,
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                api_base="", verbose=getattr(CFG, "verbose", True),
            )
            steps_done["Planning"] = True; steps_prog["Planning"] = 100; _refresh()

            # Render
            steps_prog["Render"] = 10; _refresh()
            out_paths = render_from_paths(vp, tl_json, pl_json, str(shorts_dir), verbose=getattr(CFG, "verbose", True))
            steps_done["Render"] = True; steps_prog["Render"] = 100; _refresh()

        elif stage == "Extract":
            # Extract
            steps_prog["Extract"] = 10; _refresh()
            ap = extract_audio(vp, str(raw_audios), profile=getattr(CFG, "audio_profile", "speech_wav"))
            steps_done["Extract"] = True; steps_prog["Extract"] = 100; _refresh()

            # ASR
            steps_prog["ASR"] = 10; _refresh()
            tr_json, _ = transcribe_audio(
                ap, str(trans_dir),
                model_size=getattr(CFG, "whisper_model", "medium"),
                device=getattr(CFG, "device", "auto"),
                word_timestamps=getattr(CFG, "word_timestamps", True),
                force_language=getattr(CFG, "force_language", None),
                conditional_force_non_en=getattr(CFG, "conditional_force_non_en", None),
                english_codes=getattr(CFG, "english_codes", ("en",)),
                verbose=getattr(CFG, "verbose", True),
            )
            steps_done["ASR"] = True; steps_prog["ASR"] = 100; _refresh()

            # Transliteration
            steps_prog["Transliteration"] = 10; _refresh()
            tl_json = transliterate_transcript(
                tr_json, out_dir=str(translit_dir),
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                batch_size=int(getattr(CFG, "transliteration_batch_size", 12)),
            )
            steps_done["Transliteration"] = True; steps_prog["Transliteration"] = 100; _refresh()

            # Planning
            steps_prog["Planning"] = 10; _refresh()
            pl_json = propose_plan_file(
                tl_json, str(plans_dir),
                num_clips=num_clips, target_duration=target_duration,
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                api_base="", verbose=getattr(CFG, "verbose", True),
            )
            steps_done["Planning"] = True; steps_prog["Planning"] = 100; _refresh()

            # Render
            steps_prog["Render"] = 10; _refresh()
            out_paths = render_from_paths(vp, tl_json, pl_json, str(shorts_dir), verbose=getattr(CFG, "verbose", True))
            steps_done["Render"] = True; steps_prog["Render"] = 100; _refresh()

        elif stage == "Transcribe":
            # Extract
            steps_prog["Extract"] = 10; _refresh()
            ap = extract_audio(vp, str(raw_audios), profile=getattr(CFG, "audio_profile", "speech_wav"))
            steps_done["Extract"] = True; steps_prog["Extract"] = 100; _refresh()

            # ASR
            steps_prog["ASR"] = 10; _refresh()
            tr_json, _ = transcribe_audio(
                ap, str(trans_dir),
                model_size=getattr(CFG, "whisper_model", "medium"),
                device=getattr(CFG, "device", "auto"),
                word_timestamps=getattr(CFG, "word_timestamps", True),
                force_language=getattr(CFG, "force_language", None),
                conditional_force_non_en=getattr(CFG, "conditional_force_non_en", None),
                english_codes=getattr(CFG, "english_codes", ("en",)),
                verbose=getattr(CFG, "verbose", True),
            )
            steps_done["ASR"] = True; steps_prog["ASR"] = 100; _refresh()

            # Transliteration
            steps_prog["Transliteration"] = 10; _refresh()
            tl_json = transliterate_transcript(
                tr_json, out_dir=str(translit_dir),
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                batch_size=int(getattr(CFG, "transliteration_batch_size", 12)),
            )
            steps_done["Transliteration"] = True; steps_prog["Transliteration"] = 100; _refresh()

            # Planning
            steps_prog["Planning"] = 10; _refresh()
            pl_json = propose_plan_file(
                tl_json, str(plans_dir),
                num_clips=num_clips, target_duration=target_duration,
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                api_base="", verbose=getattr(CFG, "verbose", True),
            )
            steps_done["Planning"] = True; steps_prog["Planning"] = 100; _refresh()

            # Render
            steps_prog["Render"] = 10; _refresh()
            out_paths = render_from_paths(vp, tl_json, pl_json, str(shorts_dir), verbose=getattr(CFG, "verbose", True))
            steps_done["Render"] = True; steps_prog["Render"] = 100; _refresh()

        elif stage == "Transliterate":
            # Transliteration
            steps_prog["Transliteration"] = 10; _refresh()
            tl_json = transliterate_transcript(
                tr_json or transcript_json, out_dir=str(translit_dir),
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                batch_size=int(getattr(CFG, "transliteration_batch_size", 12)),
            )
            steps_done["Transliteration"] = True; steps_prog["Transliteration"] = 100; _refresh()

            # Planning
            steps_prog["Planning"] = 10; _refresh()
            pl_json = propose_plan_file(
                tl_json, str(plans_dir),
                num_clips=num_clips, target_duration=target_duration,
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                api_base="", verbose=getattr(CFG, "verbose", True),
            )
            steps_done["Planning"] = True; steps_prog["Planning"] = 100; _refresh()

            # Render
            steps_prog["Render"] = 10; _refresh()
            out_paths = render_from_paths(vp, tl_json, pl_json, str(shorts_dir), verbose=getattr(CFG, "verbose", True))
            steps_done["Render"] = True; steps_prog["Render"] = 100; _refresh()

        elif stage == "Plan":
            # Planning
            steps_prog["Planning"] = 10; _refresh()
            pl_json = propose_plan_file(
                tl_json or translit_json, str(plans_dir),
                num_clips=num_clips, target_duration=target_duration,
                model=getattr(CFG, "llm_model", "gpt-4o-mini"),
                api_base="", verbose=getattr(CFG, "verbose", True),
            )
            steps_done["Planning"] = True; steps_prog["Planning"] = 100; _refresh()

            # Render
            steps_prog["Render"] = 10; _refresh()
            out_paths = render_from_paths(vp, tl_json or translit_json, pl_json, str(shorts_dir),
                                          verbose=getattr(CFG, "verbose", True))
            steps_done["Render"] = True; steps_prog["Render"] = 100; _refresh()

        else:  # Render
            # Render
            steps_prog["Render"] = 10; _refresh()
            out_paths = render_from_paths(vp, tl_json or translit_json, pl_json or plan_json, str(shorts_dir),
                                          verbose=getattr(CFG, "verbose", True))
            steps_done["Render"] = True; steps_prog["Render"] = 100; _refresh()

        count = len(out_paths or [])
        st.success(f"{count} videos have been saved.")
        if count:
            with st.expander("Saved files"):
                for p in out_paths:
                    st.write(p)

    except Exception as err:
        st.error(f"Error: {err}")


def main():
    _topbar()
    tabs = st.tabs(["Basic", "Advanced"])
    with tabs[0]:
        _pipeline_basic_ui()
    with tabs[1]:
        _advanced_mode_ui()


if __name__ == "__main__":
    main()
