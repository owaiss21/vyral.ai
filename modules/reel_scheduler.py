
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import json, time, threading

from .instagram_uploader import InstagramUploader
from .tiktok_uploader import TikTokUploader


def default_caption_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").strip()


class ReelPostingScheduler:
    """
    Posts reels from a directory one-by-one with a fixed delay (minutes).
    Can post to Instagram and/or TikTok depending on which uploaders are connected.
    """

    def __init__(
        self,
        reels_dir: str | Path,
        delay_minutes: int = 180,
        caption_fn: Callable[[Path], str] = default_caption_from_path,
        posted_log_path: str | Path | None = None,
        session_path: str | Path | None = None,
        proxy: str | None = None,
        tt_uploader: TikTokUploader | None = None,
        enable_instagram: bool = True,
        enable_tiktok: bool = True,
    ):
        self.reels_dir = Path(reels_dir)
        self.delay_minutes = int(delay_minutes)
        self.caption_fn = caption_fn
        self.posted_log_path = Path(posted_log_path) if posted_log_path else self.reels_dir / "posted.json"
        # Session path for Instagram; TikTok keeps its own session inside tt_uploader
        self.session_path = Path(session_path) if session_path else Path("workdir/.ig_session.json")
        self.proxy = proxy

        # Uploaders (created in run() for IG; injected for TikTok)
        self._uploader: InstagramUploader | None = None
        self._tt_uploader: TikTokUploader | None = tt_uploader

        # Flags
        self.enable_instagram = bool(enable_instagram)
        self.enable_tiktok = bool(enable_tiktok)

        # Controls
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Posted index
        self._posted_index: dict[str, dict] = self._read_posted_index()

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return (self._uploader is not None or self._tt_uploader is not None) and not self._stop.is_set()

    def run(self, username: str, password: str) -> None:
        # Instagram: instantiate & login if enabled
        if self.enable_instagram:
            self._uploader = InstagramUploader(session_path=self.session_path, proxy=self.proxy)
            if not self._uploader.login(username, password):
                self._uploader = None  # continue with TikTok only if IG login fails

        # TikTok: require injected uploader already connected
        if self.enable_tiktok and not self._tt_uploader:
            self.enable_tiktok = False

        while not self._stop.is_set():
            next_file = self._next_unposted_file()
            if not next_file:
                self._sleep_with_cancel(60)
                continue

            caption = self.caption_fn(next_file)
            results: Dict[str, bool] = {}

            if self.enable_instagram and self._uploader is not None:
                ig_ok = self._uploader.upload_reel_from_path(next_file, caption=caption)
                results["instagram"] = bool(ig_ok)

            if self.enable_tiktok and self._tt_uploader is not None:
                tt_ok = self._tt_uploader.upload_reel_from_path(next_file, caption=caption)
                results["tiktok"] = bool(tt_ok)

            ts = int(time.time())
            if results:
                entry = {"path": str(next_file), "caption": caption, "posted_at_ts": ts, "result": results}
                self._mark_posted(next_file, entry)

            self._sleep_with_cancel(self.delay_minutes * 60)

        try:
            if self._uploader:
                self._uploader.cleanup()
        except Exception:
            pass
        self._uploader = None

    # --- internals ---
    def _sleep_with_cancel(self, seconds: int) -> None:
        # Sleep in small chunks so we can react to stop() quickly
        remain = float(seconds)
        while remain > 0 and not self._stop.is_set():
            time.sleep(min(1.0, remain))
            remain -= 1.0

    def _next_unposted_file(self) -> Optional[Path]:
        posted = set(self._posted_index.keys())
        for p in sorted(self.reels_dir.glob("*.mp4")) + sorted(self.reels_dir.glob("*.mov")):
            if str(p) not in posted:
                return p
        return None

    def _read_posted_index(self) -> Dict[str, Any]:
        try:
            if self.posted_log_path.exists():
                with open(self.posted_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _mark_posted(self, path: Path, entry: Dict[str, Any]) -> None:
        with self._lock:
            self._posted_index[str(path)] = entry
            try:
                self.posted_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.posted_log_path, "w", encoding="utf-8") as f:
                    json.dump(self._posted_index, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
