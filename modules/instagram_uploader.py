from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from urllib.parse import urlparse
import tempfile, os, time

import requests
from instagrapi import Client
from pydantic import ValidationError

# --- BEGIN: instagrapi extractor normalization patch ---
_PATCHED_EXTRACTOR = False

def _normalize_media_for_validation(media_dict: dict) -> dict:
    """Fill missing nested keys that Pydantic models expect but IG sometimes omits.
    This prevents ValidationError: 'Field required' on nested clips metadata.
    """
    try:
        cm = media_dict.get("clips_metadata")
        if isinstance(cm, dict):
            osi = cm.get("original_sound_info")
            if not isinstance(osi, dict):
                osi = {}
            artist = osi.get("ig_artist")
            if not isinstance(artist, dict):
                artist = {}
            # Keys that have been observed missing (instagrapi#2263 / #2254)
            artist.setdefault("profile_pic_id", "")
            artist.setdefault("profile_pic_url", "")
            # Keep existing values if present; otherwise ensure string defaults
            artist.setdefault("username", artist.get("username", ""))
            artist.setdefault("full_name", artist.get("full_name", ""))
            osi["ig_artist"] = artist
            cm["original_sound_info"] = osi
            media_dict["clips_metadata"] = cm
    except Exception:
        # Be permissive; this is a best-effort normalizer
        pass
    return media_dict


def _patch_instagrapi_extractors_once():
    global _PATCHED_EXTRACTOR
    if _PATCHED_EXTRACTOR:
        return
    try:
        import instagrapi.extractors as _ex
        _orig = _ex.extract_media_v1

        def _wrapped_extract_media_v1(media: dict):
            media = dict(media or {})
            media = _normalize_media_for_validation(media)
            return _orig(media)

        # monkey-patch
        _ex.extract_media_v1 = _wrapped_extract_media_v1  # type: ignore[attr-defined]
        _PATCHED_EXTRACTOR = True
    except Exception:
        # If anything goes wrong, we leave the original behavior intact
        pass
# --- END: instagrapi extractor normalization patch ---



class InstagramUploader:
    """
    Wrapper around instagrapi.Client with session persistence and two upload helpers.
    Includes a workaround for Pydantic validation glitches in clip_upload responses.
    """
    def __init__(
        self,
        session_path: Optional[Union[str, Path]] = ".ig_session.json",
        proxy: Optional[str] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.client = Client()
        if proxy:
            try:
                self.client.set_proxy(proxy)
            except Exception:
                pass

        self.session_path = Path(session_path) if session_path else None
        if self.session_path and self.session_path.exists():
            try:
                self.client.load_settings(str(self.session_path))
            except Exception:
                # continue without settings
                pass

        self._temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="ig_temp_"))
        self._tmp_files: List[Path] = []

    # ---------------- Login / Session ----------------
    def login(self, username: str, password: str, verification_code: Optional[str] = None) -> bool:
        try:
            if verification_code:
                self.client.login(username, password, verification_code=verification_code)
            else:
                self.client.login(username, password)
            if self.session_path:
                try:
                    self.client.dump_settings(str(self.session_path))
                except Exception:
                    pass
            return True
        except Exception as e:
            print(f"[InstagramUploader] Login failed: {e}")
            return False

    # ---------------- Helpers ----------------
    def _download_to_temp(self, url: str) -> Path:
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "video"
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"
        path = self._temp_dir / filename
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        self._tmp_files.append(path)
        return path

    def _confirm_recent_post(self, caption: str, within_sec: int = 120) -> Optional[Dict[str, Any]]:
        """
        Workaround for instagrapi clip_upload pydantic ValidationError.
        We poll the user's recent clips and match by caption and recency.
        """
        try:
            me = int(self.client.user_id)
            clips = self.client.user_clips(me, amount=5)  # recent
            now = time.time()
            for m in clips:
                try:
                    d = m.dict()
                except Exception:
                    continue
                cap = d.get("caption_text") or ""
                taken = d.get("taken_at")
                ts = taken.timestamp() if hasattr(taken, "timestamp") else None
                if ts and now - ts <= within_sec and (caption.strip() == cap.strip() or (caption and cap and caption.strip() in cap)):
                    return d
        except Exception as _e:
            return None
        return None

    # ---------------- Uploads ----------------
    def upload_reel_from_path(self, path: Union[str, Path], caption: str = "") -> Optional[Dict[str, Any]]:
        try:
            media = self.client.clip_upload(str(path), caption=caption)
            # Return best-effort dict
            return getattr(media, "dict", lambda: {"pk": getattr(media, "pk", None), "caption_text": caption})()
        except ValidationError as ve:
            # Known instagrapi bug: response validation for clips_metadata may fail (issue #2263)
            print(f"[InstagramUploader] clip_upload validation warning: {ve}")
            # Try to confirm upload by inspecting recent clips
            found = self._confirm_recent_post(caption)
            if found:
                return found
            return None
        except Exception as e:
            print(f"[InstagramUploader] Upload failed (path): {e}")
            return None

    def upload_reel_from_url(self, url: str, caption: str = "") -> Optional[Dict[str, Any]]:
        try:
            p = self._download_to_temp(url)
            return self.upload_reel_from_path(p, caption=caption)
        except Exception as e:
            print(f"[InstagramUploader] Upload failed (url): {e}")
            return None

    # ---------------- Cleanup ----------------
    def cleanup(self) -> None:
        for p in list(self._tmp_files):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        self._tmp_files.clear()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
