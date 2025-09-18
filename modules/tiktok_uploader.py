from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json, time, tempfile

try:
    # pip install tiktok-uploader>=1.1.4
    from tiktok_uploader.upload import upload_video
except Exception:
    upload_video = None


class TikTokUploader:
    """
    Cookies-based TikTok uploader wrapper.
    Supports:
      - connect_from_local_browsers()    # auto-fetch cookies from Chrome/Edge/Firefox/Brave/Opera
      - connect_with_cookies_file(path)  # cookies.txt (Netscape)
      - connect_with_sessionid(value)    # just the 'sessionid' cookie
      - upload_reel_from_path(path, caption=...)
    """

    def __init__(self, *, session_path: Optional[Path] = None, headless: bool = False, browser: str = "chrome"):
        self.session_path = Path(session_path) if session_path else None
        self.cookies_list: Optional[List[Dict[str, Any]]] = None
        self.cookies_file: Optional[Path] = None
        self.headless = headless      # default to visible to avoid site overlays & anti-bot quirks
        self.browser = browser or "chrome"

        if self.session_path and self.session_path.exists():
            try:
                saved = json.loads(self.session_path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    cf = saved.get("cookies_file")
                    self.cookies_file = Path(cf) if cf else None
                    self.cookies_list = saved.get("cookies_list")
            except Exception:
                pass

    # ---------------- session helpers ----------------
    def _save_session(self) -> None:
        if not self.session_path:
            return
        try:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "cookies_file": str(self.cookies_file) if self.cookies_file else None,
                "cookies_list": self.cookies_list,
                "headless": self.headless,
                "browser": self.browser,
            }
            self.session_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def is_connected(self) -> bool:
        return bool(self.cookies_file or self.cookies_list)

    # ---------------- connect methods ----------------
    def connect_with_cookies_file(self, cookies_txt_path: Union[str, Path]) -> bool:
        p = Path(cookies_txt_path)
        if not p.exists():
            return False
        if self.session_path:
            try:
                stable = self.session_path.with_suffix(".cookies.txt")
                stable.parent.mkdir(parents=True, exist_ok=True)
                stable.write_bytes(p.read_bytes())
                self.cookies_file = stable
                self.cookies_list = None
                self._save_session()
                return True
            except Exception:
                return False
        else:
            self.cookies_file = p
            self.cookies_list = None
            return True

    def connect_with_sessionid(self, sessionid: str) -> bool:
        sessionid = (sessionid or "").strip()
        if not sessionid:
            return False
        self.cookies_list = [{
            "name": "sessionid",
            "value": sessionid,
            "domain": ".tiktok.com",
            "path": "/",
        }]
        self.cookies_file = None
        self._save_session()
        return True

    def connect_from_local_browsers(self) -> bool:
        """Auto-fetch TikTok cookies directly from installed browsers via browser-cookie3."""
        try:
            import browser_cookie3  # pip install browser-cookie3
        except Exception:
            return False

        jars = []
        for getter_name in ["chrome", "edge", "firefox", "brave", "opera"]:
            getter = getattr(browser_cookie3, getter_name, None)
            if getter is None:
                continue
            try:
                jars.append(getter(domain_name="tiktok.com"))
            except Exception:
                pass

        cookies = []
        for jar in jars:
            try:
                cookies.extend(list(jar))
            except Exception:
                pass

        if not cookies:
            return False

        clist: List[Dict[str, Any]] = []
        have_session = False
        for c in cookies:
            name = getattr(c, "name", None)
            value = getattr(c, "value", None)
            domain = getattr(c, "domain", ".tiktok.com")
            path = getattr(c, "path", "/")
            if not name or value is None:
                continue
            clist.append({"name": name, "value": value, "domain": domain, "path": path})
            if name == "sessionid" and value:
                have_session = True

        if not have_session:
            return False

        self.cookies_list = clist
        self.cookies_file = None
        self._save_session()
        return True

    # ---------------- upload ----------------
    def _chrome_options(self):
        """Conservative Chrome options that reduce overlays/automation hints."""
        try:
            from selenium.webdriver.chrome.options import Options
        except Exception:
            return None  # If selenium not present, uploader will still try
        opts = Options()
        opts.add_argument("start-maximized")
        opts.add_argument("--disable-notifications")
        opts.add_argument("--disable-infobars")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        # reduce detection surface
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        return opts

    def upload_reel_from_path(self, video_path: Path, *, caption: str, **kwargs) -> bool:
        if upload_video is None:
            raise RuntimeError("tiktok-uploader is not installed. Add 'tiktok-uploader>=1.1.4' to requirements and pip install.")
        video_path = Path(video_path)
        if not video_path.exists():
            return False
        if not self.is_connected():
            return False

        def _try_once(window_size: str | None = None) -> bool:
            extra = dict(
                description=caption or "",
                headless=False,                 # visible run is more reliable for TikTok
                browser="chrome",
            )
            opts = self._chrome_options()
            if opts is not None:
                if window_size:
                    opts.add_argument(f"--window-size={window_size}")
                extra["options"] = opts

            try:
                if self.cookies_file:
                    upload_video(str(video_path), cookies=str(self.cookies_file), **extra, **kwargs)
                else:
                    upload_video(str(video_path), cookies_list=self.cookies_list, **extra, **kwargs)
                time.sleep(2.0)
                return True
            except Exception:
                return False

        # First attempt (maximized)
        if _try_once():
            return True
        # Second attempt with explicit window size — helps when footers overlap inputs
        time.sleep(2.0)
        return _try_once("1440,1000")

    def upload_reel_from_url(self, url: str, *, caption: str, temp_dir: Optional[Path] = None) -> bool:
        try:
            import requests
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            tmp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="tt_temp_"))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_dir / "video.mp4"
            tmp.write_bytes(r.content)
            ok = self.upload_reel_from_path(tmp, caption=caption)
            try:
                tmp.unlink()
            except Exception:
                pass
            return ok
        except Exception:
            return False

    def cleanup(self) -> None:
        pass
