"""Content-versioned parent panel assets; no device or user data access."""
import hashlib
from pathlib import Path
import re

from fastapi.responses import Response


SCRIPT_NAMES = frozenset({
    "identity_panel.js", "learning_panel.js", "patrol_panel.js", "gesture_panel.js",
    "offline_library_panel.js", "gateway_config_panel.js", "routine_panel.js", "action_drafts_panel.js",
})
_SCRIPT_SOURCE = re.compile(r'(<script\b[^>]*\bsrc=["\'])(/([a-z_]+\.js))(?:\?[^"\']*)?(["\'])')


def content_version(content):
    return hashlib.sha256(content).hexdigest()


class PanelAssets:
    def __init__(self, directory):
        self.directory = Path(directory)

    def page(self):
        source = (self.directory / "panel.html").read_text(encoding="utf-8")

        def versioned(match):
            name = match[3]
            if name not in SCRIPT_NAMES:
                return match[0]
            version = content_version((self.directory / name).read_bytes())
            return match[1] + match[2] + "?v=" + version + match[4]

        # Every document request renders current script hashes. Existing browser
        # entries at the old unversioned URL cannot satisfy these new URLs.
        return Response(_SCRIPT_SOURCE.sub(versioned, source), media_type="text/html",
                        headers={"Cache-Control": "no-store"})

    def script(self, name, if_none_match=None):
        if name not in SCRIPT_NAMES:
            raise ValueError("Unknown parent panel script")
        content = (self.directory / name).read_bytes()
        etag = '"' + content_version(content) + '"'
        headers = {"Cache-Control": "no-cache, must-revalidate", "ETag": etag}
        # Hash the bytes we actually send, not file size/mtime. GET validators
        # use weak comparison, including a list of possible cached ETags.
        if if_none_match and any(token.strip() == "*" or token.strip().removeprefix("W/") == etag
                                 for token in if_none_match.split(",")):
            return Response(status_code=304, headers=headers)
        return Response(content, media_type="application/javascript", headers=headers)
