"""Real FastAPI routes and temporary assets; no lifespan, device or audio."""
import hashlib
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient

_IMPORT = tempfile.TemporaryDirectory(prefix="puppy-panel-assets-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT.name}):
    from . import server
from .panel_assets import PanelAssets, SCRIPT_NAMES


class Sources(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.urls = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "script" and "src" in dict(attrs):
            self.urls.append(dict(attrs)["src"])


class PanelAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-panel-assets-")
        self.directory = Path(self.temp.name)
        source = Path(__file__).parent
        for name in ("panel.html", *SCRIPT_NAMES):
            shutil.copyfile(source / name, self.directory / name)
        self.assets = PanelAssets(self.directory)
        self.patcher = patch.object(server, "PanelAssets", return_value=self.assets)
        self.patcher.start()
        # TestClient without its context manager never starts gateway lifespan.
        self.client = TestClient(server.app)

    def tearDown(self):
        self.client.close()
        self.patcher.stop()
        self.temp.cleanup()

    def urls(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        return {Path(urlsplit(url).path).name: url for url in Sources(response.text).urls}

    def test_page_versions_match_all_eight_actual_script_responses(self):
        page = self.client.get("/")
        self.assertEqual(page.headers["cache-control"], "no-store")
        urls = self.urls()
        self.assertEqual(set(urls), SCRIPT_NAMES)
        for name, url in urls.items():
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, (self.directory / name).read_bytes())
            version = hashlib.sha256(response.content).hexdigest()
            self.assertEqual(parse_qs(urlsplit(url).query), {"v": [version]})
            self.assertEqual(response.headers["etag"], '"' + version + '"')
            self.assertEqual(response.headers["cache-control"], "no-cache, must-revalidate")
            self.assertIn("application/javascript", response.headers["content-type"])

    def test_unchanged_get_validators_return_empty_304_and_stale_ones_return_body(self):
        url = self.urls()["learning_panel.js"]
        original = self.client.get(url)
        etag = original.headers["etag"]
        for validator in (etag, "W/" + etag, '"old", W/' + etag, "*"):
            response = self.client.get(url, headers={"If-None-Match": validator})
            self.assertEqual(response.status_code, 304)
            self.assertEqual(response.content, b"")
            self.assertEqual(response.headers["etag"], etag)
        for headers in ({"If-None-Match": '"old"'},
                        {"If-Modified-Since": "Thu, 31 Dec 2099 23:59:59 GMT"}):
            response = self.client.get(url, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, original.content)

    def test_equal_size_and_mtime_upgrade_changes_url_and_old_validator_cannot_304(self):
        target = self.directory / "learning_panel.js"
        target.write_bytes(target.read_bytes() + b"\n// release A\n")
        before_urls = self.urls()
        before = self.client.get(before_urls[target.name])
        stat = target.stat()
        target.write_bytes(target.read_bytes().replace(b"// release A", b"// release B"))
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(target.stat().st_size, stat.st_size)
        self.assertEqual(target.stat().st_mtime_ns, stat.st_mtime_ns)
        after_urls = self.urls()
        self.assertNotEqual(before_urls[target.name], after_urls[target.name])
        for name in SCRIPT_NAMES - {target.name}:
            self.assertEqual(before_urls[name], after_urls[name])
        # Includes stale old version URL and legacy unversioned URL. Neither is
        # permitted to return 304 for old bytes following deployment.
        for url in (before_urls[target.name], after_urls[target.name], "/" + target.name):
            after = self.client.get(url, headers={"If-None-Match": before.headers["etag"]})
            self.assertEqual(after.status_code, 200)
            self.assertEqual(after.content, target.read_bytes())
            self.assertNotEqual(after.headers["etag"], before.headers["etag"])

    def test_reload_always_gets_current_document_even_with_prior_validators(self):
        original = self.client.get("/")
        target = self.directory / "panel.html"
        target.write_text(target.read_text().replace("<title>Puppy 家长中心</title>", "<title>Puppy 更新面板</title>"), encoding="utf-8")
        response = self.client.get("/", headers={"If-None-Match": "*",
                                  "If-Modified-Since": "Thu, 31 Dec 2099 23:59:59 GMT"})
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.content, original.content)
        self.assertIn("Puppy 更新面板", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_parent_token_is_not_kept_in_browser_storage(self):
        page = self.client.get("/").text
        self.assertIn('id="token" type="password"', page)
        self.assertIn("localStorage.removeItem('puppy-parent-token')", page)
        self.assertNotIn("localStorage.setItem('puppy-parent-token'", page)
        self.assertNotIn("sessionStorage.setItem('puppy-parent-token'", page)

    def test_status_and_motion_stop_buttons_avoid_browser_global_names(self):
        page = self.client.get("/").text
        self.assertIn('onclick="refreshDeviceStatus()"', page)
        self.assertIn('onclick="stopMotion()"', page)
        self.assertIn("function refreshDeviceStatus()", page)
        self.assertIn("function stopMotion()", page)
        self.assertNotIn('onclick="status()"', page)
        self.assertNotIn('onclick="stop()"', page)

    def test_versioned_document_bypasses_preexisting_legacy_url_cache(self):
        legacy_cache = {"/" + name: b"cached old script" for name in SCRIPT_NAMES}
        for url in self.urls().values():
            self.assertNotIn(url, legacy_cache)
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertNotEqual(response.content, b"cached old script")


if __name__ == "__main__":
    unittest.main()
