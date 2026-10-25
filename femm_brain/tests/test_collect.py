"""Source inventory integrity and bounded non-executing attachment intake."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from femm_brain import collect


class CollectorTests(unittest.TestCase):
    def test_only_official_pages_and_research_files_are_discovered(self):
        self.assertEqual(collect.page_id("https://www.femm.info/wiki/InductionMotorExample"), "inductionmotorexample")
        self.assertEqual(collect.page_id("https://femm.info/doku/doku.php?id=LRKAnalysis#files"), "lrkanalysis")
        for url in ("https://femm.info/doku/doku.php?id=examples&do=login", "https://femm.info.evil/wiki/Start",
                    "file:///tmp/model.fem", "https://user@femm.info/wiki/Start"):
            self.assertIsNone(collect.page_id(url))
        self.assertEqual(collect.asset_url("https://www.femm.info/a/model.fem"), "https://femm.info/a/model.fem")
        self.assertIsNone(collect.asset_url("https://femm.info/a/program.exe"))
        self.assertTrue(collect.asset_url("https://femm.info/a/script.sce.upload"))
        media = collect.asset_url("https://femm.info/doku/lib/exe/detail.php?media=upload:a.pdf&do=show")
        self.assertIn("fetch.php?media=upload%3Aa.pdf", media)

    def test_parser_retains_source_text_and_resolves_attachment_links(self):
        parser = collect.PageParser("https://femm.info/example/")
        parser.feed('<title>Example [FEMM]</title><h1>Motor</h1><script>secret()</script><p>One <b>pole</b></p><a href="rotor.fem">Model file</a>')
        self.assertEqual(parser.title, "Example")
        self.assertIn("One pole", parser.text)
        self.assertNotIn("secret", parser.text)
        self.assertEqual(parser.links, [{"url": "https://femm.info/example/rotor.fem", "title": "Model file"}])

    def test_soft_404_is_missing_not_a_fetched_engineering_example(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(collect, "fetch", return_value=(
                b"<title>Absent</title><p>This topic does not exist yet</p>", "text/html", collect.page_url("absent"))):
            source = collect.collect_page("absent", Path(directory))
            self.assertEqual(source["status"], "missing")
            self.assertTrue(source["sha256"])
            self.assertTrue((Path(directory) / source["text_cache_key"]).is_file())

    def test_network_errors_stay_visible_in_inventory(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(collect, "fetch", side_effect=OSError("fixture failure")):
            source = collect.collect_page("example", Path(directory))
            artifact = collect.collect_attachment("https://femm.info/model.fem", {"example"}, Path(directory))[0]
            self.assertEqual(source["status"], "failed")
            self.assertEqual(artifact["status"], "failed")
            self.assertIn("fixture failure", artifact["error"])

    def test_actual_model_and_script_metadata_are_observed_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            model = collect._artifact(b'[Format] = 4.0\n[Frequency] = 0\n[NumPoints] = 12\n', url="https://femm.info/model.fem", sources={"case"}, name="model.fem", cache=cache)
            self.assertEqual(model["metadata"]["NumPoints"], "12")
            script = collect._artifact(b'-- FEMM 4.2\nmi_addnode(0,0)\nmi_analyze(1)\nos.execute("do not run")', url="https://femm.info/model.lua", sources={"case"}, name="model.lua", cache=cache)
            self.assertEqual(script["metadata"]["functions"], ["mi_addnode", "mi_analyze"])
            self.assertEqual(script["metadata"]["version_mentions"], ["FEMM 4.2"])
            self.assertEqual(script["redistribution_rights"], "not_established")

    def test_html_error_and_fake_pdf_are_not_successful_attachments(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collect, "fetch", return_value=(b"<html>Error</html>", "text/html", "https://femm.info/a.fem")):
                self.assertEqual(collect.collect_attachment("https://femm.info/a.fem", {"case"}, Path(directory))[0]["status"], "failed")
            with patch.object(collect, "fetch", return_value=(b"not a pdf", "application/pdf", "https://femm.info/a.pdf")):
                self.assertEqual(collect.collect_attachment("https://femm.info/a.pdf", {"case"}, Path(directory))[0]["status"], "failed")

    def test_zip_paths_symlinks_and_expansion_budgets_are_rejected(self):
        for name in ("../outside.fem", "/absolute.fem", "C:/outside.fem", "..\\outside.fem"):
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as archive:
                archive.writestr(name, "model")
            with zipfile.ZipFile(io.BytesIO(data.getvalue())) as archive, self.assertRaises(ValueError):
                list(collect._safe_members(archive))
        data = io.BytesIO()
        link = zipfile.ZipInfo("link.fem")
        link.external_attr = 0o120777 << 16
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr(link, "target")
        with zipfile.ZipFile(io.BytesIO(data.getvalue())) as archive, self.assertRaises(ValueError):
            list(collect._safe_members(archive))
        with patch.object(collect, "MAX_ARCHIVE_BYTES", 2):
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as archive:
                archive.writestr("big.fem", "123")
            with zipfile.ZipFile(io.BytesIO(data.getvalue())) as archive, self.assertRaises(ValueError):
                list(collect._safe_members(archive))

    def test_archive_members_are_content_addressed_and_keep_parent_evidence(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("models/a.fem", "[Format] = 4.0")
            archive.writestr("run.exe", "binary is not collected")
        with tempfile.TemporaryDirectory() as directory, patch.object(collect, "fetch", return_value=(data.getvalue(), "application/zip", "https://femm.info/a.zip")):
            records = collect.collect_attachment("https://femm.info/a.zip", {"case"}, Path(directory))
            self.assertEqual(len(records), 2)
            model = records[1]
            self.assertEqual(model["archive_id"], records[0]["id"])
            self.assertEqual(model["archive_member"], "models/a.fem")
            self.assertTrue((Path(directory) / model["cache_key"]).is_file())
            self.assertFalse((Path(directory) / "models").exists())

    def test_invalid_crawl_limits_fail_before_network(self):
        for kwargs in ({"max_pages": 0}, {"max_attachments": 401}, {"depth": 5}, {"timeout": 0}, {"workers": 8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collect.crawl(**kwargs)


if __name__ == "__main__":
    unittest.main()
