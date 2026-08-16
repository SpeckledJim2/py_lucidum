from __future__ import annotations

import html
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


def _github_heading_fragment(heading_markup: str) -> str:
    """Return the GitHub fragment for the heading forms used in the README."""

    leading_empty_image = re.match(
        r'^<img\b(?=[^>]*\balt="")[^>]*>\s*',
        heading_markup,
        flags=re.IGNORECASE,
    )
    visible_text = re.sub(r"<[^>]+>", "", heading_markup)
    visible_text = html.unescape(visible_text)
    visible_text = re.sub(r"[`*_~]", "", visible_text)
    slug = re.sub(r"[^\w\s-]", "", visible_text.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    prefix = "-" if leading_empty_image else ""
    return f"#{prefix}{slug}"


class DocumentationTests(unittest.TestCase):
    def test_readme_internal_fragments_match_github_headings(self) -> None:
        source = README_PATH.read_text(encoding="utf-8")
        headings = re.findall(r"^#{1,6}\s+(.+?)\s*$", source, flags=re.MULTILINE)
        heading_fragments = {_github_heading_fragment(heading) for heading in headings}

        markdown_fragments = re.findall(r"\]\((#[^)]+)\)", source)
        html_fragments = re.findall(r'href="(#[^"]+)"', source)
        linked_fragments = set(markdown_fragments + html_fragments)
        missing = sorted(linked_fragments - heading_fragments)

        self.assertEqual(
            missing,
            [],
            "README links do not match GitHub's rendered heading fragments",
        )


if __name__ == "__main__":
    unittest.main()
