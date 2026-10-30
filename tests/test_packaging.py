"""What the PyPI project page shows is built from pyproject.toml and README.md, and it only changes when a new
version is uploaded. These tests catch the mistakes that otherwise show up after the upload:

* a relative link or image in README.md, which resolves against pypi.org and 404s there;
* README.md that does not render, or that loses images and links to PyPI's HTML sanitiser;
* a summary that is too long, a classifier PyPI rejects, a missing project link.
"""
import html
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Every package this repository publishes to PyPI: the folder with its pyproject.toml and README.md.
PACKAGES = {"kinescript": ROOT, "kinescript-kokoro": ROOT / "plugins" / "kinescript-kokoro"}
each_package = pytest.mark.parametrize("pkg", list(PACKAGES))

# Link targets that work on PyPI. Everything else (docs/hero.gif, ../x, /x, //host/x, a bare filename) is
# resolved against pypi.org and breaks.
ABSOLUTE = re.compile(r"(?:https?://|mailto:|#)", re.IGNORECASE)

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[`~]*[ \t]*$", re.DOTALL | re.MULTILINE)
_CODE_SPAN = re.compile(r"(`+).+?\1", re.DOTALL)
_INLINE = re.compile(r"\]\(\s*(<[^>\n]*>|[^)\s]*)")  # [text](target "title") and ![alt](target)
_REFDEF = re.compile(r"^ {0,3}\[[^\]\n]+\]:[ \t]*(<[^>\n]*>|\S+)", re.MULTILINE)  # [label]: target
_ATTR = re.compile(r"""\b(src|href|poster|srcset)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+))""", re.IGNORECASE)


def _prose(text):
    """The Markdown without fenced blocks and code spans, where `docs/x.png` is only text."""
    return _CODE_SPAN.sub("", _FENCE.sub("", text))


def link_targets(text):
    """Every link or image target in Markdown source: inline, reference definitions and HTML attributes."""
    text = _prose(text)
    targets = [t.strip("<>") for t in _INLINE.findall(text)]
    targets += [t.strip("<>") for t in _REFDEF.findall(text)]
    for name, dq, sq, bare in _ATTR.findall(text):
        value = dq or sq or bare
        if name.lower() == "srcset":
            targets += [c.split()[0] for c in value.split(",") if c.strip()]
        else:
            targets.append(value)
    return [t for t in targets if t]


def relative_targets(text):
    return [t for t in link_targets(text) if not ABSOLUTE.match(t)]


def _readme(pkg="kinescript"):
    return (PACKAGES[pkg] / "README.md").read_text(encoding="utf-8")


# --- README.md ---------------------------------------------------------------------------------------------

@each_package
def test_readme_has_no_relative_links_or_images(pkg):
    bad = relative_targets(_readme(pkg))
    assert not bad, (
        "README.md is the PyPI long description, so a relative target resolves against pypi.org and breaks, "
        "and PyPI drops any other scheme (data:, javascript:). "
        "Use an absolute https URL (images: https://raw.githubusercontent.com/drknowhow/kinescript/main/<path>, "
        f"links: the GitHub or Pages URL). Found: {bad}"
    )


REPO_FILE = re.compile(
    r"https://(?:raw\.githubusercontent\.com/drknowhow/kinescript/main"
    r"|github\.com/drknowhow/kinescript/(?:blob|tree)/main)/([^#?]+)",
    re.IGNORECASE,
)


@each_package
def test_readme_points_at_files_that_exist_in_the_repo(pkg):
    # The README's images are served from the main branch. PyPI keeps the README of every release forever, so a
    # file that is renamed or deleted breaks the page of every release that links to it.
    hits = [m for m in map(REPO_FILE.match, link_targets(_readme(pkg))) if m]
    missing = [m.group(0) for m in hits if not (ROOT / unquote(m.group(1))).exists()]
    assert not missing, f"README.md links to repo files that do not exist: {missing}"


def test_relative_target_checker():
    sample = (
        "![logo](brand/logo.png) [docs](docs/x.md) [ok](https://example.com/a_(b) \"t\") [anchor](#install)\n"
        "[mail](mailto:a@b.c) [up](../x) [root](/x) [proto](//host/x)\n"
        '<img src="docs/hero.gif" width="720"> <a href=\'LICENSE\'>l</a> <img src="https://example.com/i.png">\n'
        "[ref]: docs/ref.md\n[ref2]: <https://example.com/r>\n"
        "`![x](in/code.png)` and ``[y](in/span.md)``\n"
        "```json\n{\"src\": \"x\"} ![z](in/fence.png)\n```\n"
        "~~~\n[w](in/tilde.md)\n~~~\n"
    )
    assert relative_targets(sample) == [
        "brand/logo.png", "docs/x.md", "../x", "/x", "//host/x", "docs/ref.md", "docs/hero.gif", "LICENSE",
    ]


def _render(text):
    rm = pytest.importorskip("readme_renderer.markdown")
    if not getattr(rm, "variants", True):
        pytest.skip("readme_renderer has no Markdown backend; install readme_renderer[md]")
    return rm.render(text)


@each_package
def test_readme_renders_the_way_pypi_renders_it(pkg):
    text = _readme(pkg)
    out = _render(text)
    assert out, "readme_renderer returned nothing: PyPI would show the plain README text instead of HTML"
    out = unquote(html.unescape(out))  # comrak percent-encodes some characters in URLs
    # PyPI's sanitiser silently drops tags it does not allow (<video>, <source src>, <iframe>, <style>), and
    # drops links and images with other schemes. Everything absolute in the source must still be there.
    lost = [t for t in link_targets(text) if ABSOLUTE.match(t) and not t.startswith("#") and unquote(t) not in out]
    assert not lost, f"PyPI's sanitiser would drop these links or images from the page: {lost}"


# --- pyproject.toml ----------------------------------------------------------------------------------------

def _project(pkg="kinescript"):
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads((PACKAGES[pkg] / "pyproject.toml").read_text(encoding="utf-8"))["project"]


@each_package
def test_summary_is_one_short_plain_line(pkg):
    summary = _project(pkg)["description"]
    assert 20 <= len(summary) <= 120, f"{len(summary)} characters: {summary!r}"
    assert "\n" not in summary and not re.search(r"[`*\[\]<>]", summary), "the summary is plain text, not Markdown"
    assert not summary.endswith(" "), summary


@each_package
def test_classifiers_are_all_valid(pkg):
    trove = pytest.importorskip("trove_classifiers")
    listed = _project(pkg)["classifiers"]
    unknown = [c for c in listed if c not in trove.classifiers]
    deprecated = [c for c in listed if c in trove.deprecated_classifiers]
    assert not unknown, f"PyPI rejects the upload for classifiers it does not know: {unknown}"
    assert not deprecated, f"deprecated classifiers: {deprecated}"
    assert len(set(listed)) == len(listed), "duplicate classifier"
    # A license expression is declared in `license`; PEP 639 forbids the License classifiers next to it.
    assert not [c for c in listed if c.startswith("License ::")]


@each_package
def test_keywords_are_unique(pkg):
    words = [k.lower() for k in _project(pkg)["keywords"]]
    assert words and len(set(words)) == len(words)


@each_package
def test_project_links_are_complete_and_verifiable(pkg):
    urls = _project(pkg)["urls"]
    # These labels get an icon on PyPI (see docs.pypi.org/project_metadata, "Link icons").
    for label in ("Homepage", "Documentation", "Source", "Issues", "Changelog"):
        assert label in urls, f"missing project link {label!r}"
    for label, url in urls.items():
        assert urlparse(url).scheme == "https", f"{label}: {url}"
    # Trusted publishing from drknowhow/kinescript marks a URL "verified" only under these two prefixes.
    verified = ("https://github.com/drknowhow/kinescript", "https://drknowhow.github.io/kinescript")
    for label in ("Homepage", "Documentation", "Source", "Issues", "Changelog"):
        assert urls[label].lower().startswith(verified), f"{label} would show as unverified on PyPI: {urls[label]}"
