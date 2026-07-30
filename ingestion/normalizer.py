"""
HTML → clean text normalizer for problem statements.

Handles:
  - Extracting the problem statement div from a full contest page
  - Converting HTML to clean text while preserving LaTeX/math notation
  - Extracting time_limit_ms and memory_limit_kb from the page header
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString


@dataclass
class ParsedProblemHTML:
    """Result of parsing a Codeforces problem page."""
    statement_html: str        # raw HTML of the statement div
    statement_text: str        # cleaned text with LaTeX preserved
    time_limit_ms: int | None = None
    memory_limit_kb: int | None = None


def extract_problem_html(full_page_html: str) -> ParsedProblemHTML:
    """
    Parse a Codeforces contest problem page and extract:
      - The problem statement div
      - Time and memory limits from the header
    """
    soup = BeautifulSoup(full_page_html, "lxml")

    # ── Extract time/memory limits ──────────────────────────────────────
    time_limit_ms = None
    memory_limit_kb = None

    time_div = soup.find("div", class_="time-limit")
    if time_div:
        text = time_div.get_text()
        m = re.search(r"(\d+(?:\.\d+)?)\s*second", text)
        if m:
            time_limit_ms = int(float(m.group(1)) * 1000)

    mem_div = soup.find("div", class_="memory-limit")
    if mem_div:
        text = mem_div.get_text()
        m = re.search(r"(\d+)\s*megabyte", text)
        if m:
            memory_limit_kb = int(m.group(1)) * 1024

    # ── Extract problem statement ───────────────────────────────────────
    statement_div = soup.find("div", class_="problem-statement")
    if statement_div is None:
        # Fallback: try to find any large text block
        statement_div = soup.find("div", class_="ttypography")

    statement_html = str(statement_div) if statement_div else ""
    statement_text = html_to_text_preserve_latex(statement_html)

    return ParsedProblemHTML(
        statement_html=statement_html,
        statement_text=statement_text,
        time_limit_ms=time_limit_ms,
        memory_limit_kb=memory_limit_kb,
    )


def html_to_text_preserve_latex(html: str) -> str:
    """
    Convert HTML to clean text while preserving:
      - LaTeX/MathJax expressions ($...$, $$...$$, \(...\), \[...\])
      - Structural formatting (paragraphs, lists, headers)

    This is intentionally NOT a full markdown converter — we just need
    clean, readable text that retains the math that carries signal for
    the extraction step.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Remove script/style tags but NOT MathJax script tags
    for tag in soup.find_all(["style"]):
        tag.decompose()
    for tag in soup.find_all("script"):
        if "mathjax" not in (tag.get("src", "") or "").lower():
            tag.decompose()

    # Process the tree
    lines: list[str] = []
    _walk(soup, lines)

    text = "\n".join(lines)

    # Clean up excessive whitespace while preserving paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text


def _walk(element, lines: list[str], depth: int = 0):
    """Recursively walk the DOM tree and collect text."""
    if isinstance(element, NavigableString):
        text = str(element).strip()
        if text:
            if lines:
                lines[-1] += " " + text
            else:
                lines.append(text)
        return

    tag_name = getattr(element, "name", None)

    # Block-level elements get their own line
    block_tags = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr", "li", "tr"}

    if tag_name in block_tags:
        lines.append("")

    # Headers get a prefix
    if tag_name and tag_name.startswith("h") and len(tag_name) == 2:
        lines.append("")

    # Handle <span class="tex-span"> (Codeforces LaTeX)
    if tag_name == "span" and "tex-span" in (element.get("class") or []):
        latex = element.get_text()
        if lines:
            lines[-1] += f" ${latex}$"
        else:
            lines.append(f"${latex}$")
        return

    # Handle <span class="tex-font-style-tt"> (monospace / code)
    if tag_name == "span" and "tex-font-style-tt" in (element.get("class") or []):
        code = element.get_text()
        if lines:
            lines[-1] += f" `{code}`"
        else:
            lines.append(f"`{code}`")
        return

    # Handle list items
    if tag_name == "li":
        lines.append("  • ")

    # Recurse into children
    for child in element.children:
        _walk(child, lines, depth + 1)

    # Add blank line after block elements
    if tag_name in {"p", "div"} and lines and lines[-1]:
        lines.append("")


def normalize_statement(raw_html: str) -> str:
    """
    Convenience function: full pipeline from raw HTML to clean text.
    Used when we have just the statement HTML (not the full page).
    """
    return html_to_text_preserve_latex(raw_html)
