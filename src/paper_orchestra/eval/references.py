# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-based extraction of a paper's reference list and of citation titles.

Used by the citation F1 autorater.  Failures raise: an empty string is only
returned when the model states that the paper has no reference list.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.eval import RetryPolicy, call_with_retries, resolve_client
from paper_orchestra.llm.base import LLMClient
from paper_orchestra.utils.pdf import load_paper

log = logging.getLogger(__name__)

#: What the model answers when the paper text contains no reference list.
NO_REFERENCES = "NO_REFERENCES"

_REFERENCE_MARKER = re.compile(r"\[\d+\]")

REFERENCE_EXTRACTION_SYSTEM_PROMPT = f"""You are a specialized academic data extraction engine. Your task is to extract the "References" or "Bibliography" section from the raw text of a research paper and return it in a specific single-line string format.

### Input data
You will receive raw text extracted from a PDF inside <paper_text> tags. This text may contain:
- Noise (headers, footers, page numbers).
- Arbitrary line breaks (sentences broken across lines).
- The body of the paper followed by the references.

### Instructions
1. **Locate:** Find the start of the "References" or "Bibliography" section. Ignore all text preceding this section.
2. **Extract:** Identify individual reference entries. These typically start with bracketed numbers (e.g., [1], [2]) or bare numbers (1., 2.).
3. **Clean:**
   - Merge multi-line citations into a single line.
   - Remove any page numbers or running headers that interrupt a citation.
4. **Format:** Output the citations as a single continuous string. Ensure every citation starts with its bracketed ID (e.g., `[1]`). If the source text uses the `1.` format, convert it to `[1]`. If the entries are not numbered, number them in order of appearance.

### Output format
Your output is parsed by a program, so it must be a single string containing only the references, formatted exactly as follows:
[1] First citation text [2] Second citation text [3] Third citation text [4] ...

If the paper text contains no reference list at all, output exactly {NO_REFERENCES} instead.

### Constraints
- Do not output JSON, Markdown lists, or XML.
- Do not output the word "References" or "Bibliography" at the start.
- Do not output any conversational text (e.g., "Here are the references").
- Do not change the content/wording of the citation titles or authors, only clean up the whitespace.
"""

TITLE_EXTRACTION_SYSTEM_PROMPT = """You are an expert bibliographic parser. Extract the full research paper title from the messy citation text given inside <citation> tags.

## Rules:
1. Return only the title text. No authors, years, venues, or labels.
2. Handling hyphens:
    - Only remove a hyphen if it clearly splits a single word due to a line break (e.g., "Net- work" -> "Network").
    - Preserve hyphens in compound words (e.g., keep "Text-to-Image", "Zero-shot").
3. Fixing merged words:
    - If spaces are missing between words (common in PDF extraction), split them.
    - Examples: Change "Imagesearch" to "Image search", "Transformerbased" to "Transformer-based" or "Transformer based".
"""


def _non_blank(text: str) -> str | None:
    stripped = text.strip()
    return stripped or None


def _reference_list(text: str) -> str | None:
    """The answer of the extraction prompt: a ``[n]``-numbered list, or ``""`` for none.

    Anything else (an apology, an unnumbered list) is not an answer: treating
    it as "no references" would score a paper 0 on a failed extraction.
    """
    stripped = text.strip()
    if _REFERENCE_MARKER.search(stripped):
        return stripped
    return "" if NO_REFERENCES in stripped else None


def extract_paper_title_from_citation(
    citation_text: str,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> str:
    """Return the paper title contained in one citation string."""
    llm = resolve_client(model_name, client)
    prompt = f"<citation>\n{citation_text}\n</citation>\n\nTitle:"
    response = call_with_retries(
        lambda: llm.generate(
            prompt,
            model=model_name,
            system=TITLE_EXTRACTION_SYSTEM_PROMPT,
            parser=_non_blank,
            effort="low",
            label="eval.title_extraction",
            parse_retries=0,
        ),
        what="title extraction",
        retry=retry,
    )
    return response.parsed


def extract_reference_from_pdf_text(
    paper_text: str,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> str:
    """Return the reference list of a paper as ``"[1] ... [2] ..."``.

    Returns ``""`` when the model reports that the text has no reference list.
    An answer that is neither a numbered list nor that report counts as a
    failed call: it is retried and finally raised as ``LLMParseError``.
    """
    llm = resolve_client(model_name, client)
    prompt = f"<paper_text>\n{paper_text}\n</paper_text>\n\nExtract the reference list."
    response = call_with_retries(
        lambda: llm.generate(
            prompt,
            model=model_name,
            system=REFERENCE_EXTRACTION_SYSTEM_PROMPT,
            parser=_reference_list,
            label="eval.reference_extraction",
            parse_retries=0,
        ),
        what="reference extraction",
        retry=retry,
    )
    return response.parsed


def get_paper_references_from_pdf(
    pdf_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> str:
    """Extract the reference list of the paper at ``pdf_path``.

    Raises :class:`~paper_orchestra.utils.pdf.PdfError` when the PDF has no
    usable text and an ``LLMError`` when the model call keeps failing.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"paper PDF not found: {path}")
    references = extract_reference_from_pdf_text(
        load_paper(path), model_name=model_name, client=client, retry=retry
    )
    if not references:
        log.warning("The model found no reference list in %s.", path)
    return references
