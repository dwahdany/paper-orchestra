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

"""The rule about sources and anonymity that ends every authoring prompt.

It explains the requirement and its reason instead of issuing absolute orders: the
harness appends the memory block right after it, and a rule that claimed to outrank
everything else would tell the model to distrust the memory it is then handed.
"""

UNIVERSAL_NO_LEAKAGE_PROMPT = """
---
### Sources and anonymity

This pipeline is evaluated on how well it reconstructs a paper from its inputs alone, and the work may already exist as a published paper. The paper is therefore written only from what this run provides: the idea, the experimental log, the provided or generated figures, the verified bibliography, the template and guidelines, and the records of them that this run keeps in its memory. Do not use anything you recall about this work, its published version, its authors or its results. If a detail of this work (a number, a design choice, an experiment, a claim) is not in this run's materials, it stays out of the paper, even when you believe it to be true.

Your general writing skill and general background knowledge are welcome wherever they help to explain the provided material, for example to define a standard term or to motivate a common design choice. Lessons in memory serve the same end: they guide how to write (LaTeX and template pitfalls, venue rules, which kinds of revision helped), while what the paper claims comes from this run's materials.

The manuscript is anonymous, as double-blind review requires: it gives no author names, affiliations or e-mail addresses for the paper itself, has no acknowledgements, and contains nothing else that points to who wrote it. Do not guess who the authors are. The authors of cited papers are named as usual.
---
"""
