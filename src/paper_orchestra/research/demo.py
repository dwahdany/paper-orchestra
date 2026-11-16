"""A demonstration project, built without any model call.

:func:`build_demo_project` writes a complete small research project the way a real
one comes to be: through :class:`~paper_orchestra.research.project.Project` and
:mod:`paper_orchestra.research.experiments`, never by writing wiki files itself, so
that it stays valid as those evolve.  It exists for looking at the web app, for the
app's tests and for screenshots.

What is real and what is not, because the difference matters in a workspace whose
rule is that nothing is invented:

* The runs are real.  E1 and E2 are executed on this machine while the project is
  built; their run records, result pages, metrics and figure are what those runs
  produced, and the findings quote E1's metrics from its run record.
* Everything an agent or the person would have written is written here by hand:
  topics, questions, hypotheses, plans, findings, decisions, the activity lines.
* The sources are real papers, described in this module's own words.  Their titles,
  authors and links were entered by hand (and compared with arXiv's index when this
  was written); they are labelled as demonstration records, because no bibliographic
  service is asked when the project is built and no abstract is quoted.
* The model cost in ``state.json`` and the token counts behind it in ``usage.json``
  are illustrative; no model was called.

The project says so itself, in its title and on its idea page.
"""

from __future__ import annotations

import contextlib
import random
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from paper_orchestra.llm.base import Usage, UsageTracker
from paper_orchestra.llm.pricing import estimate_cost_usd, price_for
from paper_orchestra.research.model import (
    IDEA_PATH,
    KINDS,
    SCHEMA,
    SECTIONS,
    ProjectError,
    format_stamp,
)
from paper_orchestra.research.project import STATE_FILE, Project
from paper_orchestra.utils.common import write_json

__all__ = ["DEMO_TITLE", "build_demo_project"]

DEMO_TITLE = "Demonstration: does warmup matter most for small batches?"
#: How long ago the story starts.  It ends a few minutes before the project is built.
STORY_SPAN = timedelta(hours=2, minutes=30)
#: What the two rounds are said to have cost in model calls (no model was called).
ILLUSTRATIVE_MODEL_USD = 4.18
#: The project's usage ledger (spec section 2), which the engine prices the model spend from.
USAGE_FILE = "usage.json"

BRIEF = """\
> **This is a demonstration project.** Its pages were written by hand to show a research
> workspace after two rounds. The runs of E1 and E2 are real: they were executed on this
> machine when the project was built, and their result records hold what they produced.
> No model was called. The activity log, its times and the model cost are part of the
> story; the run records show when the runs really happened. The source pages are
> summaries written for the demonstration, not verified records.

Learning-rate warmup is assumed to matter most for large batches. I suspect that with
adaptive optimisers it is small batches that need it, because early second-moment
estimates are noisy.

What I would like to know:

- Is this already known, and where does the large-batch assumption come from?
- Can a small experiment show the effect, or its absence, at batch sizes 16 and 256?
- If it is real, does the warmup change where training ends or only how it starts?

Attached: my notes, an earlier run (`prior_run.csv`, plotted in `loss_curves.png`) and
the Adam paper.
"""

# ---------------------------------------------------------------------------
# The person's attachments
# ---------------------------------------------------------------------------

NOTES = """\
# Notes on warmup (input of the demonstration project)

- Everyone I ask says warmup is for large batches (the linear scaling rule). The runs of
  mine that blow up in the first hundred steps are the small-batch Adam ones, though.
- Hunch: Adam divides by the square root of its second-moment estimate. Early on that
  estimate rests on a handful of gradients. With 16 examples per step and sparse features
  most coordinates have seen one gradient or none.
- `prior_run.csv`: two runs of the sparse logistic-regression toy (4000 examples, 1000
  binary features, 8 active per example), batch size 16, Adam at learning rate 0.15, with
  and without a linear warmup of 100 steps. One seed each. `loss_curves.png` plots them.
- Not tried yet: batch size 256, more seeds, other learning rates.
"""

#: (step, loss without warmup, loss with warmup) of the person's earlier run.
PRIOR_RUN: tuple[tuple[int, float, float], ...] = (
    (0, 0.69, 0.69),
    (20, 0.79, 0.67),
    (40, 0.82, 0.63),
    (60, 0.80, 0.61),
    (80, 0.78, 0.62),
    (100, 0.76, 0.65),
    (150, 0.74, 0.72),
    (200, 0.72, 0.72),
    (300, 0.71, 0.71),
)

PDF_NOTE = (
    "Placeholder for the person's copy of Kingma and Ba (2015), "
    "'Adam: A Method for Stochastic Optimization'.\n"
    "The demonstration project does not redistribute the paper; "
    "see https://arxiv.org/abs/1412.6980."
)


def _write_curves(path: Path) -> None:
    """Plot the prior run: a small line chart drawn directly, without a plotting library."""
    from PIL import Image, ImageDraw

    width, height, margin = 640, 360, 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    low, high, last = 0.55, 0.85, PRIOR_RUN[-1][0]

    def point(step: int, loss: float) -> tuple[float, float]:
        x = margin + (width - 2 * margin) * step / last
        y = height - margin - (height - 2 * margin) * (loss - low) / (high - low)
        return x, y

    draw.rectangle([margin, margin, width - margin, height - margin], outline="#888888")
    for column, colour in ((1, "#b3261e"), (2, "#1f5fa8")):
        draw.line([point(row[0], row[column]) for row in PRIOR_RUN], fill=colour, width=3)
    draw.text((margin, 16), "Batch size 16: training loss by step", fill="black")
    draw.text((margin + 12, margin + 8), "red: no warmup   blue: 100-step warmup", fill="black")
    image.save(path)


def _write_pdf(path: Path) -> None:
    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_text((72, 96), PDF_NOTE, fontsize=11)
    document.save(path)
    document.close()


def _write_attachments(folder: Path) -> list[Path]:
    """The four files the person attached, written to ``folder``."""
    notes, curves = folder / "notes.md", folder / "loss_curves.png"
    table, paper = folder / "prior_run.csv", folder / "kingma2015.pdf"
    notes.write_text(NOTES, encoding="utf-8")
    rows = ["step,loss_no_warmup,loss_warmup", *(f"{s},{a:.2f},{b:.2f}" for s, a, b in PRIOR_RUN)]
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _write_curves(curves)
    _write_pdf(paper)
    return [notes, table, curves, paper]


# ---------------------------------------------------------------------------
# Sources: real papers, described in this module's words
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    key: str
    title: str
    authors: tuple[str, ...]
    year: int
    venue: str
    link: str
    #: What the paper reports, in one or two sentences written for the demonstration.
    reports: str


SOURCES: tuple[_Source, ...] = (
    _Source(
        "Kingma2015Adam",
        "Adam: A Method for Stochastic Optimization",
        ("Diederik P. Kingma", "Jimmy Ba"),
        2015,
        "ICLR",
        "https://arxiv.org/abs/1412.6980",
        "Introduces Adam, which scales each parameter's step by running estimates of the "
        "first and second moments of its gradient, and corrects both estimates for "
        "starting at zero.",
    ),
    _Source(
        "Liu2020Variance",
        "On the Variance of the Adaptive Learning Rate and Beyond",
        ("Liyuan Liu", "Haoming Jiang", "Pengcheng He", "Weizhu Chen", "Xiaodong Liu",
         "Jianfeng Gao", "Jiawei Han"),
        2020,
        "ICLR",
        "https://arxiv.org/abs/1908.03265",
        "Argues that the adaptive learning rate has a problematically large variance in "
        "the first steps of training, when the second-moment estimate rests on few "
        "gradients. Reads warmup as a variance reduction and proposes RAdam, which "
        "rectifies that variance.",
    ),
    _Source(
        "Goyal2017Accurate",
        "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
        ("Priya Goyal", "Piotr Dollár", "Ross Girshick", "Pieter Noordhuis",
         "Lukasz Wesolowski", "Aapo Kyrola", "Andrew Tulloch", "Yangqing Jia", "Kaiming He"),
        2017,
        "arXiv",
        "https://arxiv.org/abs/1706.02677",
        "Trains ResNet-50 on ImageNet with minibatches of up to 8192 images without loss "
        "of accuracy, using a learning rate scaled linearly with the batch size and a "
        "gradual warmup over the first epochs.",
    ),
    _Source(
        "Gotmare2019Closer",
        "A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and "
        "Distillation",
        ("Akhilesh Gotmare", "Nitish Shirish Keskar", "Caiming Xiong", "Richard Socher"),
        2019,
        "ICLR",
        "https://arxiv.org/abs/1810.13243",
        "Studies warmup, learning-rate restarts and distillation empirically. Finds that "
        "warmup limits how much the deeper layers change early in training, and that "
        "freezing those layers has a similar effect.",
    ),
    _Source(
        "Vaswani2017Attention",
        "Attention Is All You Need",
        ("Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones",
         "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"),
        2017,
        "NeurIPS",
        "https://arxiv.org/abs/1706.03762",
        "Introduces the Transformer. Trains it with Adam and a learning rate that rises "
        "linearly over the first warmup steps and then decays with the inverse square "
        "root of the step number.",
    ),
    _Source(
        "Ma2021Adequacy",
        "On the Adequacy of Untuned Warmup for Adaptive Optimization",
        ("Jerry Ma", "Denis Yarats"),
        2021,
        "AAAI",
        "https://arxiv.org/abs/1910.04209",
        "Questions the variance explanation behind RAdam, argues that the size of the "
        "early updates is what matters for stability, and shows that simple untuned "
        "warmup schedules for Adam perform about as well as RAdam.",
    ),
    _Source(
        "Smith2018Dont",
        "Don't Decay the Learning Rate, Increase the Batch Size",
        ("Samuel L. Smith", "Pieter-Jan Kindermans", "Chris Ying", "Quoc V. Le"),
        2018,
        "ICLR",
        "https://arxiv.org/abs/1711.00489",
        "Shows that increasing the batch size during training can stand in for decaying "
        "the learning rate, reaching similar test accuracy with fewer parameter updates.",
    ),
    _Source(
        "Keskar2017Large",
        "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima",
        ("Nitish Shirish Keskar", "Dheevatsa Mudigere", "Jorge Nocedal",
         "Mikhail Smelyanskiy", "Ping Tak Peter Tang"),
        2017,
        "ICLR",
        "https://arxiv.org/abs/1609.04836",
        "Observes that large-batch training tends to converge to sharp minimisers that "
        "generalise worse than the flat minimisers small-batch training finds.",
    ),
    _Source(
        "Loshchilov2019Decoupled",
        "Decoupled Weight Decay Regularization",
        ("Ilya Loshchilov", "Frank Hutter"),
        2019,
        "ICLR",
        "https://arxiv.org/abs/1711.05101",
        "Shows that L2 regularisation and weight decay are not equivalent for Adam and "
        "proposes AdamW, which applies the decay separately from the gradient-based "
        "update.",
    ),
    _Source(
        "Reddi2018Convergence",
        "On the Convergence of Adam and Beyond",
        ("Sashank J. Reddi", "Satyen Kale", "Sanjiv Kumar"),
        2018,
        "ICLR",
        "https://arxiv.org/abs/1904.09237",
        "Shows that Adam can fail to converge even on simple convex problems, because its "
        "exponential moving average forgets large past gradients, and proposes AMSGrad "
        "as a fix.",
    ),
    _Source(
        "Xiong2020Layer",
        "On Layer Normalization in the Transformer Architecture",
        ("Ruibin Xiong", "Yunchang Yang", "Di He", "Kai Zheng", "Shuxin Zheng", "Chen Xing",
         "Huishuai Zhang", "Yanyan Lan", "Liwei Wang", "Tie-Yan Liu"),
        2020,
        "ICML",
        "https://arxiv.org/abs/2002.04745",
        "Shows that in the original post-norm Transformer the gradients near the output "
        "are large at initialisation, which makes warmup necessary, and that the pre-norm "
        "variant trains without it.",
    ),
    _Source(
        "Gilmer2022Loss",
        "A Loss Curvature Perspective on Training Instability in Deep Learning",
        ("Justin Gilmer", "Behrooz Ghorbani", "Ankush Garg", "Sneha Kudugunta",
         "Behnam Neyshabur", "David Cardoze", "George Dahl", "Zachary Nado", "Orhan Firat"),
        2022,
        "ICLR",
        "https://arxiv.org/abs/2110.04369",
        "Studies training instabilities through the curvature of the loss. Finds that "
        "warmup, among other techniques, helps by keeping the sharpness low enough for "
        "the learning rate in use.",
    ),
    _Source(
        "Kalra2024Why",
        "Why Warmup the Learning Rate? Underlying Mechanisms and Improvements",
        ("Dayal Singh Kalra", "Maissam Barkeshli"),
        2024,
        "NeurIPS",
        "https://arxiv.org/abs/2406.09405",
        "Examines why warmup helps for SGD and Adam. Attributes the benefit mainly to "
        "letting the network tolerate a larger target learning rate, and proposes a "
        "different initialisation of Adam's second moment that has a similar effect.",
    ),
    _Source(
        "He2019Bag",
        "Bag of Tricks for Image Classification with Convolutional Neural Networks",
        ("Tong He", "Zhi Zhang", "Hang Zhang", "Zhongyue Zhang", "Junyuan Xie", "Mu Li"),
        2019,
        "CVPR",
        "https://arxiv.org/abs/1812.01187",
        "Collects training refinements for image classification, among them learning-rate "
        "warmup, cosine decay and label smoothing, and reports their combined effect on "
        "ResNet-50 and other models.",
    ),
)  # fmt: skip


def _credit(source: _Source) -> str:
    """``Liu et al. (2020)``, ``Ma and Yarats (2021)``: how a source is named in one line."""
    surnames = [author.split()[-1] for author in source.authors]
    who = " and ".join(surnames) if len(surnames) <= 2 else f"{surnames[0]} et al."
    return f"{who} ({source.year})"


def _source_body(source: _Source) -> str:
    return (
        f"- **Authors:** {', '.join(source.authors)}\n"
        f"- **Year:** {source.year}\n"
        f"- **Venue:** {source.venue}\n"
        f"- **Link:** <{source.link}>\n\n"
        "## What the paper reports\n\n"
        f"{source.reports}\n\n"
        "_Demonstration record. The bibliographic data was entered by hand and the summary "
        "above was written for the demonstration; it is not the paper's abstract. In a "
        "real project this record is written by code from a bibliographic service and "
        "quotes the abstract._\n"
    )


# ---------------------------------------------------------------------------
# Pages the agents and the person would have written
# ---------------------------------------------------------------------------

TOPIC_FILES = "What the attached files contribute"
TOPIC_ACCOUNTS = "Why warmup helps: three accounts"
TOPIC_VARIANCE = "Variance of the adaptive learning rate"
TOPIC_BATCH = "Warmup and batch size: where the large-batch assumption comes from"

FILES_BODY = """\
The person attached four files. This page says what each adds; none of it has been
compared with the literature yet.

- `notes.md`: the hunch in the person's words. Their small-batch Adam runs "blow up in the
  first hundred steps", and the suspected cause is a second-moment estimate that rests on
  a handful of gradients when features are sparse.
- `prior_run.csv`: the training loss of two earlier runs on a sparse logistic-regression
  toy (batch size 16, learning rate 0.15), without warmup and with a linear warmup of 100
  steps, at nine steps between 0 and 300. Without warmup the loss rises from 0.69 to 0.82
  by step 40; with warmup it does not rise.
- `loss_curves.png`: a plot of the same two runs.
- `kingma2015.pdf`: a one-page placeholder for the person's copy of the Adam paper.

What the files do not contain: a run at a large batch size, more than one seed, or the
code that produced the numbers. [[Q1]] asks about the first of these.
"""

ACCOUNTS_OPEN = """\
None of the three says how the effect should depend on the batch size under Adam. The
large-batch results that made warmup standard are for SGD; see
[[topics/warmup-and-batch-size|Warmup and batch size]].
"""

ACCOUNTS_OPEN_PROPOSED = """\
None of the three says how the effect should depend on the batch size under Adam.
[[F1]] now gives one data point: on a sparse toy problem the early spike is there at
batch size 16 and gone at 256, and a 100-step warmup removes it. That is what the
variance account predicts, and the update-size account does not rule it out. The
large-batch results that made warmup standard are for SGD; see
[[topics/warmup-and-batch-size|Warmup and batch size]].
"""

ACCOUNTS_BODY = f"""\
Three explanations of why a learning-rate warmup helps are on record. They do not exclude
one another, and they differ in what they imply for small batches.

## The variance account

[[sources/Liu2020Variance|Liu et al. (2020)]] argue that in the first steps Adam's
adaptive learning rate has a very large variance, because the second-moment estimate
rests on only a few gradients. On this reading warmup works as a variance reduction, and
RAdam builds the correction into the optimiser. The estimate is noisier the fewer
examples a step sees, so this account suggests that small batches need warmup more.

## The update-size account

[[sources/Ma2021Adequacy|Ma and Yarats (2021)]] question that explanation. They argue
that what destabilises early training is the size of the updates, and show that simple
untuned warmup schedules do about as well as RAdam. By itself this account says nothing
about the batch size.

## The curvature account

[[sources/Gilmer2022Loss|Gilmer et al. (2022)]] and
[[sources/Kalra2024Why|Kalra and Barkeshli (2024)]] look at the sharpness of the loss:
early in training the loss is too sharp for the target learning rate, and a warmup lets
the optimiser reach a flatter region first.
[[sources/Gotmare2019Closer|Gotmare et al. (2019)]] find, in the same spirit, that warmup
mainly keeps the deeper layers from changing too fast.

## What the accounts leave open

{ACCOUNTS_OPEN}"""

#: What the person adds to the scout's page at the first checkpoint.
ACCOUNTS_MY_READING = """
## My reading

My bet is on the variance account: with 16 examples per step and sparse features, most
coordinates see one gradient or none in the first steps. If that is right, the spike
should shrink smoothly as the batch grows, not switch off at some size.
"""

VARIANCE_BODY = """\
Adam divides each parameter's step by the square root of a running average of its squared
gradients ([[sources/Kingma2015Adam|Kingma and Ba, 2015]]). The average starts at zero and
is corrected for that start, but the correction fixes its expected size, not its noise:
after a handful of steps it is still an average over a handful of gradients.

- [[sources/Liu2020Variance|Liu et al. (2020)]] make this the centre of their explanation
  of warmup: the variance of the adaptive learning rate is largest in the first steps and
  shrinks as gradients accumulate.
- [[sources/Ma2021Adequacy|Ma and Yarats (2021)]] agree that the early updates are the
  problem, but locate it in their size rather than in that variance.
- [[sources/Reddi2018Convergence|Reddi et al. (2018)]] show a different weakness of the
  same running average: it forgets large gradients seen long ago, which can keep Adam
  from converging. That is a late-training effect, not the one this project is after.

Two things make the early estimate worse, and both are in the person's notes: few examples
per step, and sparse features, where most coordinates receive no gradient at all in a
given step. None of the sources above varies the batch size to test this ([[Q2]], [[Q5]]).
"""

BATCH_BODY = """\
The belief that warmup is a large-batch technique goes back to work on SGD.

- [[sources/Goyal2017Accurate|Goyal et al. (2017)]] scale the learning rate linearly with
  the batch size and add a gradual warmup over the first epochs; with both, minibatches of
  8192 images train ResNet-50 as well as small ones. The warmup is there because the
  scaled learning rate is too large for the first steps.
- [[sources/He2019Bag|He et al. (2019)]] list warmup among the standard refinements for
  image classification, in the same large-batch setting.
- [[sources/Keskar2017Large|Keskar et al. (2017)]] and
  [[sources/Smith2018Dont|Smith et al. (2018)]] treat batch size and learning rate as two
  sides of the noise in SGD: larger batches mean less noise, and the learning rate can
  grow with them.
- [[sources/Vaswani2017Attention|Vaswani et al. (2017)]] use a warmup with Adam for a
  different reason, which [[sources/Xiong2020Layer|Xiong et al. (2020)]] trace to where
  layer normalisation sits in the Transformer.

In the SGD work the step is the learning rate times the gradient, so a larger batch with a
larger learning rate takes a larger step. Adam normalises the step, so the argument does
not carry over as it stands. Whether anything takes its place is [[Q4]].
"""

#: (title, summary, body, relations, author) of the questions of round 1.
QUESTIONS: tuple[tuple[str, str, str, dict[str, list[str]], str], ...] = (
    (
        "Does the early loss spike in the attached runs go away at a large batch size?",
        "The attached runs are all at batch size 16.",
        "Both runs in `prior_run.csv` use batch size 16. The person's idea is about the "
        "contrast with large batches, and nothing attached shows a large batch.\n",
        {"about": ["idea", "topics/attached-files"]},
        "agent:intake",
    ),
    (
        "Does the variance account predict how the need for warmup changes with batch size?",
        "Liu et al. tie the problem to the number of gradients seen, not to the batch size.",
        "[[sources/Liu2020Variance|Liu et al. (2020)]] derive the variance from the number "
        "of steps taken. A larger batch averages more examples per step, which should "
        "lower the noise in each squared gradient, but the paper does not vary it.\n",
        {"about": ["topics/adaptive-learning-rate-variance"], "cites": ["Liu2020Variance"]},
        "agent:scout",
    ),
    (
        "Does warmup under Adam change where training ends, or only how it starts?",
        "The accounts agree on the start and are silent or split on the end.",
        "The variance and update-size accounts are about the first steps. The curvature "
        "account says a warmup lets training reach a region where a larger learning rate "
        "is stable, which would also show at the end.\n",
        {"about": ["topics/why-warmup-helps"]},
        "agent:scout",
    ),
    (
        "Is there an argument for Adam that takes the place of the linear scaling rule?",
        "The large-batch case for warmup is made for SGD.",
        "The linear scaling rule ties the learning rate to the batch size for SGD "
        "([[sources/Goyal2017Accurate|Goyal et al., 2017]]). No source on record gives a "
        "counterpart for an optimiser that normalises its step.\n",
        {
            "about": ["topics/warmup-and-batch-size"],
            "cites": ["Goyal2017Accurate"],
        },
        "agent:scout",
    ),
    (
        "How much of the effect needs sparse gradients?",
        "The person's toy has sparse features; most of the literature does not.",
        "With sparse features most coordinates get no gradient in a given step, so their "
        "second-moment estimates stay close to their first few values for longer. A dense "
        "problem might show a much weaker effect, or none.\n",
        {"about": ["topics/adaptive-learning-rate-variance"]},
        "agent:scout",
    ),
)

QUESTION_NEXT = (
    "Is the full learning rate simply too high for batch size 16?",
    "After the warmup ends, the batch-16 run climbs back to the level of the run without one.",
    "[[F1]] shows the warmup run losing its early advantage once the learning rate reaches "
    "its full value. If a lower learning rate removes the spike and ends lower, the warmup "
    "was hiding a badly chosen learning rate rather than fixing the start.\n",
    {"about": ["idea"], "follows": ["F1"]},
)


def _hypothesis(statement: str, rationale: str, predictions: str, test: str, risks: str) -> str:
    return (
        f"## Statement\n\n{statement}\n\n## Rationale\n\n{rationale}\n\n"
        f"## Predictions\n\n{predictions}\n\n## How to test\n\n{test}\n\n## Risks\n\n{risks}\n"
    )


#: (title, summary, body, relations) of the hypotheses of round 1, H1 to H4.
HYPOTHESES: tuple[tuple[str, str, str, dict[str, list[str]]], ...] = (
    (
        "Without warmup, Adam's training loss spikes early at batch size 16 and not at 256",
        "The early spike is a small-batch effect.",
        _hypothesis(
            "With Adam and no warmup, the training loss rises above its starting value "
            "within the first 100 steps at batch size 16 and does not at batch size 256, on "
            "the same problem and at the same learning rate.",
            "On the variance account ([[topics/adaptive-learning-rate-variance|"
            "Variance of the adaptive learning rate]]) the early second-moment estimate is "
            "noisier the fewer examples a step sees. The person's earlier run shows the rise "
            "at batch size 16 ([[Q1]]).",
            "- At batch size 16 without warmup the loss, averaged over seeds, exceeds its "
            "initial value somewhere in steps 1 to 100.\n"
            "- At batch size 256 it does not.\n"
            "- The hypothesis is wrong if both batch sizes show the rise, or neither does.",
            "The sparse logistic-regression toy of the earlier run, at both batch sizes, "
            "with a few seeds. It runs in seconds on a laptop.",
            "One problem and one learning rate. A rise at both sizes would point at the "
            "learning rate, not at the batch size.",
        ),
        {"motivated_by": ["Liu2020Variance", "topics/adaptive-learning-rate-variance", "Q1"]},
    ),
    (
        "A linear warmup of 100 steps removes the early spike at batch size 16",
        "A short warmup is enough to remove the spike.",
        _hypothesis(
            "At batch size 16, a learning rate that rises linearly over the first 100 steps "
            "keeps the training loss from rising above its starting value in those steps.",
            "Both the variance account and the update-size account "
            "([[sources/Ma2021Adequacy|Ma and Yarats, 2021]]) expect smaller early steps to "
            "help, for different reasons. The earlier run shows no rise with this warmup.",
            "- With the warmup, the mean loss stays at or below its initial value through "
            "step 100.\n"
            "- The hypothesis is wrong if the rise is merely delayed to within those steps.",
            "The same toy, batch size 16, with and without the warmup, same seeds.",
            "It says nothing about what happens once the warmup is over.",
        ),
        {"motivated_by": ["Ma2021Adequacy", "topics/why-warmup-helps"]},
    ),
    (
        "At batch size 16, warmup lowers the training loss reached after 300 steps",
        "The benefit of warmup lasts beyond the first steps.",
        _hypothesis(
            "At batch size 16 the run with a 100-step warmup has a lower training loss "
            "after 300 steps than the run without.",
            "If a warmup lets training reach a region where the full learning rate is "
            "stable, as the curvature account holds, the gain should still be there after "
            "the warmup is over ([[Q3]]).",
            "- After 300 steps the mean loss with warmup is lower than without.\n"
            "- The hypothesis is wrong if the two are equal or the order is reversed.",
            "The same runs as for H1 and H2, read at step 300.",
            "300 steps may be too few for either run to settle.",
        ),
        {"motivated_by": ["Q3", "topics/why-warmup-helps"]},
    ),
    (
        "With RAdam in place of Adam, warmup makes no difference at any batch size",
        "RAdam already contains the correction a warmup provides.",
        _hypothesis(
            "Replacing Adam by RAdam removes the early spike at batch size 16 without a "
            "warmup, and adding a warmup to RAdam changes nothing.",
            "[[sources/Liu2020Variance|Liu et al. (2020)]] design RAdam to rectify the "
            "variance that warmup is said to reduce.",
            "- RAdam without warmup shows no rise at batch size 16.\n"
            "- RAdam with and without warmup end at the same loss.",
            "The same toy with RAdam implemented next to Adam.",
            "It tests the optimiser of one paper rather than the person's question.",
        ),
        {"motivated_by": ["Liu2020Variance"]},
    ),
)
H4_REJECTION = (
    "Ma and Yarats (2021) already compare RAdam with untuned warmup. I want to stay with "
    "plain Adam until the batch-size question is settled."
)

HYPOTHESIS_NEXT = (
    "Starting Adam's second-moment estimate from a large first batch removes the spike "
    "without a warmup",
    "If the noisy early estimate is the cause, a better first estimate should be the cure.",
    _hypothesis(
        "At batch size 16, initialising the second-moment estimate from the squared gradient "
        "of one batch of 256 examples keeps the training loss from rising above its starting "
        "value in the first 100 steps, with no warmup.",
        "[[F1]] shows the spike at batch size 16 and not at 256. "
        "[[sources/Kalra2024Why|Kalra and Barkeshli (2024)]] propose a different "
        "initialisation of the second moment as a stand-in for warmup, and on the variance "
        "account of [[sources/Liu2020Variance|Liu et al. (2020)]] the first estimate is "
        "where the noise comes from.",
        "- With the estimate initialised from 256 examples, the spike at batch size 16 is "
        "at most a quarter of its size without.\n"
        "- The hypothesis is wrong if the spike is unchanged: the cause is then the step "
        "size, not the estimate.",
        "One more configuration in the toy: batch size 16, no warmup, initialised estimate.",
        "A good first estimate also makes the first steps smaller, so the two accounts are "
        "not cleanly separated by this alone.",
    ),
    {"motivated_by": ["F1", "Kalra2024Why", "Liu2020Variance"], "follows": ["F1"]},
)


def _experiment(objective: str, method: str, metrics: str, resources: str) -> str:
    return (
        f"## Objective\n\n{objective}\n\n## Method\n\n{method}\n\n"
        f"## Metrics and success criteria\n\n{metrics}\n\n## Resources\n\n{resources}\n"
    )


E1_TITLE = "Warmup against no warmup at batch sizes 16 and 256"
E1_BODY = _experiment(
    "Test [[H1]], [[H2]] and [[H3]] in one small run: does the training loss spike early at "
    "batch size 16 and not at 256, does a 100-step warmup remove the spike, and does the "
    "run with warmup end lower?",
    "The toy problem of the person's earlier run (`inputs/prior_run.csv`): logistic "
    "regression on 4000 examples with 1000 sparse binary features, 8 active per example, "
    "weights starting at zero. Adam at learning rate 0.15 for 300 steps, at batch sizes 16 "
    "and 256, each without warmup and with a linear warmup over 100 steps. Three seeds per "
    "configuration; the loss is measured on the full training set before and after every "
    "step and averaged over the seeds. The data seed is fixed.\n\n"
    "Code: `main.py` (the sweep, the metrics, the figure) and `toy.py` (the problem and "
    "the optimiser).",
    "- `spike_<configuration>`: how far the mean loss rises above its initial value within "
    "the first 100 steps (0 if it never does).\n"
    "- `lowest_loss_`, `loss_step100_` and `final_loss_<configuration>`.\n"
    "- [[H1]] holds if the spike is clearly above zero at batch size 16 without warmup and "
    "zero at 256; [[H2]] if it is zero at 16 with warmup; [[H3]] if the final loss at 16 "
    "is lower with warmup than without.",
    "CPU only, a few seconds. Needs numpy and matplotlib. No network.",
)
E2_TITLE = "Size of the early spike at batch sizes from 8 to 4096"
E2_BODY = _experiment(
    "Find where the spike of [[H1]] stops: measure it at every power of two from 8 to "
    "4096, with and without the 100-step warmup.",
    "The same toy and optimiser as [[E1]], 100 steps, two seeds. Batches are drawn without "
    "replacement, so that a large batch never counts an example twice.\n\n"
    "Code: `main.py` and `toy.py`.",
    "- `spike_b<batch>_no_warmup` and `spike_b<batch>_warmup` for every batch size.\n"
    "- [[H1]] gains support if the spike shrinks steadily as the batch grows.",
    "CPU only, a few seconds. Needs numpy and matplotlib. No network.",
)
E3_TITLE = "Does the spike appear in a small Transformer?"
E3_BODY = _experiment(
    "Check [[H1]] and [[H2]] on a model closer to practice than the toy, as the person "
    "asked: a two-layer Transformer on a synthetic sequence task.",
    "A pre-norm Transformer with two layers and 128 hidden units learns to reverse random "
    "token sequences of length 32. AdamW ([[sources/Loshchilov2019Decoupled|Loshchilov and "
    "Hutter, 2019]]) at learning rate 0.001 for 2000 steps, at batch sizes 16 and 256, "
    "each without warmup and with a linear warmup over 200 steps, three seeds. The data "
    "is generated in the script, so nothing is downloaded.\n\n"
    "Code: `main.py`.",
    "- `spike_<configuration>`: the rise of the smoothed training loss above its initial "
    "value within the first 200 steps.\n"
    "- `final_loss_<configuration>` after 2000 steps.\n"
    "- The toy result carries over if the spike is present at batch size 16 without "
    "warmup and absent in the other three configurations.",
    "One T4 GPU on Modal, an estimated 20 minutes and $0.20. Needs torch. No network.",
)
E4_TITLE = "How long does the warmup need to be at batch size 16?"
E4_BODY = _experiment(
    "Follow up on [[F1]] as the person asked: instead of one warmup of 100 steps, sweep "
    "its length and see where the spike of [[H2]] disappears.",
    "The toy and optimiser of [[E1]], batch size 16, learning rate 0.15, 400 steps, three "
    "seeds, with linear warmups of 0, 10, 25, 50, 100 and 200 steps.\n\n"
    "Code: `main.py` and `toy.py`.",
    "- `spike_warmup<length>`, `lowest_loss_warmup<length>` and "
    "`final_loss_warmup<length>` for every length.\n"
    "- [[H2]] is sharpened if the spike falls steadily with the length and reaches zero "
    "at or before 100 steps.",
    "CPU only, a few seconds. Needs numpy and matplotlib. No network.",
)

# ---------------------------------------------------------------------------
# Experiment code (what the designer wrote; E1, E2 and E4 run as they are)
# ---------------------------------------------------------------------------

TOY_CODE = '''\
"""The toy problem shared by this project's script experiments.

Logistic regression on sparse binary features: every example has a few active
features out of many, so with a small batch most coordinates see a gradient from
one example or none.  That is the regime in which Adam's second-moment estimate
is noisiest.  Everything is deterministic given the seeds.
"""

import numpy as np

EXAMPLES, FEATURES, ACTIVE = 4000, 1000, 8
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


def make_data(seed):
    """Indices of the active features of each example, and binary labels."""
    rng = np.random.default_rng(seed)
    active = rng.integers(0, FEATURES, (EXAMPLES, ACTIVE))
    truth = np.where(rng.random(FEATURES) < 0.2, rng.choice([-1.5, 1.5], FEATURES), 0.0)
    labels = (rng.random(EXAMPLES) < 1.0 / (1.0 + np.exp(-truth[active].sum(1)))).astype(float)
    return active, labels


def full_loss(weights, active, labels):
    """Mean logistic loss over the whole training set."""
    logits = weights[active].sum(1)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def train(active, labels, *, batch, warmup, learning_rate, steps, seed, replace=True):
    """Adam from zero weights; returns the full training loss before and after every step.

    ``warmup`` is the number of steps over which the learning rate rises linearly to
    ``learning_rate`` (0: no warmup).
    """
    rng = np.random.default_rng(seed)
    weights = np.zeros(FEATURES)
    first, second = np.zeros(FEATURES), np.zeros(FEATURES)
    curve = [full_loss(weights, active, labels)]
    for step in range(1, steps + 1):
        rows = rng.choice(EXAMPLES, batch, replace=replace)
        logits = weights[active[rows]].sum(1)
        error = (1.0 / (1.0 + np.exp(-logits)) - labels[rows]) / batch
        gradient = np.bincount(
            active[rows].ravel(), weights=np.repeat(error, ACTIVE), minlength=FEATURES
        )
        rate = learning_rate * min(1.0, step / warmup) if warmup else learning_rate
        first = BETA1 * first + (1.0 - BETA1) * gradient
        second = BETA2 * second + (1.0 - BETA2) * gradient**2
        corrected = (first / (1.0 - BETA1**step)) / (
            np.sqrt(second / (1.0 - BETA2**step)) + EPSILON
        )
        weights -= rate * corrected
        curve.append(full_loss(weights, active, labels))
    return np.array(curve)
'''

_SCRIPT_HEAD = """\
import json
import os

# The arrays are small; extra threads would only make the timing noisy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
"""

E1_CODE = (
    '''\
"""E1: Adam with and without a linear warmup, at batch sizes 16 and 256.

Writes metrics.json (the early loss spike, the lowest loss, the loss at step 100 and
the loss after training, per configuration, averaged over the seeds) and
artifacts/loss_curve.png.
"""

'''
    + _SCRIPT_HEAD
    + """\
from toy import make_data, train

DATA_SEED = 0
SEEDS = (1, 2, 3)
BATCH_SIZES = (16, 256)
LEARNING_RATE = 0.15
STEPS = 300
WARMUP_STEPS = 100
# The spike is looked for in the first steps only: a later rise is a different effect.
WINDOW = 100
COLOURS = {16: "#b3261e", 256: "#1f5fa8"}


def main():
    active, labels = make_data(DATA_SEED)
    curves = {}
    for batch in BATCH_SIZES:
        for warmup in (0, WARMUP_STEPS):
            runs = [
                train(active, labels, batch=batch, warmup=warmup,
                      learning_rate=LEARNING_RATE, steps=STEPS, seed=seed)
                for seed in SEEDS
            ]
            curve = np.mean(runs, axis=0)
            curves[(batch, warmup)] = curve
            print(f"batch {batch}, warmup {warmup}: loss {curve[0]:.3f} at the start, "
                  f"highest {curve[: WINDOW + 1].max():.3f} in the first {WINDOW} steps, "
                  f"{curve[-1]:.3f} after {STEPS} steps", flush=True)

    initial = float(next(iter(curves.values()))[0])
    names = {key: f"b{key[0]}_{'warmup' if key[1] else 'no_warmup'}" for key in curves}
    metrics = {"initial_loss": round(initial, 3)}
    for key, curve in curves.items():
        rise = float(curve[: WINDOW + 1].max()) - initial
        metrics[f"spike_{names[key]}"] = round(max(0.0, rise), 3)
    for key, curve in curves.items():
        metrics[f"lowest_loss_{names[key]}"] = round(float(curve.min()), 3)
    for key, curve in curves.items():
        metrics[f"loss_step100_{names[key]}"] = round(float(curve[100]), 3)
    for key, curve in curves.items():
        metrics[f"final_loss_{names[key]}"] = round(float(curve[-1]), 3)
    metrics.update(
        batch_sizes=", ".join(str(batch) for batch in BATCH_SIZES),
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        steps=STEPS,
        seeds=len(SEEDS),
    )
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")

    Path("artifacts").mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for (batch, warmup), curve in curves.items():
        axis.plot(curve, color=COLOURS[batch], linestyle="--" if warmup else "-",
                  linewidth=1.6,
                  label=f"batch {batch}, {'warmup' if warmup else 'no warmup'}")
    axis.axhline(initial, color="0.6", linewidth=0.8)
    axis.axvline(WARMUP_STEPS, color="0.6", linewidth=0.8, linestyle=":")
    axis.set_xlabel("step")
    axis.set_ylabel(f"training loss (full set, mean of {len(SEEDS)} seeds)")
    axis.set_title(f"Adam at learning rate {LEARNING_RATE}: {WARMUP_STEPS}-step warmup or none")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig("artifacts/loss_curve.png", dpi=110)
    print("wrote metrics.json and artifacts/loss_curve.png", flush=True)


if __name__ == "__main__":
    main()
"""
)

#: E2 as the designer wrote it, with a misspelt name in the progress line.
E2_CODE = '''\
"""E2: how the early loss spike depends on the batch size, from 8 to 4096.

For every batch size Adam is run with and without a 100-step linear warmup.  Batches
are drawn without replacement, so that a large batch never counts an example twice.
Writes metrics.json (the spike per batch size) and artifacts/spike_by_batch.png.
"""

import json
import os

# The arrays are small; extra threads would only make the timing noisy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
from toy import make_data, train

DATA_SEED = 0
SEEDS = (1, 2)
BATCH_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
LEARNING_RATE = 0.15
WARMUP_STEPS = 100
# The spike is looked for in the first steps, so that is all that is run.
STEPS = 100


def spike(curve):
    """How far the loss rises above its starting value."""
    return max(0.0, float(curve.max() - curve[0]))


def plot(spikes):
    # Imported here: the sweep should not wait for the plotting library.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("artifacts").mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(BATCH_SIZES, spikes[0], marker="o", label="no warmup")
    axis.plot(BATCH_SIZES, spikes[WARMUP_STEPS], marker="o", linestyle="--", label="warmup")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("batch size")
    axis.set_ylabel("early loss spike")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig("artifacts/spike_by_batch.png", dpi=110)


def main():
    active, labels = make_data(DATA_SEED)
    spikes = {0: [], WARMUP_STEPS: []}
    for batch in BATCH_SIZES:
        for warmup in spikes:
            runs = [
                train(active, labels, batch=batch, warmup=warmup,
                      learning_rate=LEARNING_RATE, steps=STEPS, seed=seed, replace=False)
                for seed in SEEDS
            ]
            spikes[warmup].append(spike(np.mean(runs, axis=0)))
        print(f"batch {batch}: spike {spikes[0][-1]:.3f} without warmup, "
              f"{spikes[WARMUP_STEP][-1]:.3f} with", flush=True)

    metrics = {"learning_rate": LEARNING_RATE, "warmup_steps": WARMUP_STEPS,
               "seeds": len(SEEDS)}
    for index, batch in enumerate(BATCH_SIZES):
        metrics[f"spike_b{batch}_no_warmup"] = round(spikes[0][index], 3)
        metrics[f"spike_b{batch}_warmup"] = round(spikes[WARMUP_STEPS][index], 3)
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")
    plot(spikes)


if __name__ == "__main__":
    main()
'''
#: The one-name correction the runner's repair makes.  The second failure is not a slip
#: in the code: the plan asks for a batch larger than the data set.
E2_SLIP, E2_FIX = "spikes[WARMUP_STEP][-1]", "spikes[WARMUP_STEPS][-1]"

E4_CODE = (
    '''\
"""E4: how long does the warmup need to be at batch size 16?

Adam at learning rate 0.15 with linear warmups of 0 to 200 steps.  Writes
metrics.json (the early spike, the lowest loss and the loss after training, per
warmup length) and artifacts/warmup_length.png.
"""

'''
    + _SCRIPT_HEAD
    + """\
from toy import make_data, train

DATA_SEED = 0
SEEDS = (1, 2, 3)
BATCH = 16
LEARNING_RATE = 0.15
STEPS = 400
WARMUP_LENGTHS = (0, 10, 25, 50, 100, 200)
WINDOW = 100


def main():
    active, labels = make_data(DATA_SEED)
    metrics = {"batch_size": BATCH, "learning_rate": LEARNING_RATE, "steps": STEPS,
               "seeds": len(SEEDS)}
    curves = {}
    for warmup in WARMUP_LENGTHS:
        runs = [
            train(active, labels, batch=BATCH, warmup=warmup, learning_rate=LEARNING_RATE,
                  steps=STEPS, seed=seed)
            for seed in SEEDS
        ]
        curve = curves[warmup] = np.mean(runs, axis=0)
        rise = float(curve[: WINDOW + 1].max() - curve[0])
        metrics[f"spike_warmup{warmup}"] = round(max(0.0, rise), 3)
        metrics[f"lowest_loss_warmup{warmup}"] = round(float(curve.min()), 3)
        metrics[f"final_loss_warmup{warmup}"] = round(float(curve[-1]), 3)
        print(f"warmup {warmup}: spike {metrics[f'spike_warmup{warmup}']}, lowest loss "
              f"{metrics[f'lowest_loss_warmup{warmup}']}, final loss "
              f"{metrics[f'final_loss_warmup{warmup}']}", flush=True)
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")

    Path("artifacts").mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for warmup, curve in curves.items():
        axis.plot(curve, linewidth=1.3, label=f"{warmup} steps")
    axis.set_xlabel("step")
    axis.set_ylabel(f"training loss (full set, mean of {len(SEEDS)} seeds)")
    axis.legend(title="warmup", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig("artifacts/warmup_length.png", dpi=110)


if __name__ == "__main__":
    main()
"""
)

E3_CODE = '''\
"""E3: the early loss spike in a small Transformer, at batch sizes 16 and 256.

A two-layer pre-norm Transformer learns to reverse random token sequences.  AdamW with
and without a linear warmup.  The data is generated here, so nothing is downloaded.
Writes metrics.json and artifacts/loss_curve.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

SEEDS = (1, 2, 3)
BATCH_SIZES = (16, 256)
LEARNING_RATE = 1e-3
STEPS = 2000
WARMUP_STEPS = 200
WINDOW = 200
VOCABULARY, LENGTH, WIDTH = 64, 32, 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Reverser(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(VOCABULARY, WIDTH)
        self.positions = nn.Embedding(LENGTH, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH, nhead=4, dim_feedforward=4 * WIDTH, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(WIDTH, VOCABULARY)

    def forward(self, tokens):
        places = torch.arange(tokens.shape[1], device=tokens.device)
        return self.head(self.encoder(self.tokens(tokens) + self.positions(places)))


def train(batch, warmup, seed):
    """The training loss of every step, smoothed over a window of 20 steps."""
    torch.manual_seed(seed)
    model = Reverser().to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for step in range(1, STEPS + 1):
        for group in optimiser.param_groups:
            group["lr"] = LEARNING_RATE * (min(1.0, step / warmup) if warmup else 1.0)
        tokens = torch.randint(0, VOCABULARY, (batch, LENGTH), device=DEVICE)
        logits = model(tokens)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCABULARY), tokens.flip(1).reshape(-1)
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    curve = torch.tensor(losses)
    return torch.nn.functional.avg_pool1d(curve[None, None], 20, stride=1)[0, 0]


def main():
    metrics, curves = {}, {}
    for batch in BATCH_SIZES:
        for warmup in (0, WARMUP_STEPS):
            name = f"b{batch}_{'warmup' if warmup else 'no_warmup'}"
            curve = torch.stack([train(batch, warmup, seed) for seed in SEEDS]).mean(0)
            curves[name] = curve
            rise = float(curve[:WINDOW].max() - curve[0])
            metrics[f"spike_{name}"] = round(max(0.0, rise), 4)
            metrics[f"final_loss_{name}"] = round(float(curve[-1]), 4)
            print(f"{name}: spike {metrics[f'spike_{name}']}, "
                  f"final loss {metrics[f'final_loss_{name}']}", flush=True)
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")

    Path("artifacts").mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for name, curve in curves.items():
        axis.plot(curve.numpy(), linewidth=1.3, label=name.replace("_", " "))
    axis.set_xlabel("step")
    axis.set_ylabel("training loss (smoothed)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig("artifacts/loss_curve.png", dpi=110)


if __name__ == "__main__":
    main()
'''

_SCRIPT_FIELDS: dict[str, Any] = {
    "mode": "script",
    "entrypoint": "main.py",
    "timeout_seconds": 300,
    "network": False,
    "packages": ["numpy", "matplotlib"],
    "est_minutes": 1,
    "est_usd": 0,
}

# ---------------------------------------------------------------------------
# Findings: the text is fixed, the numbers come from E1's run record
# ---------------------------------------------------------------------------

#: The metrics of E1 the findings quote.
_QUOTED = (
    "initial_loss",
    "learning_rate",
    "seeds",
    *(
        f"{measure}_b{batch}_{schedule}"
        for measure in ("spike", "lowest_loss", "loss_step100", "final_loss")
        for batch in (16, 256)
        for schedule in ("no_warmup", "warmup")
    ),
)

F1_TITLE = "Warmup removes the early loss spike at batch size 16 (E1)"
F1_BODY = """\
## What was observed

In [[results/E1-run1|run 1 of E1]], Adam at learning rate {learning_rate} without warmup
raised the training loss at batch size 16 by {spike_b16_no_warmup} above its initial value
of {initial_loss} within the first 100 steps. With a linear warmup over 100 steps the rise
was {spike_b16_warmup}. At batch size 256 it was {spike_b256_no_warmup} without warmup and
{spike_b256_warmup} with it. Each figure is the mean of {seeds} seeds.

After 300 steps the batch-16 runs stood at {final_loss_b16_no_warmup} without warmup and
at {final_loss_b16_warmup} with it; the batch-256 runs at {final_loss_b256_no_warmup} and
{final_loss_b256_warmup}.

## What it means

On this problem the early spike is a small-batch effect, and a short warmup is enough to
remove it, as [[H1]] and [[H2]] predicted. The run with warmup did not end lower than the
run without, so [[H3]] does not hold here.

## Limits

One toy problem with sparse features, one learning rate, {seeds} seeds, 300 steps. The
spike is read off the mean curve, not per seed. Nothing here says which of the accounts
in [[topics/why-warmup-helps|Why warmup helps]] is behind it.
"""

F2_TITLE = "After the warmup ends, the batch-16 run gives back what it gained (E1)"
F2_BODY = """\
## What was observed

With the warmup, the batch-16 loss in [[results/E1-run1|run 1 of E1]] fell as low as
{lowest_loss_b16_warmup} and stood at {loss_step100_b16_warmup} at step 100, against
{loss_step100_b16_no_warmup} without warmup. From there it rose again: at step 300 it was
{final_loss_b16_warmup}, against {final_loss_b16_no_warmup}.

## What it means

The warmup did not buy a better end point. Once the learning rate reaches
{learning_rate}, the batch-16 run returns to the level of the run that never warmed up,
which suggests that this learning rate is too high for batch size 16 however training
starts. A straight-line fit to the last 50 steps of the warmup run reaches the loss of
the batch-256 runs only after about 1840 further steps.

## Limits

The fit is not part of the run's record. 300 steps may be too few to see where either
run settles, and the result is for one learning rate.
"""

JOURNAL_ROUND_1 = """\
## What was done

- Research: 14 sources, three topic pages and four questions, next to the intake page on
  the attached files.
- Hypotheses: four proposed. The person accepted [[H1]], [[H2]] and [[H3]] and turned down
  the RAdam hypothesis.
- Experiments: [[E1]] ran and succeeded. [[E2]] failed twice.

## What the results say

[[F1]]: the early spike is there at batch size 16 and not at 256, and a 100-step warmup
removes it. The run with warmup does not end lower.

## What did not work

[[E2]] first stopped on a misspelt name in its progress line, which the runner corrected.
The second run measured batch sizes 8 to 2048 and then stopped: the plan asks for a batch
of 4096 drawn without replacement from 4000 examples. That is a matter of the plan, not
of the code. Either the largest batch size comes down or batches are drawn with
replacement. The numbers the second run printed before it stopped are in its log; they
are not a result.

## Suggested next steps

- Sweep the length of the warmup at batch size 16.
- Repeat the comparison at a lower learning rate.
- Try the comparison on a model with dense gradients ([[Q5]]).
"""

LESSON_TITLE = "A batch drawn without replacement cannot be larger than the data set"
LESSON_BODY = """\
When a plan sweeps the batch size, check its largest value against the number of training
examples before any code is written. Sampling without replacement fails once the batch
exceeds the data set, and it fails late: the sweep has by then spent its time on the
smaller sizes and writes no metrics. Seen in [[E2]].
"""

DECISION_TITLE = "Decision: plain Adam before RAdam"
DECISION_BODY = """\
I turned down the RAdam hypothesis. [[sources/Ma2021Adequacy|Ma and Yarats (2021)]] already
compare RAdam with untuned warmup, and it would answer a question about one optimiser, not
mine about the batch size. If the batch-size effect holds up under plain Adam, RAdam is a
natural follow-up.
"""

# ---------------------------------------------------------------------------
# The story: a clock, and the engine's bookkeeping
# ---------------------------------------------------------------------------


class _Timeline:
    """The clock of the story: it starts in the past, runs at the speed of real time and
    is moved forward by hand between the steps, but never past the present."""

    def __init__(self, start: datetime) -> None:
        self._at = start
        self._mark = time.monotonic()

    def __call__(self) -> datetime:
        told = self._at + timedelta(seconds=time.monotonic() - self._mark)
        return min(told, datetime.now(UTC))

    def skip(self, **delta: float) -> None:
        self._at = self() + timedelta(**delta)
        self._mark = time.monotonic()


def _count(number: int, kind: str) -> str:
    many = SECTIONS.get(kind) or f"{kind}s"
    return f"{number} {kind if number == 1 else many}"


def _rounded(value: float, digits: int) -> float:
    """``round``, for the methods below whose ``round`` is the round of the project."""
    return round(value, digits)


def _and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _tokens_worth(usd: float, model: str) -> Usage | None:
    """Token counts that the price table values at ``usd`` for ``model``, a quarter of it
    as output; ``None`` for a model the table has no price for."""
    price = price_for(model)
    if price is None or price.input <= 0 or price.output <= 0:
        return None
    written = int(usd * 0.25 / price.output * 1_000_000)
    read = round((usd - written * price.output / 1_000_000) / price.input * 1_000_000)
    return Usage(input_tokens=read, output_tokens=written, requests=max(1, read // 20_000))


class _Story:
    """Tells the project's history in order: phases, checkpoints and the time between.

    It keeps ``state.json`` the way the engine does (section 5.2), so that the events
    the project emits carry the round and phase they happened in.
    """

    def __init__(self, project: Project, clock: _Timeline) -> None:
        self.project = project
        self.clock = clock
        #: The illustrative model usage of the phases so far, and what it is priced at.
        self._usage = UsageTracker()
        self._model_usd = 0.0
        #: The agent of the phase that ran last: a checkpoint is told in its name.
        self._agent = ""
        self.state: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "idle",
            "round": 0,
            "phase": None,
            "next_phase": "intake",
            "checkpoint": None,
            "phases": [],
            "worker": None,
            "spent": {"model_usd": 0.0, "experiment_seconds": 0.0, "experiment_usd": 0.0},
            "error": None,
        }

    def _save(self, **changes: Any) -> None:
        self.state.update(changes)
        write_json(self.project.root / STATE_FILE, self.state)

    def wait(self, **delta: float) -> None:
        self.clock.skip(**delta)

    def say(self, agent: str, summary: str, *, after: float = 45) -> None:
        """One line of an agent's progress, ``after`` seconds after the last thing."""
        self.wait(seconds=after)
        self.project.emit("agent.activity", summary, agent=agent)

    def fail(self, error: BaseException) -> None:
        """Leave a build that broke off as what it is: a project whose phase failed."""
        self._save(status="failed", error=f"The demonstration could not be built: {error}")

    @contextlib.contextmanager
    def phase(
        self, name: str, *, round: int, agent: str, opening: str, closing: str, share: float
    ) -> Iterator[None]:
        """Run one phase: what happens inside is recorded as its work.

        ``share`` is the part of the illustrative model cost this phase is given.
        """
        began = self.clock()
        before = {page.path for page in self.project.pages()}
        self._agent = agent
        self._save(status="running", round=round, phase=name, next_phase=name, checkpoint=None)
        self.project.emit("phase.started", opening, phase=name, agent=agent)
        yield
        self.wait(seconds=20)
        produced = sorted({page.path for page in self.project.pages()} - before)
        seconds = _rounded((self.clock() - began).total_seconds(), 1)
        self.state["phases"].append(
            {
                "round": round,
                "phase": name,
                "status": "completed",
                "started": format_stamp(began),
                "finished": format_stamp(self.clock()),
                "seconds": seconds,
                "attempts": 1,
                "error": None,
                "produced": produced,
                "instruction": None,
            }
        )
        self._save(spent={**self.state["spent"], "model_usd": self._spend_on_models(name, share)})
        # The same data the engine gives these events, for whoever reads them.
        self.project.emit(
            "phase.finished",
            closing,
            phase=name,
            agent=agent,
            refs=produced[:50],
            data={"produced": produced, "seconds": seconds},
        )

    def _spend_on_models(self, name: str, share: float) -> float:
        """Enter the phase's part of the illustrative cost in ``usage.json`` and return
        what the ledger now adds up to.

        ``state.json`` only repeats what the ledger says: the engine prices the model
        spend from ``usage.json`` whenever it works on a project, so a cost that the
        ledger did not account for would be gone the first time the person continues.
        """
        from paper_orchestra.research.agents import model_for

        model = model_for(self.project, self._agent)
        tokens = _tokens_worth(share * ILLUSTRATIVE_MODEL_USD, model)
        if tokens is not None:
            # Labelled the way a session is: the phase, then the agent.
            self._usage.record(f"{name}.{self._agent}", model, tokens)
            self._model_usd += estimate_cost_usd(model, tokens) or 0.0
        total = _rounded(self._model_usd, 4)
        ledger = {**self._usage.to_dict(), "estimated_cost_usd": total, "unpriced_models": []}
        write_json(self.project.root / USAGE_FILE, ledger)
        return total

    def checkpoint(self, following: str, lead: str) -> None:
        """Stop for the person: ``lead`` says what the phase did, the rest what is waiting."""
        counts = self.project.counts()["pending"]
        parts = [
            _count(counts["by_kind"][kind], kind) for kind in KINDS if kind in counts["by_kind"]
        ]
        if counts["proposals"]:
            edits = counts["proposals"]
            parts.append(f"{edits} suggested edit{'s' if edits != 1 else ''}")
        waiting = f"{_and(parts)} {'is' if counts['total'] == 1 else 'are'} waiting for you."
        summary = f"{lead} {waiting[0].upper()}{waiting[1:]}" if parts else lead
        self._save(
            status="checkpoint",
            next_phase=following,
            checkpoint={
                "round": self.state["round"],
                "phase": self.state["phase"],
                "at": format_stamp(self.clock()),
                "summary": summary,
                "pending": dict(counts["by_kind"]),
            },
        )
        self.project.emit(
            "checkpoint",
            summary,
            phase=self.state["phase"],
            agent=self._agent,
            refs=[page.path for page in self.project.pages(review="proposed")][:50],
            data={
                "pending": dict(counts["by_kind"]),
                "edits": counts["proposals"],
                "next": following,
            },
        )

    def spend_on_runs(self, records: list[Mapping[str, Any]]) -> None:
        """Account for the time the runs really took."""
        seconds = sum(float(record.get("seconds") or 0) for record in records)
        self._save(spent={**self.state["spent"], "experiment_seconds": _rounded(seconds, 1)})


# ---------------------------------------------------------------------------
# The two rounds
# ---------------------------------------------------------------------------


def _intake(story: _Story) -> None:
    project = story.project
    with story.phase(
        "intake",
        round=1,
        agent="intake",
        opening="The intake agent started reading the brief and the four attached files.",
        closing="Intake is done: one page on the attached files and one open question.",
        share=0.04,
    ):
        story.say("intake", "Read notes.md and prior_run.csv; looked at loss_curves.png.")
        story.say("intake", "kingma2015.pdf is a one-page placeholder; nothing to take from it.")
        project.create_page(
            "topic",
            TOPIC_FILES,
            FILES_BODY,
            author="agent:intake",
            summary="What each of the four attached files adds, and what none of them shows.",
            slug="attached-files",
        )
        story.wait(seconds=40)
        title, summary, body, relations, author = QUESTIONS[0]
        project.create_page(
            "question", title, body, author=author, summary=summary, relations=relations
        )


def _research(story: _Story) -> None:
    project = story.project
    with story.phase(
        "research",
        round=1,
        agent="scout",
        opening="The scout started looking for prior work on warmup and adaptive optimisers.",
        closing="Research is done: 14 sources, three topic pages and four more questions.",
        share=0.34,
    ):
        story.say("scout", "Searching for work on why learning-rate warmup helps.")
        for index, source in enumerate(SOURCES):
            if index == 5:
                story.say("scout", "Searching for the origin of warmup in large-batch training.")
            if index == 10:
                story.say("scout", "Following up on warmup in Transformers and loss curvature.")
            story.wait(seconds=70)
            project.write_record(
                "source",
                source.key,
                source.title,
                _source_body(source),
                summary=f"{_credit(source)}: {source.reports.split('. ')[0].rstrip('.')}.",
                fields={
                    "authors": list(source.authors),
                    "year": source.year,
                    "venue": source.venue,
                    "link": source.link,
                    "verified_by": "demonstration",
                },
            )
        story.say(
            "scout",
            "Found 17 candidate papers; 14 are on record as sources, 3 were off topic.",
            after=90,
        )
        for slug, title, body, summary, cites in (
            (
                "why-warmup-helps",
                TOPIC_ACCOUNTS,
                ACCOUNTS_BODY,
                "The variance, update-size and curvature accounts, and what they imply "
                "for small batches.",
                ["Liu2020Variance", "Ma2021Adequacy", "Gilmer2022Loss", "Kalra2024Why",
                 "Gotmare2019Closer"],
            ),
            (
                "adaptive-learning-rate-variance",
                TOPIC_VARIANCE,
                VARIANCE_BODY,
                "Why Adam's early second-moment estimate is noisy, and what makes it worse.",
                ["Kingma2015Adam", "Liu2020Variance", "Ma2021Adequacy", "Reddi2018Convergence"],
            ),
            (
                "warmup-and-batch-size",
                TOPIC_BATCH,
                BATCH_BODY,
                "The case for warmup at large batches is made for SGD and does not carry "
                "over to Adam as it stands.",
                ["Goyal2017Accurate", "He2019Bag", "Keskar2017Large", "Smith2018Dont",
                 "Vaswani2017Attention", "Xiong2020Layer"],
            ),
        ):  # fmt: skip
            story.say("scout", f'Writing the topic page "{title}".', after=150)
            project.create_page(
                "topic", title, body, author="agent:scout", summary=summary, slug=slug,
                relations={"cites": cites},
            )  # fmt: skip
        story.say("scout", "Writing down what the literature leaves open.", after=120)
        for title, summary, body, relations, author in QUESTIONS[1:]:
            story.wait(seconds=35)
            project.create_page(
                "question", title, body, author=author, summary=summary, relations=relations
            )
    story.checkpoint(
        "hypothesize", "Round 1, research: 14 sources, 4 topic pages and 5 questions are in."
    )


def _review_research(story: _Story) -> None:
    """The person reads the first round of pages, accepts them and adds a reading of their own."""
    project = story.project
    story.wait(minutes=14)
    for page in project.pages(kind=("topic", "question"), review="proposed"):
        story.wait(seconds=50)
        project.review(page.path, "accept")
    story.wait(minutes=4)
    accounts = project.page("topics/why-warmup-helps")
    project.update_page(
        accounts.path,
        author="human",
        body=accounts.body.rstrip("\n") + "\n" + ACCOUNTS_MY_READING,
        base_updated=accounts.updated,
    )


def _hypothesize(story: _Story) -> None:
    project = story.project
    story.wait(minutes=3)
    with story.phase(
        "hypothesize",
        round=1,
        agent="theorist",
        opening="The theorist started turning the questions into hypotheses that can fail.",
        closing="Four hypotheses are proposed, each with predictions an experiment can contradict.",
        share=0.12,
    ):
        story.say("theorist", "Reading the topic pages, the questions and the person's reading.")
        for title, summary, body, relations in HYPOTHESES:
            story.wait(seconds=80)
            project.create_page(
                "hypothesis", title, body, author="agent:theorist", summary=summary,
                relations=relations,
            )  # fmt: skip
    story.checkpoint("design", "Round 1, hypotheses: 4 proposed.")

    story.wait(minutes=11)
    for ref in ("H1", "H2", "H3"):
        story.wait(seconds=45)
        project.review(ref, "accept")
    story.wait(seconds=90)
    project.review("H4", "reject", reason=H4_REJECTION)
    story.wait(minutes=2)
    project.create_page(
        "journal", DECISION_TITLE, DECISION_BODY, author="human",
        summary="Why the RAdam hypothesis was turned down.",
        relations={"cites": ["Ma2021Adequacy"]},
    )  # fmt: skip
    story.wait(seconds=70)
    project.add_request(
        "Keep the first experiments small enough to run on my laptop in under a minute.",
        role="designer",
    )


def _design(story: _Story, experiments: Any) -> None:
    project = story.project
    story.wait(minutes=1)
    request = project.requests()[0]
    with story.phase(
        "design",
        round=1,
        agent="designer",
        opening="The designer started planning experiments for H1, H2 and H3.",
        closing="Two experiments are proposed with their code; both run on a laptop in seconds.",
        share=0.17,
    ):
        story.say(
            "designer", "The person asked for experiments that run on a laptop in under a minute."
        )
        story.say("designer", "Planning one run that tests H1, H2 and H3 together.", after=100)
        project.create_page(
            "experiment", E1_TITLE, E1_BODY, author="agent:designer",
            summary="Adam with and without a 100-step warmup at two batch sizes, three seeds.",
            fields=_SCRIPT_FIELDS, relations={"tests": ["H1", "H2", "H3"]},
        )  # fmt: skip
        story.wait(seconds=110)
        experiments.write_code(
            project, "E1", {"main.py": E1_CODE, "toy.py": TOY_CODE}, author="agent:designer"
        )
        story.say("designer", "E1: main.py and toy.py are written and parse.", after=15)
        story.say("designer", "Planning a sweep over batch sizes to see where the spike stops.")
        project.create_page(
            "experiment", E2_TITLE, E2_BODY, author="agent:designer",
            summary="The spike at every power of two from 8 to 4096, with and without warmup.",
            fields=_SCRIPT_FIELDS, relations={"tests": ["H1"]},
        )  # fmt: skip
        story.wait(seconds=95)
        experiments.write_code(
            project, "E2", {"main.py": E2_CODE, "toy.py": TOY_CODE}, author="agent:designer"
        )
        story.say("designer", "E2: main.py and toy.py are written and parse.", after=15)
        project.close_request(request.id, round=1, phase="design")
    story.checkpoint("run", "Round 1, design: 2 experiments are planned.")

    story.wait(minutes=8)
    for ref in ("E1", "E2"):
        story.wait(seconds=75)
        project.review(ref, "accept")
        story.wait(seconds=20)
        experiments.approve(project, ref, backend="local")
    for ref in ("H1", "H2", "H3"):
        # What the engine does before a run: a hypothesis under test is no longer its
        # author's to rewrite.  Nothing happens where approving has done it already.
        project.set_state(ref, "testing", by="system")


class _Repair:
    """The runner's repair, scripted: it corrects the misspelt name once.

    The second failure of E2 has no fix in the code that keeps the plan, so the repair
    declines, which is what leaves E2 failed with something for the person to decide.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, plan: Any, run_record: Any, log_tail: str) -> dict[str, str] | None:
        self.calls += 1
        if self.calls > 1:
            return None
        return {"main.py": E2_CODE.replace(E2_SLIP, E2_FIX)}


def _run(story: _Story, experiments: Any, backend: Any) -> dict[str, Any]:
    """Run E1 and E2 for real.  Returns E1's metrics, which the findings quote."""
    project = story.project
    story.wait(minutes=2)
    with story.phase(
        "run",
        round=1,
        agent="runner",
        opening="The runner started the two approved experiments on this machine.",
        closing="E1 succeeded. E2 failed after one repair; its plan needs a decision.",
        share=0.03,
    ):
        first = experiments.run(project, "E1", compute=backend)
        story.wait(seconds=5)
        experiments.run(project, "E2", compute=backend, repair=_Repair())
    story.spend_on_runs([*experiments.runs(project, "E1"), *experiments.runs(project, "E2")])
    return _genuine_metrics(project, first, experiments.runs(project, "E2"))


def _genuine_metrics(project: Project, first: Mapping[str, Any], second: list) -> dict[str, Any]:
    """E1's metrics, after checking that the runs show what the story goes on to tell.

    The pages written after this point interpret these runs.  If the runs came out
    differently on this machine, those pages would state what no record supports, so
    the build stops instead.
    """
    metrics = dict(first.get("metrics") or {})
    log = project.experiment_dir("E1") / "runs" / str(first.get("number", 1)) / "log.txt"
    if first.get("status") != "succeeded" or any(name not in metrics for name in _QUOTED):
        raise ProjectError(
            "The demonstration's experiment E1 did not succeed on this machine "
            f"({first.get('error') or first.get('status')}), so its finding cannot be "
            f"written. The run's log is {log}; it needs numpy and matplotlib."
        )
    told = (
        metrics["spike_b16_no_warmup"] > 0.02
        and max(
            metrics[f"spike_{name}"] for name in ("b16_warmup", "b256_no_warmup", "b256_warmup")
        )
        < 0.01
        and metrics["final_loss_b16_warmup"] >= metrics["final_loss_b16_no_warmup"]
    )
    if not told:
        raise ProjectError(
            "The demonstration's experiment E1 ran, but its metrics do not show what the "
            f"demonstration's finding says ({metrics}). The finding is not written: see "
            f"{log} and research/demo.py."
        )
    if not second or any(record.get("status") == "succeeded" for record in second):
        raise ProjectError(
            "The demonstration's experiment E2 is meant to fail (its plan asks for a batch "
            "larger than its data set), but it did not. See research/demo.py."
        )
    return metrics


def _analyze(story: _Story, experiments: Any, metrics: Mapping[str, Any]) -> None:
    project = story.project
    quoted = {name: metrics[name] for name in _QUOTED}
    notes = []
    with story.phase(
        "analyze",
        round=1,
        agent="analyst",
        opening="The analyst started reading the result records of E1 and E2.",
        closing="Two findings, the round's journal and four suggested edits are written.",
        share=0.14,
    ):
        story.say(
            "analyst", "E1 succeeded: reading its metrics against the predictions of H1 to H3."
        )
        project.create_page(
            "finding", F1_TITLE, F1_BODY.format(**quoted), author="agent:analyst",
            summary="At batch size 16 the loss spikes without warmup and not with it; at "
            "256 it does not spike at all.",
            relations={"based_on": ["E1-run1"], "supports": ["H1", "H2"], "refutes": ["H3"],
                       "answers": ["Q1"]},
        )  # fmt: skip
        story.wait(seconds=100)
        second = project.create_page(
            "finding", F2_TITLE, F2_BODY.format(**quoted), author="agent:analyst",
            summary="The loss the warmup run gains by step 100 is gone by step 300.",
            relations={"based_on": ["E1-run1"], "refutes": ["H3"]},
        )  # fmt: skip
        # The check the engine makes after the analyst's session, with what it finds.
        # Nothing is stored on the page: the check is made again wherever it is shown.
        report = experiments.number_report(project, second)
        if report["unverified"]:
            numbers = ", ".join(report["unverified"])
            notes.append(f"F2: {report['summary']} Not found: {numbers}.")
            story.say(
                "analyst",
                f"F2 quotes {numbers}, which is in no run record, source or input file; "
                "the page and the checkpoint say so.",
                after=30,
            )
        story.say("analyst", "E2 failed twice: the second failure is in the plan, not the code.")
        project.create_page(
            "lesson", LESSON_TITLE, LESSON_BODY, author="agent:analyst",
            summary="Check the largest batch of a sweep against the size of the data set.",
        )  # fmt: skip
        for ref, state, summary in (
            ("H1", "supported", "Mark H1 as supported by F1"),
            ("H2", "supported", "Mark H2 as supported by F1"),
            ("H3", "refuted", "Mark H3 as refuted by F1"),
        ):
            story.wait(seconds=25)
            project.propose_edit(
                ref, author="agent:analyst", summary=summary, fields={"state": state}
            )
        story.wait(seconds=60)
        accounts = project.page("topics/why-warmup-helps")
        if ACCOUNTS_OPEN not in accounts.body:
            raise ProjectError(
                "The demonstration's topic page lost the paragraph its edit changes."
            )
        project.propose_edit(
            accounts.path,
            author="agent:analyst",
            summary="Say what F1 adds to the question the three accounts leave open",
            body=accounts.body.replace(ACCOUNTS_OPEN, ACCOUNTS_OPEN_PROPOSED),
        )
        story.wait(seconds=45)
        project.propose_edit(
            IDEA_PATH,
            author="agent:analyst",
            summary="Narrow the idea to problems with sparse gradients",
            body=project.page(IDEA_PATH).body.replace(
                "it is small batches that need it",
                "it is small batches with sparse gradients that need it",
            ),
        )
        story.wait(seconds=80)
        project.create_page(
            "journal", "Round 1", JOURNAL_ROUND_1, author="agent:analyst", slug="round-1",
            summary="E1 supports H1 and H2 and speaks against H3; E2 needs its plan changed.",
        )  # fmt: skip
        for result in project.pages(kind="result"):
            # What the engine does when an analysis ends: a record it has read is not
            # new to the next one.  Without the mark, continuing the project would
            # have these results analysed a second time.
            project.update_page(result.path, author="system", fields={"analysed": 1})
    story.checkpoint(
        "hypothesize", " ".join(["Round 1, analysis: E1 succeeded and E2 failed.", *notes])
    )


def _review_analysis(story: _Story) -> None:
    """The person accepts the first finding and two verdicts, and says what they want next."""
    project = story.project
    story.wait(minutes=6)
    project.review("F1", "accept")
    by_summary = {proposal.summary: proposal for proposal in project.proposals()}
    for summary in ("Mark H1 as supported by F1", "Mark H3 as refuted by F1"):
        story.wait(seconds=40)
        project.review(by_summary[summary].id, "accept")
    story.wait(seconds=55)
    project.review(
        by_summary["Narrow the idea to problems with sparse gradients"].id,
        "reject",
        reason="The idea stays as I wrote it. Whether it needs sparse gradients is Q5.",
    )
    story.wait(seconds=30)
    project.set_state("Q1", "answered", by="human", reason="F1: no spike at batch size 256.")
    story.wait(minutes=1)
    project.add_request(
        "Sweep the warmup length instead of fixing it at 100 steps.", target="H2", role="designer"
    )
    story.wait(seconds=50)
    project.add_request(
        "Before I rely on this, try it on something closer to a real model than the toy.",
        target="H1",
        role="designer",
    )


def _second_round(story: _Story, experiments: Any, pages: int | None) -> None:
    project = story.project
    story.wait(minutes=1)
    with story.phase(
        "hypothesize",
        round=2,
        agent="theorist",
        opening="The theorist started on round 2 from F1 and what the person turned down.",
        closing="One hypothesis about the cause of the spike and one question are proposed.",
        share=0.07,
    ):
        story.say("theorist", "H4 was rejected: staying with plain Adam, as the person decided.")
        title, summary, body, relations = HYPOTHESIS_NEXT
        project.create_page(
            "hypothesis", title, body, author="agent:theorist", summary=summary,
            relations=relations,
        )  # fmt: skip
        story.wait(seconds=70)
        title, summary, body, relations = QUESTION_NEXT
        project.create_page(
            "question", title, body, author="agent:theorist", summary=summary,
            relations=relations,
        )  # fmt: skip
    story.checkpoint("design", "Round 2, hypotheses: 1 proposed.")

    story.wait(minutes=2)
    handled = [request for request in project.requests() if request.role == "designer"]
    with story.phase(
        "design",
        round=2,
        agent="designer",
        opening="The designer started on the two things the person asked for.",
        closing="Two experiments are proposed: a warmup-length sweep and a small Transformer.",
        share=0.09,
    ):
        story.say("designer", "The person asked for something closer to a real model (H1).")
        project.create_page(
            "experiment", E3_TITLE, E3_BODY, author="agent:designer",
            summary="A two-layer Transformer at batch sizes 16 and 256, with and without "
            "warmup, on one T4.",
            fields={
                "mode": "gpu", "entrypoint": "main.py", "timeout_seconds": 1800, "gpu": "T4",
                "network": False, "packages": ["torch", "matplotlib"], "est_minutes": 20,
                "est_usd": 0.2,
            },
            relations={"tests": ["H1", "H2"], "cites": ["Loshchilov2019Decoupled"]},
        )  # fmt: skip
        story.wait(seconds=120)
        experiments.write_code(project, "E3", {"main.py": E3_CODE}, author="agent:designer")
        story.say("designer", "E3: main.py is written and parses; it needs a GPU to run.", after=15)
        story.say("designer", "The person asked for a sweep of the warmup length (H2).")
        project.create_page(
            "experiment", E4_TITLE, E4_BODY, author="agent:designer",
            summary="Warmups of 0 to 200 steps at batch size 16.",
            fields=_SCRIPT_FIELDS, relations={"tests": ["H2"], "follows": ["F1"]},
        )  # fmt: skip
        story.wait(seconds=85)
        experiments.write_code(
            project, "E4", {"main.py": E4_CODE, "toy.py": TOY_CODE}, author="agent:designer"
        )
        story.say("designer", "E4: main.py and toy.py are written and parse.", after=15)
        for request in handled:
            project.close_request(request.id, round=2, phase="design")
        if pages:
            _fill(project, pages)
    story.checkpoint("run", "Round 2, design: 2 experiments are planned.")

    story.wait(minutes=3)
    project.review("E4", "accept")
    story.wait(seconds=25)
    experiments.approve(project, "E4", backend="local")
    story.wait(seconds=70)
    # One for every agent, one for the role it concerns: both wait for the next phases.
    project.add_request("Also look at schedule-free optimisers.")
    story.wait(seconds=45)
    project.add_request(
        "Can E3 finish on a T4 in under 20 minutes? If not, shrink the model first.",
        target="E3",
        role="designer",
    )


# ---------------------------------------------------------------------------
# A large project for layout tests
# ---------------------------------------------------------------------------

_FILLER_NOTE = "_Generated to fill the demonstration project for layout tests; it says nothing._\n"
_FILLER_SUBJECTS = (
    "gradient clipping", "label smoothing", "cosine decay", "weight decay",
    "batch normalisation", "layer-wise learning rates", "gradient noise", "momentum",
    "the epsilon of Adam", "mixed precision", "data order", "initialisation scale",
)  # fmt: skip
#: Sources and questions outnumber the rest, as they do in a real project.
_FILLER_KINDS = (
    "source", "source", "source", "topic", "question", "question", "hypothesis", "hypothesis",
    "experiment", "finding",
)  # fmt: skip
#: Who would have written a page of each kind, and what it is called.
_FILLER_PAGES = {
    "topic": ("agent:scout", "Generated topic {number}: what is known about {subject}"),
    "question": ("agent:scout", "Generated question {number}: does {subject} change the picture?"),
    "hypothesis": (
        "agent:theorist",
        "Generated hypothesis {number}: {subject} interacts with warmup",
    ),
    "experiment": (
        "agent:designer",
        "Generated experiment {number}: warmup with and without {subject}",
    ),
    "finding": ("agent:analyst", "Generated finding {number}: no effect of {subject} was seen"),
}


def _fill(project: Project, pages: int) -> None:
    """Add generated pages until the wiki has ``pages`` of them.

    The pages say that they are generated.  They are there to show how the views hold
    up at size: a long tree, a crowded graph, an inbox with many items.
    """
    rng = random.Random(pages)
    number = 0
    while project.counts()["pages"] < pages:
        number += 1
        _filler(project, rng.choice(_FILLER_KINDS), number, rng)


def _filler(project: Project, kind: str, number: int, rng: random.Random) -> None:
    """One generated page of ``kind``, linked to pages that exist, in a review of its own."""
    subject = _FILLER_SUBJECTS[number % len(_FILLER_SUBJECTS)]
    summary = f"Generated {kind} {number} on {subject}."
    if kind == "source":
        year = 2000 + number % 25
        project.write_record(
            "source",
            f"Generated{year}Page{number}",
            f"Generated source {number}: a study of {subject}",
            f"- **Authors:** not a real paper\n- **Year:** {year}\n\n{_FILLER_NOTE}",
            summary=summary,
            fields={"year": year, "verified_by": "demonstration"},
        )
        return
    sources = [page.id for page in project.pages(kind="source") if page.id]
    hypotheses = [page.id for page in project.pages(kind="hypothesis", review="accepted")]
    relations: dict[str, list[str]] = {"cites": rng.sample(sources, 2)}
    if kind == "question":
        relations["about"] = ["idea"]
    elif kind == "hypothesis":
        relations["motivated_by"] = rng.sample(sources, 1)
    elif kind == "experiment":
        relations["tests"] = rng.sample(hypotheses, 1)
    elif kind == "finding":
        relations.update(based_on=["E1-run1"], supports=rng.sample(hypotheses, 1))
    author, title = _FILLER_PAGES[kind]
    page = project.create_page(
        kind,
        title.format(number=number, subject=subject),
        _FILLER_NOTE,
        author=author,
        summary=summary,
        relations=relations,
        fields={"mode": "script"} if kind == "experiment" else None,
    )
    # Some are accepted, some turned down, some left waiting: every view has all three.
    decision = rng.choice(("accept", "accept", "accept", "wait", "wait", "reject"))
    if decision == "reject":
        project.review(page.path, "reject", reason="Generated: a rejected page for layout tests.")
    elif decision == "accept":
        project.review(page.path, "accept")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def build_demo_project(root: str | Path, *, pages: int | None = None) -> Project:
    """Build the demonstration project in ``root`` (a new or empty directory).

    The project is at a checkpoint of its second round that waits for the person:
    14 sources, 4 topics, 6 questions, hypotheses H1 to H5 and experiments E1 to E4 in
    every review and lifecycle state, a succeeded run with a figure, a failed one with
    a repair on record, two findings (one with a number no record supports), pending
    and decided edit proposals, open requests and the activity of both rounds.

    E1 and E2 are run on this machine with the local backend, which takes a few
    seconds and needs numpy and matplotlib.  No model is called and nothing is sent
    anywhere.  ``pages`` adds generated pages until the wiki has that many, for
    looking at the views with a large project.

    Returns the project, opened afresh.  Raises ``ProjectError`` when ``root`` is in
    use, or when the runs do not come out as the pages written about them say.
    """
    from paper_orchestra.compute import get_backend
    from paper_orchestra.research import experiments

    clock = _Timeline(datetime.now(UTC) - STORY_SPAN)
    with tempfile.TemporaryDirectory(prefix="paper-orchestra-demo-") as folder:
        project = Project.create(
            root,
            title=DEMO_TITLE,
            brief=BRIEF,
            files=_write_attachments(Path(folder)),
            # Room for the GPU experiment the designer goes on to propose (30 minutes).
            settings={"limits": {"experiment_seconds": 1800}},
            clock=clock,
        )
    story = _Story(project, clock)
    backend = get_backend("local")
    try:
        story.wait(minutes=1)
        _intake(story)
        _research(story)
        _review_research(story)
        _hypothesize(story)
        _design(story, experiments)
        metrics = _run(story, experiments, backend)
        _analyze(story, experiments, metrics)
        _review_analysis(story)
        _second_round(story, experiments, pages)
    except BaseException as exc:
        # What was built so far stays, to be looked at; it must not read as still running.
        story.fail(exc)
        raise
    finally:
        backend.close()
    # The story's clock stays with the object that told it.
    return Project.open(root)
