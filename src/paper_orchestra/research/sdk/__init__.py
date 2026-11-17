"""The ``sdk`` runtime: the roles of a research project as Claude Agent SDK agents.

``driver`` is the seam to the SDK (``claude`` the real driver, ``fake`` the
scripted one), ``guard`` the guardrails, ``session`` one role's agent with its
board tools, spend, stop and transcript, and ``experiment`` the agent that
carries out an approved experiment.  Nothing here imports ``claude_agent_sdk``
until an agent is started.
"""
