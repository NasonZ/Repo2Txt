"""Experiment: every tool reply ends with the selected paths, not only their count.

Before the prefix change, the system prompt restated the selection's paths every round; moving
that line out (so a prompt cache can reuse the opening) cost 0.041 recall on 100 tuning tasks.
This puts the same paths where the model reads them next, at the end of its latest tool reply,
which keeps the opening of every request unchanged.
"""

from repo2txt import selector

_original = selector.Selector._run


def _run(self, call):
    reply = _original(self, call)
    paths = self.selection.paths
    listed = ", ".join(paths) if paths else "nothing"
    return f"{reply}\n[Selection, most important first: {listed}]"


def apply() -> None:
    selector.Selector._run = _run
