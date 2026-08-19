"""5703 crisis-dialogue TTS rendering pipeline.

The distribution and CLI are named ``5703tts``; Python identifiers cannot
start with a digit, so the import package is ``tts5703``.
"""

from .pipeline import PipelineResult, run_dialogue

__all__ = ["PipelineResult", "run_dialogue"]
