"""Runnable pipeline, analysis and utility code.

Installed as a package (see pyproject.toml) so these modules import each other
by name -- `from scripts.pipeline.audio_processing_config import ...` -- instead
of injecting paths into sys.path at the top of every file.
"""
