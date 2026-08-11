"""Experiment Scheduling Center for the Joint-Problem project.

A Python backend that turns an approved experiment plan into a manifest,
splits it into shardable tasks, schedules them across 4 fixed worker slots
(2 local + 2 MetaCentrum PBS), monitors progress, and updates MLflow and
result tables.
"""

__version__ = "0.1.0"
