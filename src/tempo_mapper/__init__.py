"""Tempo Mapper — automatic BPM and tempo-change detection."""

from .tempo import TempoAnalysis, analyze
from .segments import Segment, detect_segments

__all__ = ["TempoAnalysis", "analyze", "Segment", "detect_segments"]
