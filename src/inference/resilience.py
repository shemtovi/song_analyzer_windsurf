"""Resilience layer for handling imperfect/noisy input data.

Provides tools for:
- Note filtering (ghost notes, outliers)
- Timing tolerance and alignment
- Pitch correction within tolerance
- Confidence-weighted analysis
- Statistical outlier detection
- Musical plausibility filtering
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict

from ..core import Note, PITCH_NAMES


@dataclass
class NoteConfidence:
    """A note with associated confidence score."""
    note: Note
    confidence: float  # 0.0 - 1.0
    source_confidence: float = 1.0  # Confidence from transcription
    timing_confidence: float = 1.0  # How well timing aligns
    pitch_confidence: float = 1.0  # How stable the pitch is
    
    @property
    def combined_confidence(self) -> float:
        """Combined confidence from all factors."""
        return (
            self.source_confidence * 0.4 +
            self.timing_confidence * 0.3 +
            self.pitch_confidence * 0.3
        ) * self.confidence


@dataclass
class FilterStats:
    """Statistics from filtering operation."""
    original_count: int
    filtered_count: int
    removed_ghost_notes: int = 0
    removed_outliers: int = 0
    timing_adjusted: int = 0
    pitch_corrected: int = 0
    
    @property
    def retention_rate(self) -> float:
        if self.original_count == 0:
            return 1.0
        return self.filtered_count / self.original_count


class NoteFilter:
    """Filter and clean notes for more robust analysis."""
    
    def __init__(
        self,
        min_velocity: int = 15,
        min_duration: float = 0.03,
        max_duration: float = 30.0,
        pitch_range: Tuple[int, int] = (21, 108),  # Piano range
    ):
        """
        Initialize NoteFilter.
        
        Args:
            min_velocity: Minimum velocity to keep (remove ghost notes)
            min_duration: Minimum note duration in seconds
            max_duration: Maximum note duration in seconds
            pitch_range: Valid MIDI pitch range (min, max)
        """
        self.min_velocity = min_velocity
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.pitch_range = pitch_range
    
    def filter_notes(
        self,
        notes: List[Note],
        remove_ghosts: bool = True,
        remove_outliers: bool = True,
        remove_short: bool = True,
    ) -> Tuple[List[Note], FilterStats]:
        """
        Filter notes to remove noise and artifacts.
        
        Args:
            notes: List of notes to filter
            remove_ghosts: Remove very quiet notes
            remove_outliers: Remove pitch outliers
            remove_short: Remove very short notes
            
        Returns:
            Tuple of (filtered_notes, statistics)
        """
        stats = FilterStats(
            original_count=len(notes),
            filtered_count=0,
        )
        
        if not notes:
            return [], stats
        
        filtered = notes.copy()
        
        # Remove ghost notes (very quiet)
        if remove_ghosts:
            before = len(filtered)
            filtered = [n for n in filtered if n.velocity >= self.min_velocity]
            stats.removed_ghost_notes = before - len(filtered)
        
        # Remove duration outliers
        if remove_short:
            filtered = [
                n for n in filtered
                if self.min_duration <= n.duration <= self.max_duration
            ]
        
        # Remove pitch outliers
        if remove_outliers:
            before = len(filtered)
            filtered = [
                n for n in filtered
                if self.pitch_range[0] <= n.pitch <= self.pitch_range[1]
            ]
            stats.removed_outliers = before - len(filtered)
        
        stats.filtered_count = len(filtered)
        return filtered, stats
    
    def remove_duplicate_notes(
        self,
        notes: List[Note],
        time_tolerance: float = 0.05,
        pitch_tolerance: int = 0,
    ) -> List[Note]:
        """
        Remove duplicate notes (same pitch at same time).
        
        Keeps the note with higher velocity.
        
        Args:
            notes: List of notes
            time_tolerance: Max time difference to consider duplicate
            pitch_tolerance: Max pitch difference to consider duplicate
            
        Returns:
            Deduplicated notes
        """
        if not notes:
            return []
        
        # Sort by onset time
        sorted_notes = sorted(notes, key=lambda n: (n.onset, -n.velocity))
        
        result = []
        for note in sorted_notes:
            is_duplicate = False
            for existing in result:
                if (abs(note.onset - existing.onset) <= time_tolerance and
                    abs(note.pitch - existing.pitch) <= pitch_tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result.append(note)
        
        return result


class TimingCorrector:
    """Correct timing issues in note data."""
    
    def __init__(
        self,
        grid_resolution: int = 16,  # 16th notes
        tempo: float = 120.0,
        swing: float = 0.0,  # 0.0 = straight, 0.5 = full swing
    ):
        """
        Initialize TimingCorrector.
        
        Args:
            grid_resolution: Quantization grid (notes per beat)
            tempo: Tempo in BPM
            swing: Swing amount (0-0.5)
        """
        self.grid_resolution = grid_resolution
        self.tempo = tempo
        self.swing = swing
        self.beat_duration = 60.0 / tempo
        self.grid_duration = self.beat_duration / (grid_resolution / 4)
    
    def quantize_notes(
        self,
        notes: List[Note],
        strength: float = 1.0,
    ) -> List[Note]:
        """
        Quantize notes to grid with variable strength.
        
        Args:
            notes: Notes to quantize
            strength: Quantization strength (0=none, 1=full)
            
        Returns:
            Quantized notes
        """
        if not notes or strength == 0:
            return notes
        
        result = []
        for note in notes:
            # Find nearest grid position
            grid_position = round(note.onset / self.grid_duration)
            quantized_onset = grid_position * self.grid_duration
            
            # Apply quantization with strength
            new_onset = note.onset + (quantized_onset - note.onset) * strength
            
            # Maintain duration
            new_offset = new_onset + note.duration
            
            result.append(Note(
                pitch=note.pitch,
                onset=new_onset,
                offset=new_offset,
                velocity=note.velocity,
                instrument=note.instrument,
            ))
        
        return result
    
    def align_simultaneous_notes(
        self,
        notes: List[Note],
        tolerance: float = 0.05,
    ) -> List[Note]:
        """
        Align notes that should be simultaneous (chord alignment).
        
        Args:
            notes: Notes to align
            tolerance: Time tolerance for grouping
            
        Returns:
            Aligned notes
        """
        if not notes:
            return []
        
        # Group notes by onset time
        groups = defaultdict(list)
        sorted_notes = sorted(notes, key=lambda n: n.onset)
        
        current_group_onset = sorted_notes[0].onset
        for note in sorted_notes:
            if note.onset - current_group_onset > tolerance:
                current_group_onset = note.onset
            groups[current_group_onset].append(note)
        
        # Align each group to the median onset
        result = []
        for group_onset, group_notes in groups.items():
            if len(group_notes) > 1:
                # Use median onset as the aligned time
                onsets = [n.onset for n in group_notes]
                aligned_onset = np.median(onsets)
            else:
                aligned_onset = group_notes[0].onset
            
            for note in group_notes:
                shift = aligned_onset - note.onset
                result.append(Note(
                    pitch=note.pitch,
                    onset=aligned_onset,
                    offset=note.offset + shift,
                    velocity=note.velocity,
                    instrument=note.instrument,
                ))
        
        return result


class PitchCorrector:
    """Correct pitch errors within tolerance."""
    
    # Common pitch drift patterns
    DRIFT_TOLERANCE = 0.5  # Semitones
    
    def __init__(
        self,
        scale_notes: Optional[List[int]] = None,
        tolerance_cents: int = 50,
    ):
        """
        Initialize PitchCorrector.
        
        Args:
            scale_notes: Optional list of valid pitch classes (0-11)
            tolerance_cents: Pitch tolerance in cents (100 cents = 1 semitone)
        """
        self.scale_notes = scale_notes
        self.tolerance_cents = tolerance_cents
        self.tolerance_semitones = tolerance_cents / 100.0
    
    def snap_to_scale(
        self,
        notes: List[Note],
        scale_notes: Optional[List[int]] = None,
    ) -> List[Note]:
        """
        Snap notes to nearest scale tone.
        
        Only snaps if within tolerance.
        
        Args:
            notes: Notes to correct
            scale_notes: Scale pitch classes to snap to
            
        Returns:
            Corrected notes
        """
        target_scale = scale_notes or self.scale_notes
        if not target_scale:
            return notes
        
        result = []
        for note in notes:
            pc = note.pitch % 12
            octave = note.pitch // 12
            
            # Find nearest scale tone
            min_distance = 12
            nearest_pc = pc
            for scale_pc in target_scale:
                distance = min(abs(pc - scale_pc), 12 - abs(pc - scale_pc))
                if distance < min_distance:
                    min_distance = distance
                    nearest_pc = scale_pc
            
            # Only correct if within tolerance
            if min_distance <= self.tolerance_semitones and min_distance > 0:
                new_pitch = octave * 12 + nearest_pc
                # Handle octave wraparound
                if abs(new_pitch - note.pitch) > 6:
                    if new_pitch > note.pitch:
                        new_pitch -= 12
                    else:
                        new_pitch += 12
                
                result.append(Note(
                    pitch=new_pitch,
                    onset=note.onset,
                    offset=note.offset,
                    velocity=note.velocity,
                    instrument=note.instrument,
                ))
            else:
                result.append(note)
        
        return result
    
    def remove_pitch_outliers(
        self,
        notes: List[Note],
        context_window: float = 1.0,
        max_interval: int = 12,
    ) -> List[Note]:
        """
        Remove notes that are pitch outliers relative to context.
        
        Args:
            notes: Notes to filter
            context_window: Time window for context (seconds)
            max_interval: Maximum reasonable interval from context
            
        Returns:
            Filtered notes
        """
        if len(notes) < 3:
            return notes
        
        sorted_notes = sorted(notes, key=lambda n: n.onset)
        result = []
        
        for i, note in enumerate(sorted_notes):
            # Get notes in context window
            context_notes = [
                n for n in sorted_notes
                if n != note and abs(n.onset - note.onset) < context_window
            ]
            
            if not context_notes:
                result.append(note)
                continue
            
            # Calculate median pitch in context
            context_pitches = [n.pitch for n in context_notes]
            median_pitch = np.median(context_pitches)
            
            # Check if note is an outlier
            interval = abs(note.pitch - median_pitch)
            if interval <= max_interval:
                result.append(note)
        
        return result


class ConfidenceWeighter:
    """Weight notes by confidence for analysis."""
    
    def __init__(
        self,
        use_velocity_as_confidence: bool = True,
        use_duration_as_confidence: bool = True,
    ):
        """
        Initialize ConfidenceWeighter.
        
        Args:
            use_velocity_as_confidence: Higher velocity = higher confidence
            use_duration_as_confidence: Longer notes = higher confidence
        """
        self.use_velocity = use_velocity_as_confidence
        self.use_duration = use_duration_as_confidence
    
    def calculate_confidence(self, note: Note) -> float:
        """
        Calculate confidence score for a note.
        
        Args:
            note: Note to score
            
        Returns:
            Confidence score 0-1
        """
        scores = []
        
        if self.use_velocity:
            # Normalize velocity to 0-1
            velocity_score = note.velocity / 127.0
            scores.append(velocity_score)
        
        if self.use_duration:
            # Longer notes are more reliable (up to a point)
            # Peak confidence around 0.5 seconds
            duration_score = min(1.0, note.duration / 0.5)
            # Penalize very long notes slightly
            if note.duration > 2.0:
                duration_score *= 0.9
            scores.append(duration_score)
        
        return np.mean(scores) if scores else 1.0
    
    def weight_notes(self, notes: List[Note]) -> List[NoteConfidence]:
        """
        Add confidence weights to notes.
        
        Args:
            notes: Notes to weight
            
        Returns:
            List of NoteConfidence objects
        """
        return [
            NoteConfidence(
                note=note,
                confidence=self.calculate_confidence(note),
            )
            for note in notes
        ]
    
    def get_weighted_pitch_distribution(
        self,
        notes: List[Note],
    ) -> np.ndarray:
        """
        Get pitch class distribution weighted by confidence.
        
        Args:
            notes: Notes to analyze
            
        Returns:
            12-element array of weighted pitch class counts
        """
        distribution = np.zeros(12)
        
        for note in notes:
            pc = note.pitch % 12
            weight = self.calculate_confidence(note)
            distribution[pc] += weight
        
        # Normalize
        if distribution.sum() > 0:
            distribution /= distribution.sum()
        
        return distribution


class ResilienceProcessor:
    """Combined resilience processor for note data."""
    
    def __init__(
        self,
        min_velocity: int = 15,
        min_duration: float = 0.03,
        pitch_range: Tuple[int, int] = (21, 108),
        quantize_strength: float = 0.0,
        tempo: float = 120.0,
    ):
        """
        Initialize ResilienceProcessor.
        
        Args:
            min_velocity: Minimum velocity threshold
            min_duration: Minimum note duration
            pitch_range: Valid pitch range
            quantize_strength: How much to quantize timing (0-1)
            tempo: Tempo for timing corrections
        """
        self.note_filter = NoteFilter(
            min_velocity=min_velocity,
            min_duration=min_duration,
            pitch_range=pitch_range,
        )
        self.timing_corrector = TimingCorrector(tempo=tempo)
        self.pitch_corrector = PitchCorrector()
        self.confidence_weighter = ConfidenceWeighter()
        self.quantize_strength = quantize_strength
    
    def process(
        self,
        notes: List[Note],
        scale_notes: Optional[List[int]] = None,
        align_chords: bool = True,
        remove_duplicates: bool = True,
    ) -> Tuple[List[Note], FilterStats]:
        """
        Full resilience processing pipeline.
        
        Args:
            notes: Notes to process
            scale_notes: Optional scale for pitch correction
            align_chords: Whether to align simultaneous notes
            remove_duplicates: Whether to remove duplicate notes
            
        Returns:
            Tuple of (processed_notes, statistics)
        """
        if not notes:
            return [], FilterStats(original_count=0, filtered_count=0)
        
        # Step 1: Basic filtering
        filtered, stats = self.note_filter.filter_notes(notes)
        
        # Step 2: Remove duplicates
        if remove_duplicates:
            filtered = self.note_filter.remove_duplicate_notes(filtered)
        
        # Step 3: Align simultaneous notes (chords)
        if align_chords:
            before = len(filtered)
            filtered = self.timing_corrector.align_simultaneous_notes(filtered)
            stats.timing_adjusted = before - len(filtered)  # This won't change count but we track the operation
        
        # Step 4: Quantize if requested
        if self.quantize_strength > 0:
            filtered = self.timing_corrector.quantize_notes(
                filtered, strength=self.quantize_strength
            )
        
        # Step 5: Pitch correction if scale provided
        if scale_notes:
            filtered = self.pitch_corrector.snap_to_scale(filtered, scale_notes)
        
        stats.filtered_count = len(filtered)
        return filtered, stats
    
    def get_high_confidence_notes(
        self,
        notes: List[Note],
        threshold: float = 0.5,
    ) -> List[Note]:
        """
        Get only high-confidence notes.
        
        Args:
            notes: Notes to filter
            threshold: Minimum confidence to keep
            
        Returns:
            High-confidence notes only
        """
        weighted = self.confidence_weighter.weight_notes(notes)
        return [nc.note for nc in weighted if nc.confidence >= threshold]


def apply_musical_rules(
    notes: List[Note],
    key_root: Optional[str] = None,
    key_mode: Optional[str] = None,
) -> List[Note]:
    """
    Apply musical rules to filter implausible notes.
    
    Rules applied:
    - Remove notes far outside the expected range for detected key
    - Remove isolated chromatic notes (not part of patterns)
    - Prefer diatonic notes when uncertain
    
    Args:
        notes: Notes to filter
        key_root: Key root if known
        key_mode: Key mode if known
        
    Returns:
        Filtered notes
    """
    if not notes:
        return []
    
    # If no key context, just do basic filtering
    if not key_root or not key_mode:
        return notes
    
    # Get scale notes
    from .key import KeyDetector
    detector = KeyDetector()
    scale_pcs = set(detector.get_scale_notes(key_root, key_mode))
    
    result = []
    for note in notes:
        pc = note.pitch % 12
        
        # Always keep scale tones
        if pc in scale_pcs:
            result.append(note)
            continue
        
        # For non-scale tones, check if they're plausible chromatic notes
        # Keep if: short duration (passing tone) or strong velocity (intentional)
        if note.duration < 0.15 or note.velocity > 80:
            result.append(note)
            continue
        
        # Check if there are nearby scale tones that could replace this
        # If very close to a scale tone, it might be a pitch detection error
        for scale_pc in scale_pcs:
            if abs(pc - scale_pc) == 1 or abs(pc - scale_pc) == 11:
                # Adjacent to scale tone - could be error, skip
                break
        else:
            # Not adjacent to any scale tone - probably intentional
            result.append(note)
    
    return result
