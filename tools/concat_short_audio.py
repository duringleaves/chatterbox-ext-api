#!/usr/bin/env python3
"""
Concatenate short WAV files by naming pattern to create longer reference audio.

Given a directory of WAV files, this script:
1. Groups files by prefix (e.g., social_*.wav, greeting_*.wav)
2. Identifies files below minimum duration threshold
3. Concatenates multiple files from same group until target duration is reached
4. Saves combined files with descriptive names
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import librosa
import soundfile as sf
import numpy as np


def get_duration(wav_path: Path) -> float:
    """Get duration of WAV file in seconds."""
    try:
        y, sr = librosa.load(wav_path, sr=None)
        return len(y) / sr
    except Exception as e:
        print(f"Warning: Could not read {wav_path}: {e}")
        return 0.0


def extract_prefix(filename: str) -> str:
    """
    Extract the common prefix from a filename.
    
    Examples:
        social_001.wav -> social
        greeting_hello_01.wav -> greeting_hello
        voice_sample_a.wav -> voice_sample
    """
    # Remove extension
    name = Path(filename).stem
    
    # Try to find pattern: text followed by underscore and numbers/single letters
    match = re.match(r'^(.+?)_[0-9]+[a-z]?$', name, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Try pattern: text followed by underscore and single letter
    match = re.match(r'^(.+?)_[a-z]$', name, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # No clear pattern, use the whole name
    return name


def group_files_by_prefix(wav_files: List[Path]) -> Dict[str, List[Path]]:
    """Group WAV files by their common prefix."""
    groups = defaultdict(list)
    
    for wav_file in wav_files:
        prefix = extract_prefix(wav_file.name)
        groups[prefix].append(wav_file)
    
    # Sort files within each group for consistent ordering
    for prefix in groups:
        groups[prefix].sort()
    
    return dict(groups)


def concatenate_audio_files(file_paths: List[Path], target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """
    Concatenate multiple audio files into a single array.
    
    Args:
        file_paths: List of paths to WAV files to concatenate
        target_sr: Target sample rate for output (default 24kHz for S3GEN_SR)
    
    Returns:
        Tuple of (concatenated audio array, sample rate)
    """
    audio_segments = []
    
    for fpath in file_paths:
        try:
            y, sr = librosa.load(fpath, sr=target_sr)
            audio_segments.append(y)
        except Exception as e:
            print(f"Warning: Skipping {fpath}: {e}")
            continue
    
    if not audio_segments:
        raise ValueError("No valid audio files to concatenate")
    
    # Concatenate all segments
    combined = np.concatenate(audio_segments)
    
    return combined, target_sr


def process_group(
    prefix: str,
    files: List[Path],
    min_duration: float,
    target_duration: float,
    output_dir: Path,
    target_sr: int = 24000,
    add_silence: bool = False,
    silence_duration: float = 0.1
) -> List[Path]:
    """
    Process a group of files with the same prefix.
    
    Args:
        prefix: Common prefix for this group
        files: List of file paths in this group
        min_duration: Minimum duration threshold (seconds)
        target_duration: Target duration to reach (seconds)
        output_dir: Directory to save concatenated files
        target_sr: Target sample rate
        add_silence: Whether to add silence between concatenated clips
        silence_duration: Duration of silence to add between clips (seconds)
    
    Returns:
        List of paths to created output files
    """
    # Get durations for all files
    file_durations = [(f, get_duration(f)) for f in files]
    file_durations = [(f, d) for f, d in file_durations if d > 0]  # Filter invalid
    
    if not file_durations:
        print(f"No valid files in group '{prefix}'")
        return []
    
    # Separate into short and long files
    short_files = [(f, d) for f, d in file_durations if d < min_duration]
    long_files = [(f, d) for f, d in file_durations if d >= min_duration]
    
    output_files = []
    
    # Copy long files as-is (they already meet the threshold)
    for fpath, duration in long_files:
        output_path = output_dir / fpath.name
        try:
            y, sr = librosa.load(fpath, sr=target_sr)
            sf.write(output_path, y, sr)
            print(f"✓ Copied {fpath.name} ({duration:.2f}s) -> {output_path.name}")
            output_files.append(output_path)
        except Exception as e:
            print(f"Warning: Could not copy {fpath}: {e}")
    
    # Concatenate short files
    if short_files:
        print(f"\nProcessing {len(short_files)} short files in group '{prefix}':")
        for f, d in short_files:
            print(f"  - {f.name}: {d:.2f}s")
        
        # Greedy approach: combine files until we reach target duration
        remaining = list(short_files)
        combo_index = 1
        
        while remaining:
            current_combo = []
            current_duration = 0.0
            
            # Add files until we reach target duration
            while remaining and current_duration < target_duration:
                fpath, duration = remaining.pop(0)
                current_combo.append(fpath)
                current_duration += duration
                
                # Add silence duration if configured and not the last file
                if add_silence and remaining and current_duration < target_duration:
                    current_duration += silence_duration
            
            # Create output filename
            if len(current_combo) == 1:
                # Single file, keep original name
                output_name = current_combo[0].name
            else:
                # Multiple files, create descriptive name
                output_name = f"{prefix}_{combo_index:02d}.wav"
                combo_index += 1
            
            output_path = output_dir / output_name
            
            # Concatenate the audio
            try:
                if add_silence and len(current_combo) > 1:
                    # Add silence between clips
                    audio_segments = []
                    silence = np.zeros(int(silence_duration * target_sr))
                    
                    for i, fpath in enumerate(current_combo):
                        y, _ = librosa.load(fpath, sr=target_sr)
                        audio_segments.append(y)
                        if i < len(current_combo) - 1:  # Don't add silence after last clip
                            audio_segments.append(silence)
                    
                    combined_audio = np.concatenate(audio_segments)
                else:
                    # Direct concatenation
                    combined_audio, _ = concatenate_audio_files(current_combo, target_sr)
                
                sf.write(output_path, combined_audio, target_sr)
                
                actual_duration = len(combined_audio) / target_sr
                file_list = " + ".join([f.name for f in current_combo])
                print(f"✓ Created {output_name} ({actual_duration:.2f}s) from {len(current_combo)} files")
                print(f"    {file_list}")
                
                output_files.append(output_path)
                
            except Exception as e:
                print(f"Error creating {output_name}: {e}")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate short WAV files by naming pattern to create longer reference audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine files under 5s to reach 10s target
  python concat_reference_audio.py ./voices --min-duration 5 --target-duration 10
  
  # Add 0.2s silence between concatenated clips
  python concat_reference_audio.py ./voices --min-duration 5 --target-duration 10 --add-silence --silence-duration 0.2
  
  # Specify custom output directory
  python concat_reference_audio.py ./voices --output ./processed_voices --min-duration 3 --target-duration 8
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing WAV files to process'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output directory (default: <input_dir>_processed)'
    )
    
    parser.add_argument(
        '--min-duration',
        type=float,
        default=5.0,
        help='Minimum duration threshold in seconds (default: 5.0)'
    )
    
    parser.add_argument(
        '--target-duration',
        type=float,
        default=10.0,
        help='Target duration to reach when concatenating in seconds (default: 10.0)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=24000,
        help='Target sample rate for output files (default: 24000)'
    )
    
    parser.add_argument(
        '--add-silence',
        action='store_true',
        help='Add silence between concatenated clips'
    )
    
    parser.add_argument(
        '--silence-duration',
        type=float,
        default=0.1,
        help='Duration of silence to add between clips in seconds (default: 0.1)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing files'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    if not args.input_dir.is_dir():
        print(f"Error: '{args.input_dir}' is not a directory")
        return 1
    
    # Set up output directory
    if args.output is None:
        # Default: create parallel directory with "_processed" suffix
        output_dir = args.input_dir.parent / f"{args.input_dir.name}_processed"
    else:
        output_dir = args.output
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created: {output_dir}")
    
    # Find all WAV files
    wav_files = list(args.input_dir.glob('*.wav')) + list(args.input_dir.glob('*.WAV'))
    
    if not wav_files:
        print(f"No WAV files found in '{args.input_dir}'")
        return 1
    
    print(f"Found {len(wav_files)} WAV files in '{args.input_dir}'")
    print(f"Min duration threshold: {args.min_duration}s")
    print(f"Target duration: {args.target_duration}s")
    print(f"Output directory: {output_dir}")
    if args.add_silence:
        print(f"Adding {args.silence_duration}s silence between clips")
    print()
    
    # Group files by prefix
    groups = group_files_by_prefix(wav_files)
    print(f"Grouped into {len(groups)} naming patterns:\n")
    
    for prefix, files in groups.items():
        print(f"  {prefix}: {len(files)} files")
    
    print()
    
    if args.dry_run:
        print("DRY RUN - No files will be modified\n")
    
    # Process each group
    total_output_files = 0
    for prefix, files in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Processing group: {prefix}")
        print(f"{'='*60}")
        
        if args.dry_run:
            # Just show what would happen
            file_durations = [(f, get_duration(f)) for f in files]
            short_files = [(f, d) for f, d in file_durations if d < args.min_duration and d > 0]
            long_files = [(f, d) for f, d in file_durations if d >= args.min_duration]
            
            print(f"Would copy {len(long_files)} files that meet threshold")
            print(f"Would concatenate {len(short_files)} short files")
            
        else:
            output_files = process_group(
                prefix=prefix,
                files=files,
                min_duration=args.min_duration,
                target_duration=args.target_duration,
                output_dir=output_dir,
                target_sr=args.sample_rate,
                add_silence=args.add_silence,
                silence_duration=args.silence_duration
            )
            total_output_files += len(output_files)
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print("DRY RUN COMPLETE - No files were modified")
    else:
        print(f"✓ Processing complete!")
        print(f"Created {total_output_files} output files in '{output_dir}'")
    
    return 0


if __name__ == '__main__':
    exit(main())