"""Compare a candidate WAV with a known-good reference WAV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.reshape(-1), int(sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference, reference_rate = load(args.reference)
    candidate, candidate_rate = load(args.candidate)
    count = min(reference.size, candidate.size)
    left, right = reference[:count], candidate[:count]
    difference = right - left
    correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if count > 1 and np.std(left) and np.std(right)
        else 0.0
    )
    print(
        json.dumps(
            {
                "reference_sample_rate": reference_rate,
                "candidate_sample_rate": candidate_rate,
                "reference_samples": int(reference.size),
                "candidate_samples": int(candidate.size),
                "exact_sample_match": bool(
                    reference_rate == candidate_rate
                    and np.array_equal(reference, candidate)
                ),
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference * difference))),
                "max_absolute_error": float(np.max(np.abs(difference))),
                "waveform_correlation": correlation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
