"""
Run All Emotion Model Tests
============================
Runs both face and voice emotion test scripts and prints a combined summary.

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --face-model dual --voice-model balanced
    python tests/run_all_tests.py --limit 300
"""

import sys
import os
import subprocess
import argparse
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(script_name, extra_args=None):
    """
    Run a test script as a subprocess and capture its output.
    Returns (returncode, stdout_text, elapsed_seconds).
    """
    script_path = os.path.join(ROOT, 'tests', script_name)
    cmd = [sys.executable, script_path] + (extra_args or [])

    print(f"\n{'='*60}")
    print(f"  Running: {script_name}")
    print(f"  Args: {' '.join(extra_args or [])}")
    print(f"{'='*60}")

    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=False,   # Stream output live to terminal
        text=True,
    )
    elapsed = time.time() - start

    return proc.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all emotion model tests")
    parser.add_argument(
        '--face-model', default='dual',
        choices=['main', 'specialist', 'both', 'dual'],
        help='Face model variant (default: dual)'
    )
    parser.add_argument(
        '--voice-model', default='balanced',
        choices=['balanced', 'improved', 'original', 'all'],
        help='Voice model variant (default: balanced)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit samples per test (e.g. --limit 500)'
    )
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cuda', 'cpu'],
        help='Device for inference'
    )
    args = parser.parse_args()

    face_args = ['--model', args.face_model, '--device', args.device]
    voice_args = ['--model', args.voice_model, '--device', args.device]

    if args.limit:
        face_args += ['--limit', str(args.limit)]
        voice_args += ['--limit', str(args.limit)]

    print("\n" + "#" * 60)
    print("#  PSYCHOLOGIST AI — FULL MODEL TEST SUITE")
    print("#" * 60)

    # --- Face Emotion Test ---
    face_rc, face_time = run_script('test_face_emotion.py', face_args)

    # --- Voice Emotion Test ---
    voice_rc, voice_time = run_script('test_voice_emotion.py', voice_args)

    # --- Summary ---
    print("\n" + "#" * 60)
    print("#  SUITE SUMMARY")
    print("#" * 60)
    print(f"  {'Test':<25}  {'Status':>8}  {'Time':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}")
    print(f"  {'Face Emotion':<25}  {'PASS' if face_rc == 0 else 'FAIL':>8}  {face_time:>6.1f}s")
    print(f"  {'Voice Emotion':<25}  {'PASS' if voice_rc == 0 else 'FAIL':>8}  {voice_time:>6.1f}s")
    print(f"\n  Total time: {face_time + voice_time:.1f}s")

    overall = 0 if (face_rc == 0 and voice_rc == 0) else 1
    if overall == 0:
        print("\n  All tests PASSED.\n")
    else:
        print("\n  One or more tests FAILED.\n")

    sys.exit(overall)


if __name__ == '__main__':
    main()
