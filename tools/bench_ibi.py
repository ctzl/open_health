"""Benchmark: pure-Python IbiCorrection vs the Rust ibi-correct binary.

Loads every golden vector, concatenates the streams into one big input
(timing only, state carries across boundaries), times the pure-Python
reference beat-by-beat, and writes the combined input to combined.json
for the Rust side to be timed against separately.
"""
import json, sys, time, glob
sys.path.insert(0, 'tools')
from ibi_correction import IbiCorrection

streams = []
for p in sorted(glob.glob('crates/ibi-correct/tests/vectors/*.json')):
    streams += json.load(open(p))['input']
n = len(streams)
print(f'beats: {n} across 25 vectors')

with open('combined.json', 'w') as f:
    json.dump(streams, f)

C = IbiCorrection()
t0 = time.perf_counter()
for ts, ibi, amp in streams:
    C.run(int(ibi), int(amp), int(ts))
t1 = time.perf_counter()
print(f'python: {t1-t0:.3f} s  ({(t1-t0)/n*1e6:.1f} us/beat)')
