#!/usr/bin/env python3
"""Advanced lock analysis for glurpc logs - tracks deltas and highlights issues."""

import re
from collections import defaultdict
from pathlib import Path
import sys

class LockTracker:
    def __init__(self, lock_id: str):
        self.lock_id = lock_id
        self.acquiring = 0
        self.acquired = 0
        self.releasing = 0
        self.timeline = []  # List of (line_num, operation, counts_snapshot, line_text)
        
    @property
    def acquire_delta(self) -> int:
        """How many acquire attempts are pending (acquiring - acquired)"""
        return self.acquiring - self.acquired
    
    @property
    def hold_delta(self) -> int:
        """How many locks are currently held (acquired - releasing)"""
        return self.acquired - self.releasing
    
    def record(self, line_num: int, operation: str, line_text: str):
        """Record an operation and check for anomalies."""
        if operation == 'Acquiring':
            self.acquiring += 1
        elif operation == 'Acquired':
            self.acquired += 1
        elif operation == 'Releasing':
            self.releasing += 1
        
        snapshot = {
            'acquiring': self.acquiring,
            'acquired': self.acquired,
            'releasing': self.releasing,
            'acquire_delta': self.acquire_delta,
            'hold_delta': self.hold_delta,
        }
        
        self.timeline.append((line_num, operation, snapshot, line_text))
        
        # Flag if deltas are suspicious
        return abs(self.acquire_delta) > 2 or abs(self.hold_delta) > 2
    
    def get_summary(self):
        return {
            'acquiring': self.acquiring,
            'acquired': self.acquired,
            'releasing': self.releasing,
            'acquire_delta': self.acquire_delta,
            'hold_delta': self.hold_delta,
            'timeline_len': len(self.timeline),
        }

def analyze_locks_advanced(log_file: Path, show_all_timeline: bool = False):
    """Parse log and track lock operations with detailed timeline."""
    
    # Track per lock type
    lock_trackers = defaultdict(lambda: LockTracker(""))
    
    # Patterns to match
    acquiring_pattern = re.compile(r'Acquiring (?:asyncio )?lock for (\w+)')
    acquired_pattern = re.compile(r'Acquired (?:asyncio )?lock for (\w+)')
    releasing_pattern = re.compile(r'Releasing (?:asyncio )?lock for (\w+)')
    
    # Also track by specific keys (e.g., handle:index combinations)
    detailed_pattern = re.compile(r'(Acquiring|Acquired|Releasing) (?:asyncio )?lock for (\w+).*?(?:key=|handle=)([^\s,)]+)')
    
    print(f"Analyzing: {log_file}")
    print("=" * 100)
    
    anomaly_lines = []  # Lines where delta exceeded threshold
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip non-lock lines
            if 'lock' not in line.lower():
                continue
                
            # Try detailed pattern first (with key/handle)
            detailed_match = detailed_pattern.search(line)
            if detailed_match:
                operation = detailed_match.group(1)
                lock_type = detailed_match.group(2)
                key = detailed_match.group(3)
                lock_id = f"{lock_type}:{key}"
            else:
                # Try simple pattern
                if 'Acquiring' in line:
                    match = acquiring_pattern.search(line)
                    if match:
                        lock_id = match.group(1)
                        operation = 'Acquiring'
                    else:
                        continue
                elif 'Acquired' in line:
                    match = acquired_pattern.search(line)
                    if match:
                        lock_id = match.group(1)
                        operation = 'Acquired'
                    else:
                        continue
                elif 'Releasing' in line:
                    match = releasing_pattern.search(line)
                    if match:
                        lock_id = match.group(1)
                        operation = 'Releasing'
                    else:
                        continue
                else:
                    continue
            
            # Get or create tracker
            if lock_id not in lock_trackers:
                lock_trackers[lock_id] = LockTracker(lock_id)
            
            tracker = lock_trackers[lock_id]
            
            # Record operation and check for anomaly
            is_anomaly = tracker.record(line_num, operation, line.strip())
            
            if is_anomaly:
                anomaly_lines.append((line_num, lock_id, operation, tracker.get_summary(), line.strip()))
    
    # Print anomalies first
    if anomaly_lines:
        print("\n" + "=" * 100)
        print(f"⚠️  FOUND {len(anomaly_lines)} ANOMALOUS LOCK OPERATIONS (delta > 2)")
        print("=" * 100)
        
        for line_num, lock_id, operation, summary, line_text in anomaly_lines[:50]:  # Show first 50
            acq_delta = summary['acquire_delta']
            hold_delta = summary['hold_delta']
            
            status = ""
            if abs(acq_delta) > 2:
                status += f"ACQUIRE_DELTA={acq_delta:+d} "
            if abs(hold_delta) > 2:
                status += f"HOLD_DELTA={hold_delta:+d} "
            
            print(f"\nLine {line_num}: [{status.strip()}]")
            print(f"  Lock: {lock_id}")
            print(f"  Operation: {operation}")
            print(f"  Counts: acquiring={summary['acquiring']}, acquired={summary['acquired']}, releasing={summary['releasing']}")
            print(f"  Text: {line_text[:120]}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("LOCK SUMMARY - EXACT COUNTS")
    print("=" * 100)
    
    problems = []
    perfect_locks = []
    
    for lock_id in sorted(lock_trackers.keys()):
        tracker = lock_trackers[lock_id]
        summary = tracker.get_summary()
        
        # Check for issues
        has_issue = False
        issues = []
        
        if summary['hold_delta'] != 0:
            has_issue = True
            issues.append(f"HOLD_DELTA={summary['hold_delta']:+d}")
        
        if summary['acquire_delta'] != 0:
            has_issue = True
            issues.append(f"ACQUIRE_DELTA={summary['acquire_delta']:+d}")
        
        if summary['releasing'] > summary['acquired']:
            has_issue = True
            issues.append("OVER_RELEASED")
        
        if has_issue:
            problems.append((lock_id, summary, issues))
        else:
            perfect_locks.append(lock_id)
    
    # Show problems
    if problems:
        print(f"\n⚠️  {len(problems)} locks with imbalances:\n")
        for lock_id, summary, issues in problems:
            print(f"  {lock_id}")
            print(f"    Acquiring: {summary['acquiring']}")
            print(f"    Acquired:  {summary['acquired']}")
            print(f"    Releasing: {summary['releasing']}")
            print(f"    Issues:    {', '.join(issues)}")
    
    print(f"\n✅ {len(perfect_locks)} locks perfectly balanced")

    # Print exact counts for balanced locks too (requested)
    if perfect_locks:
        print("\nBalanced lock stats:")
        header = (
            f"{'lock_id':60s}  "
            f"{'acquiring':>9s}  {'acquired':>8s}  {'releasing':>9s}  "
            f"{'Δacq':>5s}  {'Δhold':>6s}"
        )
        print(header)
        print("-" * len(header))
        for lock_id in perfect_locks:
            summary = lock_trackers[lock_id].get_summary()
            print(
                f"{lock_id[:60]:60s}  "
                f"{summary['acquiring']:9d}  {summary['acquired']:8d}  {summary['releasing']:9d}  "
                f"{summary['acquire_delta']:+5d}  {summary['hold_delta']:+6d}"
            )
    
    # Show timeline for problematic locks
    if problems and show_all_timeline:
        print("\n" + "=" * 100)
        print("DETAILED TIMELINE FOR PROBLEMATIC LOCKS")
        print("=" * 100)
        
        for lock_id, summary, issues in problems[:5]:  # Show first 5
            tracker = lock_trackers[lock_id]
            print(f"\n{lock_id}:")
            print(f"  Timeline ({len(tracker.timeline)} events):")
            
            for line_num, operation, snapshot, line_text in tracker.timeline[-20:]:  # Last 20 events
                acq_delta = snapshot['acquire_delta']
                hold_delta = snapshot['hold_delta']
                flag = "⚠️ " if abs(acq_delta) > 2 or abs(hold_delta) > 2 else "   "
                
                print(f"    {flag}Line {line_num:6d}: {operation:10s} "
                      f"[acq:{snapshot['acquiring']:2d} got:{snapshot['acquired']:2d} rel:{snapshot['releasing']:2d}] "
                      f"[Δacq:{acq_delta:+2d} Δhold:{hold_delta:+2d}]")
    
    print("\n" + "=" * 100)
    
    if problems:
        print(f"❌ FOUND {len(problems)} LOCKS WITH ISSUES")
        print(f"⚠️  FOUND {len(anomaly_lines)} LINES WITH SUSPICIOUS DELTAS")
    else:
        print("✅ All locks properly balanced!")
    
    print("=" * 100)
    
    return problems, anomaly_lines

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced lock analysis for glurpc logs')
    parser.add_argument('log_file', nargs='?', help='Path to log file')
    parser.add_argument('--timeline', '-t', action='store_true', help='Show detailed timeline for problematic locks')
    args = parser.parse_args()
    
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        # Find the most recent log
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("Error: logs directory not found")
            sys.exit(1)
        
        log_files = sorted(logs_dir.glob("glurpc_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("Error: no log files found")
            sys.exit(1)
        
        log_file = log_files[0]
    
    problems, anomaly_lines = analyze_locks_advanced(log_file, show_all_timeline=args.timeline)
    
    sys.exit(len(problems))
