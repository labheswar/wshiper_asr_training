#!/usr/bin/env python3
"""
Apply Remaining Distributed Training Changes to train_asr.py

This script helps apply the remaining manual changes needed for distributed training.
Run this after reviewing the changes in MANUAL_UPDATES_train_asr.py
"""

import sys
from pathlib import Path

# Script to show what changes are still needed
SCRIPT_DIR = Path(__file__).parent
TRAIN_ASR_PATH = SCRIPT_DIR / "scripts" / "train_asr.py"

print("="*80)
print("DISTRIBUTED TRAINING - REMAINING CHANGES CHECK")
print("="*80)

if not TRAIN_ASR_PATH.exists():
    print(f"❌ ERROR: {TRAIN_ASR_PATH} not found!")
    sys.exit(1)

with open(TRAIN_ASR_PATH, 'r') as f:
    content = f.read()

print("\nChecking for required changes...\n")

checks = []

# Check 1: Distributed import
if "import torch.distributed as dist" in content:
    print("✅ 1. torch.distributed imported")
    checks.append(True)
else:
    print("❌ 1. torch.distributed NOT imported")
    print("   Add: import torch.distributed as dist")
    checks.append(False)

# Check 2: setup_distributed function
if "def setup_distributed():" in content:
    print("✅ 2. setup_distributed() function defined")
    checks.append(True)
else:
    print("❌ 2. setup_distributed() function NOT found")
    print("   See MANUAL_UPDATES_train_asr.py for implementation")
    checks.append(False)

# Check 3: cleanup_distributed function
if "def cleanup_distributed():" in content:
    print("✅ 3. cleanup_distributed() function defined")
    checks.append(True)
else:
    print("❌ 3. cleanup_distributed() function NOT found")
    print("   See MANUAL_UPDATES_train_asr.py for implementation")
    checks.append(False)

# Check 4: Trainer initialization
if "self.dist_info = setup_distributed()" in content:
    print("✅ 4. Trainer.__init__() calls setup_distributed()")
    checks.append(True)
else:
    print("❌ 4. Trainer.__init__() does NOT call setup_distributed()")
    print("   Add: self.dist_info = setup_distributed() at start of __init__")
    checks.append(False)

# Check 5: Device binding
if "torch.cuda.set_device(local_rank)" in content:
    print("✅ 5. Device binding with torch.cuda.set_device()")
    checks.append(True)
else:
    print("❌ 5. Device binding NOT implemented")
    print("   See MANUAL_UPDATES_train_asr.py section 1")
    checks.append(False)

# Check 6: Distributed device_map
if "device_map = {'': self.dist_info['local_rank']}" in content or "device_map = {'': local_rank}" in content:
    print("✅ 6. Model loading with distributed device_map")
    checks.append(True)
else:
    print("❌ 6. Model loading device_map NOT updated for distributed")
    print("   See MANUAL_UPDATES_train_asr.py section 3")
    checks.append(False)

# Check 7: DDP arguments
if "local_rank=self.dist_info['local_rank']" in content:
    print("✅ 7. TrainingArguments include local_rank")
    checks.append(True)
else:
    print("❌ 7. TrainingArguments missing DDP settings")
    print("   See MANUAL_UPDATES_train_asr.py section 4")
    checks.append(False)

# Check 8: Distributed barriers
if "dist.barrier()" in content:
    print("✅ 8. Process synchronization barriers present")
    checks.append(True)
else:
    print("❌ 8. No process synchronization barriers found")
    print("   See MANUAL_UPDATES_train_asr.py section 6")
    checks.append(False)

# Check 9: Cleanup on exit
if "cleanup_distributed()" in content:
    print("✅ 9. cleanup_distributed() called in run() method")
    checks.append(True)
else:
    print("❌ 9. cleanup_distributed() NOT called")
    print("   See MANUAL_UPDATES_train_asr.py section 8")
    checks.append(False)

print("\n" + "="*80)
passed = sum(checks)
total = len(checks)
percentage = (passed / total) * 100

print(f"RESULTS: {passed}/{total} checks passed ({percentage:.0f}%)")

if passed == total:
    print("\n✅ ALL CHANGES APPLIED - Ready for distributed training!")
    print("\nNext steps:")
    print("1. Run validation: torchrun --nproc_per_node=4 --standalone validate_distributed.py")
    print("2. Test training: torchrun --nproc_per_node=2 --standalone scripts/train_asr.py --config config/training_config.yaml --max-parquet-files 2")
    sys.exit(0)
else:
    print(f"\n⚠️  {total - passed} changes still needed")
    print("\nRefer to these files for detailed instructions:")
    print("  - MANUAL_UPDATES_train_asr.py (line-by-line changes)")
    print("  - DISTRIBUTED_TRAINING_README.md (complete guide)")
    print("  - IMPLEMENTATION_SUMMARY.md (overview)")
    sys.exit(1)
