#!/usr/bin/env python3
"""
Distributed Training Validation Script
Tests if distributed training setup is working correctly
"""

import os
import sys
import torch
import torch.distributed as dist

def test_distributed_setup():
    """Test distributed training configuration"""
    print("="*80)
    print("DISTRIBUTED TRAINING VALIDATION")
    print("="*80)
    
    # Check environment variables
    print("\n1. Environment Variables:")
    print(f"   LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"   RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"   WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"   MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"   MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Check CUDA
    print("\n2. CUDA Availability:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   NCCL Available: {dist.is_nccl_available()}")
        if dist.is_nccl_available():
            print(f"   NCCL Version: {torch.cuda.nccl.version()}")
    
    # Test distributed initialization
    print("\n3. Testing Distributed Initialization:")
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    if local_rank == -1:
        print("   ⚠️  Not running in distributed mode (LOCAL_RANK not set)")
        print("   This is expected if running without torchrun")
        print("   To test distributed mode, run:")
        print("   torchrun --nproc_per_node=4 --standalone validate_distributed.py")
        return False
    
    try:
        # Set device
        torch.cuda.set_device(local_rank)
        print(f"   ✓ Device set to: cuda:{local_rank}")
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
            print(f"   ✓ Process group initialized")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f"   ✓ Rank: {rank}/{world_size}")
        print(f"   ✓ Local Rank (GPU): {local_rank}")
        print(f"   ✓ Backend: {dist.get_backend()}")
        
        # Test barrier synchronization
        print("\n4. Testing Process Synchronization:")
        dist.barrier()
        print(f"   ✓ Barrier synchronization successful")
        
        # Test tensor operations
        print("\n5. Testing GPU Tensor Operations:")
        test_tensor = torch.randn(10, 10).cuda(local_rank)
        print(f"   ✓ Created tensor on GPU {local_rank}: shape {test_tensor.shape}")
        
        # Test all_reduce
        print("\n6. Testing All-Reduce Communication:")
        test_value = torch.tensor([rank], dtype=torch.float32).cuda(local_rank)
        print(f"   Before all_reduce: {test_value.item()}")
        dist.all_reduce(test_value, op=dist.ReduceOp.SUM)
        print(f"   After all_reduce: {test_value.item()}")
        expected_sum = sum(range(world_size))
        if test_value.item() == expected_sum:
            print(f"   ✓ All-reduce successful (expected {expected_sum})")
        else:
            print(f"   ❌ All-reduce failed (expected {expected_sum}, got {test_value.item()})")
            return False
        
        # Cleanup
        dist.barrier()
        dist.destroy_process_group()
        print("\n7. Cleanup:")
        print("   ✓ Process group destroyed")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print(f"\nProcess rank {rank} on GPU {local_rank} is ready for distributed training!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_distributed_setup()
    
    if not success:
        print("\n" + "="*80)
        print("TROUBLESHOOTING TIPS")
        print("="*80)
        print("1. Make sure to run with torchrun:")
        print("   torchrun --nproc_per_node=4 --standalone validate_distributed.py")
        print("\n2. Check NCCL environment variables:")
        print("   export NCCL_DEBUG=INFO")
        print("   export NCCL_P2P_LEVEL=NVL")
        print("\n3. Verify all GPUs are accessible:")
        print("   nvidia-smi")
        print("\n4. Check CUDA_VISIBLE_DEVICES:")
        print("   echo $CUDA_VISIBLE_DEVICES")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
