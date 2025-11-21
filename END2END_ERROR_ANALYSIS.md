# End2End Error Analysis and Fix Plan

## Error Message
```
[Length 1024] Error: The size of tensor a (0) must match the size of tensor b (1024) at non-singleton dimension 2
```

## Problem Analysis

### Root Cause Hypothesis
The error "The size of tensor a (0) must match the size of tensor b (1024) at non-singleton dimension 2" typically occurs in attention computation when two tensors have mismatched sequence lengths. The error indicates:
- One tensor has sequence length 0
- Another tensor has sequence length 1024 (original input length)

### Possible Causes

1. **In `prefill_with_pruning` function:**
   - Although we fixed `apply_token_pruning` to guarantee minimum token retention, there might be edge cases where:
     - A layer forward pass produces zero-length output
     - Position IDs are created with zero length
     - Attention computation fails due to dimension mismatch

2. **In `run_end2end_sdtp` function:**
   - After `prefill_with_pruning`, we call `model.generate()` with original `input_ids`
   - If `prefill_with_pruning` somehow modified model state, `model.generate()` might fail
   - However, `prefill_with_pruning` should not modify model state (it's a manual forward)

3. **Edge Case in Pruning Logic:**
   - When sequence length becomes very small after multiple pruning steps
   - Some calculation might result in zero-length tensor
   - Position IDs or attention masks might become invalid

## Fixes Applied

### 1. Added Safety Checks in `prefill_with_pruning`
- Check sequence length before each layer forward
- Check sequence length after each layer forward
- Check sequence length after each pruning step
- Check final output before returning

### 2. Enhanced Error Messages
- Added detailed error messages with layer index and sequence lengths
- Wrapped layer forward in try-except to catch and re-raise with context

### 3. Final Output Validation
- Verify `hidden_states` sequence length before final norm+lm_head
- Verify `logits` sequence length before returning

## Next Steps

1. **Run with debug logging enabled** to see which layer fails
2. **Check debug/prune_log.jsonl** for pruning statistics
3. **Add more detailed error tracking** if needed

## Testing Plan

After applying fixes, test with:
```bash
bash scripts/run_inference.sh profile end2end
```

Expected behavior:
- Should not produce "tensor a (0)" errors
- Debug logs should show pruning statistics for each layer
- If error occurs, should show detailed information about which layer failed

