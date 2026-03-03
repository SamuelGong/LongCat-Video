#!/bin/bash

# Script to compare performance between standard and KV cache versions
# Usage: ./compare_kv_cache_performance.sh

set -e  # Exit on error

# Default parameters (can be overridden via environment variables)
INPUT_VIDEO="${INPUT_VIDEO:-assets/motorcycle.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_comparison}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./weights/LongCat-Video}"
PROMPT="${PROMPT:-A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides.}"
TOTAL_TARGET_FRAMES="${TOTAL_TARGET_FRAMES:-500}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-13}"
FRAMES_PER_SEGMENT="${FRAMES_PER_SEGMENT:-80}"
RESOLUTION="${RESOLUTION:-480p}"
TARGET_FPS="${TARGET_FPS:-15}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
SEED="${SEED:-42}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Output file names
OUTPUT_STANDARD="${OUTPUT_DIR}/output_standard.mp4"
OUTPUT_KV="${OUTPUT_DIR}/output_kv.mp4"

echo "=========================================="
echo "KV Cache Performance Comparison"
echo "=========================================="
echo "Input video: $INPUT_VIDEO"
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Total target frames: $TOTAL_TARGET_FRAMES"
echo "Number of GPUs: $NPROC_PER_NODE"
echo "=========================================="
echo ""

# Function to run a script and capture timing
run_with_timing() {
    local script_name=$1
    local output_file=$2
    local log_file=$3
    
    echo "Running: $script_name"
    echo "Output: $output_file"
    echo "Log: $log_file"
    echo ""
    
    # Run the script and capture output
    torchrun --nproc_per_node=$NPROC_PER_NODE "$script_name" \
        --input_video "$INPUT_VIDEO" \
        --output_path "$output_file" \
        --prompt "$PROMPT" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --total_target_frames "$TOTAL_TARGET_FRAMES" \
        --num_cond_frames "$NUM_COND_FRAMES" \
        --frames_per_segment "$FRAMES_PER_SEGMENT" \
        --resolution "$RESOLUTION" \
        --target_fps "$TARGET_FPS" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --seed "$SEED" \
        --context_parallel_size "$CONTEXT_PARALLEL_SIZE" \
        --enable_compile \
        2>&1 | tee "$log_file"
    
    echo ""
}

# Run standard version
echo "=========================================="
echo "Step 1: Running STANDARD version (without KV cache reuse)"
echo "=========================================="
STANDARD_LOG="${OUTPUT_DIR}/standard.log"
run_with_timing "run_demo_long_video_continuation.py" "$OUTPUT_STANDARD" "$STANDARD_LOG"

# Wait a bit between runs
echo "Waiting 10 seconds before next run..."
sleep 10

# Run KV cache version
echo "=========================================="
echo "Step 2: Running KV CACHE version (with KV cache reuse)"
echo "=========================================="
KV_LOG="${OUTPUT_DIR}/kv.log"
run_with_timing "run_demo_long_video_continuation_kv.py" "$OUTPUT_KV" "$KV_LOG"

# Extract timing information from JSON files
STANDARD_TIMING="${OUTPUT_STANDARD%.mp4}_timing.json"
KV_TIMING="${OUTPUT_KV%.mp4}_timing.json"

# Compare results
echo ""
echo "=========================================="
echo "Performance Comparison Summary"
echo "=========================================="

if [ -f "$STANDARD_TIMING" ] && [ -f "$KV_TIMING" ]; then
    echo ""
    echo "Standard version timing:"
    python3 << EOF
import json
with open('$STANDARD_TIMING', 'r') as f:
    data = json.load(f)
    print(f"  Total time: {data['total_time']:.2f}s")
    print(f"  Model loading: {data['model_loading_time']:.2f}s")
    print(f"  Video loading: {data['video_loading_time']:.2f}s")
    print(f"  Number of segments: {len(data['segments'])}")
    if data['segments']:
        avg_gen_time = sum(s['generation_time'] for s in data['segments']) / len(data['segments'])
        print(f"  Average generation time per segment: {avg_gen_time:.2f}s")
EOF

    echo ""
    echo "KV cache version timing:"
    python3 << EOF
import json
with open('$KV_TIMING', 'r') as f:
    data = json.load(f)
    print(f"  Total time: {data['total_time']:.2f}s")
    print(f"  Model loading: {data['model_loading_time']:.2f}s")
    print(f"  Video loading: {data['video_loading_time']:.2f}s")
    print(f"  Number of segments: {len(data['segments'])}")
    print(f"  KV cache reused in segments: {data.get('kv_cache_reused_segments', [])}")
    if data['segments']:
        avg_gen_time = sum(s['generation_time'] for s in data['segments']) / len(data['segments'])
        print(f"  Average generation time per segment: {avg_gen_time:.2f}s")
EOF

    echo ""
    echo "Speedup calculation:"
    python3 << EOF
import json
with open('$STANDARD_TIMING', 'r') as f:
    standard = json.load(f)
with open('$KV_TIMING', 'r') as f:
    kv = json.load(f)
    
standard_total = standard['total_time']
kv_total = kv['total_time']
speedup = standard_total / kv_total if kv_total > 0 else 0

print(f"  Standard total time: {standard_total:.2f}s")
print(f"  KV cache total time: {kv_total:.2f}s")
print(f"  Speedup: {speedup:.2f}x")
print(f"  Time saved: {standard_total - kv_total:.2f}s ({((standard_total - kv_total) / standard_total * 100):.1f}%)")

# Compare per-segment times
if standard['segments'] and kv['segments']:
    print(f"\n  Per-segment comparison:")
    for i, (s_seg, k_seg) in enumerate(zip(standard['segments'], kv['segments']), 1):
        s_time = s_seg['generation_time']
        k_time = k_seg['generation_time']
        seg_speedup = s_time / k_time if k_time > 0 else 0
        reused = k_seg.get('kv_cache_reused', False)
        print(f"    Segment {i}: {s_time:.2f}s -> {k_time:.2f}s ({seg_speedup:.2f}x) {'[KV reused]' if reused else ''}")
EOF

else
    echo "Warning: Timing JSON files not found. Check logs for details."
    echo "Standard log: $STANDARD_LOG"
    echo "KV log: $KV_LOG"
fi

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
echo "Output files:"
echo "  Standard: $OUTPUT_STANDARD"
echo "  KV cache: $OUTPUT_KV"
echo "Timing files:"
echo "  Standard: $STANDARD_TIMING"
echo "  KV cache: $KV_TIMING"
echo "Log files:"
echo "  Standard: $STANDARD_LOG"
echo "  KV cache: $KV_LOG"
echo "=========================================="
