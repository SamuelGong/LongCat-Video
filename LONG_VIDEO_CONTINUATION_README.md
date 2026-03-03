# 长视频延续脚本使用说明

## 概述

`run_demo_long_video_continuation.py` 实现了基于固定上文窗口的长视频延续功能。该脚本按块生成视频，每个块使用固定数量的条件帧（默认13帧）来生成新帧（默认80帧），从而实现超长视频的生成。

## 核心特性

1. **固定上文窗口**: 始终使用13帧作为条件帧（可配置）
2. **按块生成**: 每个segment生成80帧新帧（可配置）
3. **自动循环**: 自动计算需要的segment数量
4. **智能裁剪**: 最后一块如果不足80帧，会先生成80帧再裁剪到目标长度
5. **总帧数可配置**: 通过 `--total_target_frames` 参数指定最终视频的总帧数

## 工作流程

```
输入视频: [13帧] (条件帧)
  ↓ Segment 1
生成: [13条件帧 + 80新帧] = 93帧
保留: [80新帧]
  ↓ Segment 2 (使用Segment 1的最后13帧作为条件)
生成: [13条件帧 + 80新帧] = 93帧
保留: [80新帧]
  ↓ Segment 3
...
  ↓ 最后一块
生成: [13条件帧 + 80新帧] = 93帧
保留: [实际需要的帧数] (裁剪到目标长度)
```

## 使用方法

### 基本用法

```bash
torchrun --nproc_per_node=1 run_demo_long_video_continuation.py \
    --input_video assets/motorcycle.mp4 \
    --output_path output_long.mp4 \
    --prompt "A person rides a motorcycle along a long, straight road..." \
    --checkpoint_dir /path/to/checkpoint \
    --total_target_frames 500
```

### 完整参数示例

```bash
torchrun --nproc_per_node=1 run_demo_long_video_continuation.py \
    --input_video assets/motorcycle.mp4 \
    --output_path output_long.mp4 \
    --prompt "A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides." \
    --negative_prompt "Bright tones, overexposed, static, blurred details..." \
    --checkpoint_dir /path/to/checkpoint \
    --total_target_frames 500 \
    --num_cond_frames 13 \
    --frames_per_segment 80 \
    --resolution 480p \
    --target_fps 15 \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 42 \
    --context_parallel_size 1 \
    --enable_compile \
    --save_intermediate
```

## 参数说明

### 必需参数

- `--input_video`: 输入视频路径
- `--prompt`: 文本提示词
- `--checkpoint_dir`: 模型checkpoint目录路径
- `--total_target_frames`: 目标总帧数（最终视频的总帧数）

### 视频生成参数

- `--num_cond_frames` (default: 13): 条件帧数量（固定上文窗口大小）
- `--frames_per_segment` (default: 80): 每个segment生成的新帧数
- `--resolution` (default: "480p"): 视频分辨率，可选 "480p" 或 "720p"
- `--target_fps` (default: 15): 输出视频的目标帧率

### 生成控制参数

- `--num_inference_steps` (default: 50): 去噪步数
- `--guidance_scale` (default: 4.0): CFG引导强度
- `--seed` (default: 42): 随机种子

### 模型参数

- `--context_parallel_size` (default: 1): Context parallel大小
- `--enable_compile`: 启用torch.compile优化

### KV Cache选项

- `--offload_kv_cache`: 将KV cache offload到CPU（节省GPU显存）
- `--enhance_hf` (default: True): 启用高频增强

### 其他选项

- `--save_intermediate`: 保存每个segment的中间结果
- `--negative_prompt`: 负向提示词（可选）

## 示例场景

### 场景1: 生成500帧的长视频

```bash
torchrun --nproc_per_node=1 run_demo_long_video_continuation.py \
    --input_video input.mp4 \
    --output_path output_500frames.mp4 \
    --prompt "A beautiful landscape scene..." \
    --checkpoint_dir ./checkpoints \
    --total_target_frames 500 \
    --num_cond_frames 13 \
    --frames_per_segment 80
```

**计算过程**:
- 假设输入视频有20帧
- 需要生成: 500 - 20 = 480帧
- Segment数量: (480 + 80 - 1) // 80 = 6个segments
- Segment 1-5: 各生成80帧
- Segment 6: 生成80帧，裁剪到80帧（正好）

### 场景2: 生成1000帧的超长视频

```bash
torchrun --nproc_per_node=1 run_demo_long_video_continuation.py \
    --input_video input.mp4 \
    --output_path output_1000frames.mp4 \
    --prompt "A long journey through different landscapes..." \
    --checkpoint_dir ./checkpoints \
    --total_target_frames 1000 \
    --num_cond_frames 13 \
    --frames_per_segment 80 \
    --save_intermediate
```

**计算过程**:
- 假设输入视频有20帧
- 需要生成: 1000 - 20 = 980帧
- Segment数量: (980 + 80 - 1) // 80 = 13个segments
- Segment 1-12: 各生成80帧（共960帧）
- Segment 13: 生成80帧，裁剪到20帧（980 - 960 = 20）

### 场景3: 高分辨率生成（720p）

```bash
torchrun --nproc_per_node=1 run_demo_long_video_continuation.py \
    --input_video input.mp4 \
    --output_path output_720p.mp4 \
    --prompt "High quality video..." \
    --checkpoint_dir ./checkpoints \
    --total_target_frames 500 \
    --resolution 720p \
    --num_inference_steps 50
```

## 输出说明

### 控制台输出

脚本会输出每个segment的生成进度：

```
Input video: 20 frames
Target total frames: 500
Frames to generate: 480
Number of segments: 6
Frames per segment (new): 80
Condition frames: 13
Total frames per segment: 93

============================================================
Generating segment 1/6...
============================================================
Condition frames: 13 frames
Will generate: 93 frames (including 13 condition frames)
Will keep: 80 new frames
Segment 1 complete: generated 80 new frames
Total frames so far: 100
...
```

### 文件输出

- **最终视频**: `output_path` 指定的路径
- **中间视频** (如果启用 `--save_intermediate`): `output_path` 替换为 `*_segment_N.mp4`

## 注意事项

1. **内存管理**: 
   - 如果GPU显存不足，使用 `--offload_kv_cache` 选项
   - 每个segment生成后会清理显存

2. **帧数约束**:
   - 每个segment生成93帧（13条件 + 80新帧）
   - 确保 `num_frames = num_cond_frames + frames_per_segment` 满足VAE约束
   - VAE要求: `(num_frames - 1) % 4 == 0`
   - 93帧满足: (93-1) % 4 = 0 ✓

3. **连续性**:
   - 每个segment使用前一个segment的最后13帧作为条件
   - 这保证了视频的时序连续性

4. **总帧数**:
   - 如果生成的总帧数超过目标帧数，会自动裁剪
   - 如果不足，会生成到最接近的segment边界

## 与原始脚本的区别

| 特性 | `run_demo_video_continuation.py` | `run_demo_long_video_continuation.py` |
|------|--------------------------------|--------------------------------------|
| **生成方式** | 单次生成93帧 | 循环生成多个segments |
| **总帧数** | 固定93帧 | 可配置（通过参数） |
| **条件帧** | 输入视频的最后13帧 | 前一个segment的最后13帧 |
| **用途** | 短视频续写 | 超长视频生成 |

## 故障排除

### 问题1: 显存不足

**解决方案**:
```bash
--offload_kv_cache  # 将KV cache移到CPU
```

### 问题2: 生成速度慢

**解决方案**:
```bash
--enable_compile  # 启用torch.compile优化
--num_inference_steps 16  # 减少去噪步数（可能影响质量）
```

### 问题3: 视频不连续

**原因**: 条件帧数量不足或prompt变化太大

**解决方案**:
- 增加 `--num_cond_frames`（但需要相应调整 `frames_per_segment`）
- 使用更一致的prompt

## 性能参考

假设生成500帧视频（6个segments）：

- **单segment时间**: ~30-60秒（取决于GPU和参数）
- **总时间**: ~3-6分钟
- **GPU显存**: ~20-40GB（取决于分辨率）
- **输出文件大小**: ~50-200MB（取决于分辨率和内容）

## 代码结构

```
run_demo_long_video_continuation.py
├── generate(args)
│   ├── 模型加载和初始化
│   ├── 输入视频预处理
│   ├── 计算segment数量
│   └── 循环生成segments
│       ├── 提取条件帧
│       ├── 调用 generate_vc()
│       ├── 提取新帧
│       └── 更新条件帧
└── _parse_args()
    └── 参数解析
```

## 示例输出

```
Input video: 20 frames
Target total frames: 500
Frames to generate: 480
Number of segments: 6
Frames per segment (new): 80
Condition frames: 13
Total frames per segment: 93

============================================================
Generating segment 1/6...
============================================================
Condition frames: 13 frames
Will generate: 93 frames (including 13 condition frames)
Will keep: 80 new frames
Segment 1 complete: generated 80 new frames
Total frames so far: 100

============================================================
Generating segment 2/6...
============================================================
...
============================================================
Generation complete!
Total frames: 500
Target frames: 500
============================================================

Final video saved: output_long.mp4
```
