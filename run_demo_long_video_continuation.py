import os
import argparse

import cv2
import datetime
import PIL.Image
import numpy as np

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from diffusers.utils import load_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return original_fps


def generate(args):
    # case setup
    video_path = args.input_video
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    num_cond_frames = args.num_cond_frames  # 固定上文窗口：13帧
    frames_per_segment = args.frames_per_segment  # 每个segment生成的新帧数：80帧
    total_target_frames = args.total_target_frames  # 目标总帧数（可配置）
    
    # 计算每个segment的总帧数（条件帧 + 新帧）
    num_frames = num_cond_frames + frames_per_segment  # 13 + 80 = 93帧
    
    # load parsed args
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile
    output_path = args.output_path

    # prepare distributed environment
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank    = dist.get_rank()
    num_processes  = dist.get_world_size()

    # initialize context parallel before loading models
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)

    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer = tokenizer,
        text_encoder = text_encoder,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
    )
    pipe.to(local_rank)

    global_seed = args.seed if args.seed is not None else 42
    seed = global_seed + global_rank

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    # Load and preprocess input video
    video = load_video(video_path)
    target_fps = args.target_fps
    target_size = video[0].size  # (width, height)
    current_fps = get_fps(video_path)
    stride = max(1, round(current_fps / target_fps))
    
    # Preprocess input video frames
    video = video[::stride]
    video = [(video[i] * 255).astype(np.uint8) for i in range(video.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]
    
    # Calculate number of segments needed
    # 第一块：输入视频的13帧 + 生成80帧 = 93帧，保留80新帧
    # 后续块：每块生成80新帧
    # 总帧数 = 输入视频帧数 + (num_segments - 1) * frames_per_segment + 最后一块的实际帧数
    input_video_frames = len(video)
    remaining_frames_to_generate = total_target_frames - input_video_frames
    
    if remaining_frames_to_generate <= 0:
        if local_rank == 0:
            print(f"Input video already has {input_video_frames} frames, which meets or exceeds target {total_target_frames} frames.")
        return
    
    # Calculate number of segments
    num_segments = (remaining_frames_to_generate + frames_per_segment - 1) // frames_per_segment  # 向上取整
    
    if local_rank == 0:
        print(f"Input video: {input_video_frames} frames")
        print(f"Target total frames: {total_target_frames}")
        print(f"Frames to generate: {remaining_frames_to_generate}")
        print(f"Number of segments: {num_segments}")
        print(f"Frames per segment (new): {frames_per_segment}")
        print(f"Condition frames: {num_cond_frames}")
        print(f"Total frames per segment: {num_frames}")

    # Initialize
    all_generated_frames = video.copy()  # 包含输入视频的所有帧
    current_video = video  # 当前用于条件的视频（第一块用输入视频）

    # Generate segments
    for segment_idx in range(num_segments):
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"Generating segment {segment_idx + 1}/{num_segments}...")
            print(f"{'='*60}")
        
        # Calculate how many frames to generate in this segment
        frames_generated_so_far = len(all_generated_frames) - input_video_frames
        remaining = remaining_frames_to_generate - frames_generated_so_far
        
        # 如果是最后一块且不足80帧，仍然生成80帧，后续会裁剪
        actual_frames_to_generate = min(frames_per_segment, remaining) if segment_idx == num_segments - 1 else frames_per_segment
        
        # 使用当前视频的最后num_cond_frames帧作为条件帧
        cond_video = current_video[-num_cond_frames:]
        
        if local_rank == 0:
            print(f"Condition frames: {len(cond_video)} frames")
            print(f"Will generate: {num_frames} frames (including {num_cond_frames} condition frames)")
            print(f"Will keep: {frames_per_segment} new frames")
        
        # Generate video continuation
        output = pipe.generate_vc(
            video=cond_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=args.resolution,  # '480p' or '720p'
            num_frames=num_frames,  # 93帧（13条件 + 80新帧）
            num_cond_frames=num_cond_frames,  # 13帧条件帧
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=args.offload_kv_cache,
            enhance_hf=args.enhance_hf,
        )[0]

        # Post-process output
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        new_video = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in new_video]
        del output
        torch_gc()

        # Extract only the new frames (skip condition frames)
        # generate_vc返回93帧，前13帧是条件帧（与输入重复），后80帧是新生成的
        new_frames_only = new_video[num_cond_frames:]
        
        # 如果是最后一块，裁剪到实际需要的帧数
        if segment_idx == num_segments - 1 and actual_frames_to_generate < frames_per_segment:
            new_frames_only = new_frames_only[:actual_frames_to_generate]
            if local_rank == 0:
                print(f"Last segment: cropping to {actual_frames_to_generate} frames")
        
        # Add new frames to the complete video
        all_generated_frames.extend(new_frames_only)
        
        # Update current_video for next segment
        # 使用新生成的视频的最后num_cond_frames帧作为下一块的条件
        current_video = new_video[-num_cond_frames:]
        
        if local_rank == 0:
            print(f"Segment {segment_idx + 1} complete: generated {len(new_frames_only)} new frames")
            print(f"Total frames so far: {len(all_generated_frames)}")
            
            # Save intermediate result
            if args.save_intermediate:
                output_tensor = torch.from_numpy(np.array(all_generated_frames))
                intermediate_path = output_path.replace('.mp4', f'_segment_{segment_idx+1}.mp4')
                write_video(intermediate_path, output_tensor, fps=target_fps, video_codec="libx264", options={"crf": f"{18}"})
                print(f"Saved intermediate video: {intermediate_path}")
                del output_tensor

    # Final output
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"Total frames: {len(all_generated_frames)}")
        print(f"Target frames: {total_target_frames}")
        print(f"{'='*60}\n")
        
        # Ensure we have exactly the target number of frames
        if len(all_generated_frames) > total_target_frames:
            all_generated_frames = all_generated_frames[:total_target_frames]
            if local_rank == 0:
                print(f"Trimmed to target length: {len(all_generated_frames)} frames")
        
        output_tensor = torch.from_numpy(np.array(all_generated_frames))
        write_video(output_path, output_tensor, fps=target_fps, video_codec="libx264", options={"crf": f"{18}"})
        print(f"Final video saved: {output_path}")
        del output_tensor


def _parse_args():
    parser = argparse.ArgumentParser(description="Long video continuation with fixed context window")
    
    # Input/Output
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_long_continuation.mp4",
        help="Path to save output video"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative prompt"
    )
    
    # Video generation parameters
    parser.add_argument(
        "--total_target_frames",
        type=int,
        required=True,
        help="Total target number of frames in the final video"
    )
    parser.add_argument(
        "--num_cond_frames",
        type=int,
        default=13,
        help="Number of condition frames (context window size). Default: 13"
    )
    parser.add_argument(
        "--frames_per_segment",
        type=int,
        default=80,
        help="Number of new frames to generate per segment. Default: 80"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        choices=["480p", "720p"],
        help="Video resolution. Default: 480p"
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=15,
        help="Target FPS for output video. Default: 15"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps. Default: 50"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale. Default: 4.0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Default: 42"
    )
    
    # Model parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size. Default: 1"
    )
    parser.add_argument(
        "--enable_compile",
        action="store_true",
        help="Enable torch.compile for model optimization"
    )
    
    # KV Cache options
    parser.add_argument(
        "--offload_kv_cache",
        action="store_true",
        help="Offload KV cache to CPU to save GPU memory"
    )
    parser.add_argument(
        "--enhance_hf",
        action="store_true",
        default=True,
        help="Enable high-frequency enhancement. Default: True"
    )
    
    # Other options
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate segment videos"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)


'''
torchrun --nproc_per_node=1 run_demo_long_video_continuation.py \
    --input_video assets/motorcycle.mp4 \
    --output_path output_long.mp4 \
    --prompt "A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides." \
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
'''
