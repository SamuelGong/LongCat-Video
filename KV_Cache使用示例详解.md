# KV Cache在Video Continuation中的使用示例

本文档通过一个具体例子详细说明KV Cache在视频续写任务中的工作原理和使用流程。

---

## 一、示例设置

### 1.1 任务参数

假设我们要续写一个视频：
- **输入视频**: 已有视频的最后13帧作为条件帧
- **生成目标**: 93帧（包含13帧条件帧 + 80帧新帧）
- **潜在空间**: 4帧条件帧 + 20帧新帧 = 24帧
- **去噪步数**: 50步
- **DIT模型**: 48层Transformer，每层都有Self-Attention

### 1.2 数据维度

```
输入视频帧: [1, 3, 13, 480, 832]  (13帧条件帧)
  ↓ VAE编码
条件潜在帧: [1, 16, 4, 60, 104]   (4帧条件帧)
  ↓ 拼接随机噪声
完整潜在空间: [1, 16, 24, 60, 104] (4条件帧 + 20新帧)
  ↓ Patch Embedding
条件帧tokens: [1, 1560, 4096]      (4帧 × 30×52 patches × 4 = 1560 tokens)
新帧tokens:   [1, 31200, 4096]     (20帧 × 30×52 patches = 31200 tokens)
```

**注意**: 
- 每帧在潜在空间是 60×104
- Patch size = (1, 2, 2)，所以每帧有 30×52 = 1560个patches
- 条件帧: 4帧 × 1560 = 6240个tokens
- 新帧: 20帧 × 1560 = 31200个tokens

---

## 二、KV Cache初始化阶段

### 2.1 提取条件帧

**位置**: `pipeline_longcat_video.py:1022-1026`

```python
# 从完整潜在空间中提取条件帧
cond_latents = latents[:, :, :num_cond_latents]  
# Shape: [1, 16, 4, 60, 104]  (4帧条件帧)

# 分离新帧部分（后续去噪循环只处理这部分）
latents = latents[:, :, num_cond_latents:]
# Shape: [1, 16, 20, 60, 104]  (20帧新帧)
```

### 2.2 预计算条件帧的KV

**位置**: `pipeline_longcat_video.py:330-342`

```python
def _cache_clean_latents(self, cond_latents, model_max_length, offload_kv_cache, device, dtype):
    # 1. 设置条件帧的时间步为0（表示完全干净，不参与去噪）
    timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2])
    # Shape: [1, 4]  (4帧条件帧，时间步都是0)
    
    # 2. 创建空的文本embedding（因为skip_crs_attn=True，不会使用）
    empty_embeds = torch.zeros([1, 1, 512, 4096], device=device, dtype=dtype)
    
    # 3. 前向传播，只计算条件帧的KV
    _, kv_cache_dict = self.dit(
        hidden_states=cond_latents,      # [1, 16, 4, 60, 104]
        timestep=timestep,               # [1, 4] (全0)
        encoder_hidden_states=empty_embeds,  # 空embedding
        return_kv=True,                  # ← 关键：返回KV cache
        skip_crs_attn=True,             # 跳过cross-attention
        offload_kv_cache=offload_kv_cache
    )
    
    # 4. 保存KV cache
    self._update_kv_cache_dict(kv_cache_dict)
```

### 2.3 KV Cache的结构

**kv_cache_dict的结构**:
```python
kv_cache_dict = {
    0: (k_cache_0, v_cache_0),  # 第1层Transformer的KV
    1: (k_cache_1, v_cache_1),  # 第2层Transformer的KV
    ...
    47: (k_cache_47, v_cache_47) # 第48层Transformer的KV
}
```

**每层KV的形状**:
```python
# 对于条件帧（4帧）
k_cache_i.shape = [1, num_heads, N_cond_tokens, head_dim]
                = [1, 32, 6240, 128]  # 32头，6240个tokens，128维/头
v_cache_i.shape = [1, 32, 6240, 128]
```

**总KV cache大小**:
- 每层: 2 × 32 × 6240 × 128 × 4 bytes (float32) ≈ 204 MB
- 48层: 204 MB × 48 ≈ 9.8 GB

---

## 三、去噪循环中的KV Cache使用

### 3.1 单次去噪步骤（第1步，t=1.0）

**位置**: `pipeline_longcat_video.py:1030-1052`

#### Step 1: 准备输入

```python
# 只处理新帧部分（20帧）
latent_model_input = latents  # [1, 16, 20, 60, 104]
latent_model_input = torch.cat([latents] * 2)  # CFG: [2, 16, 20, 60, 104]

# 时间步（只对新帧）
timestep = t.expand(2).unsqueeze(-1).repeat(1, 20)  # [2, 20] (t=1.0)
```

#### Step 2: DIT前向传播（使用KV Cache）

**位置**: `longcat_video_dit.py:336-346`, `attention.py:149-181`

```python
noise_pred = self.dit(
    hidden_states=latent_model_input,  # [2, 16, 20, 60, 104] (只有新帧)
    timestep=timestep,                 # [2, 20]
    encoder_hidden_states=prompt_embeds,  # [2, 512, 4096]
    encoder_attention_mask=prompt_attention_mask,
    num_cond_latents=4,               # ← 关键：告诉模型有4帧条件帧
    kv_cache_dict=kv_cache_dict       # ← 关键：传入缓存的KV
)
```

#### Step 3: 每层Transformer Block的处理

**位置**: `longcat_video_dit.py:343-346`, `blocks.py:68-115`

对于每一层（例如第i层）：

```python
# 1. Patch Embedding（只处理新帧）
hidden_states = self.x_embedder(latent_model_input)
# [2, 16, 20, 60, 104] → [2, 31200, 4096]  (20帧的tokens)

# 2. Self-Attention（使用KV Cache）
block_outputs = block(
    hidden_states,                    # [2, 31200, 4096] (新帧tokens)
    encoder_hidden_states,           # 文本条件
    t,                               # 时间步
    y_seqlens,                       # 文本长度
    (20, 30, 52),                    # 新帧的shape (T, H, W)
    num_cond_latents=4,              # ← 关键：4帧条件帧
    kv_cache_dict={i: kv_cache_dict[i]}  # ← 关键：传入该层的KV cache
)
```

#### Step 4: Attention中的KV Cache使用

**位置**: `attention.py:149-181`

```python
def forward_with_kv_cache(self, x, shape, num_cond_latents, kv_cache):
    # x: [2, 31200, 4096] (新帧的tokens)
    # kv_cache: (k_cache, v_cache) from kv_cache_dict[i]
    #   k_cache: [1, 32, 6240, 128] (条件帧的K)
    #   v_cache: [1, 32, 6240, 128] (条件帧的V)
    
    # 1. 计算新帧的QKV
    qkv = self.qkv(x)  # [2, 31200, 4096*3]
    q, k, v = ...      # 分离Q, K, V
    # q: [2, 32, 31200, 128] (新帧的Q)
    # k: [2, 32, 31200, 128] (新帧的K)
    # v: [2, 32, 31200, 128] (新帧的V)
    
    # 2. 拼接缓存的KV和新帧的KV
    k_cache, v_cache = kv_cache  # 条件帧的KV（已缓存）
    k_full = torch.cat([k_cache, k], dim=2)  # [2, 32, 6240+31200, 128]
    v_full = torch.cat([v_cache, v], dim=2)  # [2, 32, 6240+31200, 128]
    # k_full: [2, 32, 37440, 128] (条件帧6240 + 新帧31200)
    
    # 3. 处理新帧的Q（需要padding以匹配完整序列长度）
    q_padding = torch.cat([torch.empty_like(k_cache), q], dim=2)
    # q_padding: [2, 32, 37440, 128] (条件帧部分为空，新帧部分有Q)
    
    # 4. 应用RoPE位置编码
    q_padding, k_full = self.rope_3d(q_padding, k_full, (24, 30, 52))
    # 完整shape: (4条件帧 + 20新帧 = 24帧)
    
    # 5. 提取新帧的Q（去掉padding部分）
    q = q_padding[:, :, -31200:]  # [2, 32, 31200, 128]
    
    # 6. Attention计算
    # Q: [2, 32, 31200, 128] (新帧)
    # K: [2, 32, 37440, 128] (条件帧 + 新帧)
    # V: [2, 32, 37440, 128] (条件帧 + 新帧)
    x = self._process_attn(q, k_full, v_full, shape)
    # 输出: [2, 32, 31200, 128] (新帧的attention输出)
    
    return x
```

**关键点**:
- ✅ **条件帧的KV**: 从cache中直接使用，**不需要重新计算**
- ✅ **新帧的KV**: 每次重新计算（因为新帧在去噪过程中会变化）
- ✅ **Attention**: 新帧的Q可以attend到所有帧（条件帧+新帧）

---

### 3.2 对比：不使用KV Cache的情况

**如果不使用KV Cache**，每次去噪步骤都需要：

```python
# 每次都需要处理完整24帧
latent_model_input = full_latents  # [2, 16, 24, 60, 104] (条件帧+新帧)

# 每次都需要计算所有24帧的KV
hidden_states = self.x_embedder(latent_model_input)
# [2, 37440, 4096]  (24帧的tokens)

# 每次都需要计算条件帧的KV（重复计算！）
qkv = self.qkv(hidden_states)
q, k, v = ...  # [2, 32, 37440, 128]
# 条件帧的KV: [2, 32, 6240, 128]  ← 每次都要重新计算！
```

**计算量对比**:
- **使用KV Cache**: 每次只计算新帧的KV (31200 tokens)
- **不使用KV Cache**: 每次计算所有帧的KV (37440 tokens)
- **节省**: 6240 / 37440 = 16.7% 的KV计算量

---

## 四、完整去噪循环示例

### 4.1 50步去噪循环

```python
# 初始化
cond_latents = latents[:, :, :4]      # [1, 16, 4, 60, 104]
latents = latents[:, :, 4:]            # [1, 16, 20, 60, 104]
kv_cache_dict = _cache_clean_latents(cond_latents)  # 预计算条件帧KV

# 50步去噪循环
for i, t in enumerate(timesteps):  # t从1.0到0.001
    # 步骤1: 准备输入（只有新帧）
    latent_model_input = latents  # [1, 16, 20, 60, 104]
    
    # 步骤2: DIT前向传播（使用KV Cache）
    noise_pred = self.dit(
        hidden_states=latent_model_input,
        timestep=t,
        num_cond_latents=4,
        kv_cache_dict=kv_cache_dict  # ← 复用条件帧的KV
    )
    
    # 步骤3: Scheduler更新（只更新新帧）
    latents = self.scheduler.step(noise_pred, t, latents)
    # latents: [1, 16, 20, 60, 104] (新帧逐步去噪)
    
    # 条件帧的KV cache在整个循环中保持不变！
```

### 4.2 关键观察

1. **条件帧KV**: 在整个50步循环中**只计算一次**（初始化时）
2. **新帧KV**: 每步都重新计算（因为新帧在变化）
3. **Attention**: 新帧的Q可以attend到条件帧（通过缓存的KV）

---

## 五、计算量分析

### 5.1 单次Attention计算量

**Attention复杂度**: O(N²)，其中N是序列长度

**使用KV Cache**:
- 条件帧KV: 计算1次（初始化）
  - QKV计算: O(6240 × 4096) = 25.5M ops
  - Attention: O(6240²) = 38.9M ops
- 新帧KV: 每步计算（50步）
  - QKV计算: O(31200 × 4096) = 127.8M ops/步
  - Attention: O(31200 × 37440) = 1.17B ops/步（新帧Q × 所有帧KV）

**不使用KV Cache**:
- 所有帧KV: 每步计算（50步）
  - QKV计算: O(37440 × 4096) = 153.4M ops/步
  - Attention: O(37440²) = 1.40B ops/步

### 5.2 总计算量对比

**使用KV Cache**:
```
初始化: 25.5M + 38.9M = 64.4M ops
50步: 50 × (127.8M + 1.17B) = 64.9B ops
总计: 64.4M + 64.9B ≈ 64.9B ops
```

**不使用KV Cache**:
```
50步: 50 × (153.4M + 1.40B) = 77.7B ops
```

**节省**: (77.7B - 64.9B) / 77.7B = **16.5%**

### 5.3 48层Transformer

**总节省**:
- 单层节省: 16.5%
- 48层总节省: 约16.5%（每层都节省）
- **实际加速**: 约1.2-1.3倍（考虑内存访问等其他因素）

---

## 六、内存使用

### 6.1 KV Cache内存

```python
# 每层的KV cache
k_cache.shape = [1, 32, 6240, 128]  # float32
v_cache.shape = [1, 32, 6240, 128]  # float32

# 单层内存
单层KV = 2 × 32 × 6240 × 128 × 4 bytes = 204 MB

# 48层总内存
总KV Cache = 204 MB × 48 = 9.8 GB
```

### 6.2 内存优化选项

**offload_kv_cache=True**:
```python
# 将KV cache offload到CPU
kv_cache_dict[i] = (k_cache.cpu(), v_cache.cpu())
```

**优点**: 节省GPU显存
**缺点**: 每次使用需要transfer回GPU，增加延迟

---

## 七、总结

### 7.1 KV Cache的工作流程

1. **初始化**: 预计算条件帧的KV，缓存48层的KV
2. **去噪循环**: 每步只计算新帧的KV，复用条件帧的KV
3. **Attention**: 新帧的Q可以attend到所有帧（条件帧+新帧）

### 7.2 关键优势

1. **减少计算**: 条件帧的KV只计算一次，节省16.5%的计算量
2. **加速推理**: 实际加速约1.2-1.3倍
3. **保持功能**: 新帧仍然可以attend到条件帧，保证连续性

### 7.3 适用场景

- ✅ **Video Continuation**: 条件帧固定，非常适合KV Cache
- ✅ **Image-to-Video**: 条件帧（第一帧）固定
- ❌ **Text-to-Video**: 没有条件帧，KV Cache无意义

---

## 八、代码位置总结

1. **KV Cache初始化**: `pipeline_longcat_video.py:330-342` (`_cache_clean_latents`)
2. **KV Cache使用**: `pipeline_longcat_video.py:1022-1026` (初始化), `1051` (传入DIT)
3. **Attention中使用**: `attention.py:149-181` (`forward_with_kv_cache`)
4. **DIT Block中使用**: `longcat_video_dit.py:336-346` (每层传入kv_cache_dict)

---

这个例子展示了KV Cache如何通过缓存条件帧的KV，在50步去噪循环中避免重复计算，从而显著加速视频续写任务的推理过程。
