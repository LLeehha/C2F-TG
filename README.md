# C2F-TG

## Architecture Overview

### 1. `TStarSearcher` (Base Class)

The base class implements a **coarse-to-fine temporal search strategy** using dynamic sampling:

**Key Components:**
- **Score Distribution**: Maintains relevance scores across video timeline
- **Adaptive Sampling**: Samples frames based on probability distribution
- **Spline Interpolation**: Smooths score distribution for better coverage
- **Peak Suppression**: Prevents selecting redundant nearby frames

**Main Methods:**
- `imageGridScoreFunction()`: Score frames using detection/VLM
- `sample_frames()`: Adaptively sample frames based on current distribution
- `pop_frames()`: Extract top-k frames after search completes
- `update_frame_distribution()`: Update scores based on detection results

### 2. `QwenTStarSearcher` (VLM Implementation)

Extends `TStarSearcher` to use **Vision-Language Models** instead of object detection:

**Key Features:**
- **Natural Language Queries**: Uses text descriptions instead of object labels
- **Grid-based Scoring**: VLM scores uniformly sampled frame grids
- **Rank-based Decay**: Combines VLM confidence with ranking decay
- **Temporal Smoothing**: Applies Gaussian smoothing for continuity
- **Spatial Diffusion**: Spreads scores to neighboring cells

**VLM Prompt Design:**
```
Query: [natural language description]
Grid: 4×4 frames with red indices
Output: JSON with 4 candidates (frame_index, confidence)
```

## Usage Example

### Using VLM-based Search

```python
from qwen_searcher import QwenTStarSearcher
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load VLM
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Initialize searcher
searcher = QwenTStarSearcher(
    qwen_model=model,
    processor=processor,
    device="cuda",
    video_path="video.mp4",
    target_objects=["person opening a door"],
    search_nframes=150,
    image_grid_shape=(4, 4),
    search_budget=T
)

# Run search iterations
for iteration in range(5):
    frame_secs, frames = searcher.sample_frames(num_samples=16)
    grid_img = searcher.create_image_grid(frames, rows=4, cols=4)
    conf_maps, det_objs = searcher.score_image_grids([grid_img], (4, 4))
    searcher.update_frame_distribution(frame_secs, conf_maps, det_objs)

# Extract top anchors
_, time_stamps = searcher.pop_frames(num_samples=8)
print(f"Selected anchors: {time_stamps}")


## Algorithm Details

### Phase 1: Coarse Search (Anchor Localization)

1. **Uniform Sampling**: Initially sample frames uniformly
2. **Grid Scoring**: VLM scores 4×4 frame grids
3. **Score Update**: Update distribution based on VLM confidence
4. **Adaptive Sampling**: Next iteration samples high-probability regions
5. **Iteration**: Repeat 3-5 times for refinement

### Phase 2: Anchor Extraction

1. **Temporal Smoothing**: Apply Gaussian filter (σ=1.5) to reduce noise
2. **Peak Suppression**: Select top-k peaks with spatial separation
3. **Output**: Return timestamps of selected anchors

### Scoring Strategy

**Rank-based Decay:**
```python
rank_decay = [1.0, 0.75, 0.5, 0.3]  # For ranks 1-4
score = confidence × rank_decay[rank]
```

**Spatial Diffusion:**
```python
# Neighboring cells get 40% of center score
neighbor_score = center_score × 0.4
```

**Temporal Smoothing:**
```python
# Gaussian filter with σ=1.5
smoothed_scores = gaussian_filter1d(scores, sigma=1.5)
```

### Complete Pipeline Example

```python
from qwen_searcher import QwenTStarSearcher
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load VLM
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# ==================== Phase 1: Anchor Localization ====================
searcher = QwenTStarSearcher(
    video_path=video_path,
    target_objects=["person opening a door"],
    qwen_model=model,
    processor=processor,
    device="cuda",
    search_nframes=min(150, int(duration / 3)),
    image_grid_shape=(4, 4),
    search_budget=int(duration * 1.0),
)

# Run iterative search
_, time_stamps = searcher.search()
print(f"[Phase 1] Anchors found at: {time_stamps}")

# ==================== Phase 2: Adaptive Window Generation ====================
time_stamps = sorted(time_stamps)

# Clustering parameters (duration-adaptive)
tau_safe = min(max(duration * 0.05, 3.0), 20.0)  # Clustering threshold
base_radius = min(max(duration * 0.025, 3.0), 25.0)  # Base expansion radius

# Cluster nearby anchors
current_cluster = [time_stamps[0]]
clusters = []

for t in time_stamps[1:]:
    if t - current_cluster[-1] <= tau_safe:
        current_cluster.append(t)
    else:
        clusters.append(current_cluster)
        current_cluster = [t]
clusters.append(current_cluster)

# Generate windows from clusters
windows = []
for cluster in clusters:
    cluster_start = cluster[0]
    cluster_end = cluster[-1]
    cluster_span = cluster_end - cluster_start
    
    # Dynamic radius: base + proportional to cluster density
    dynamic_radius = base_radius + (cluster_span * 0.5)
    
    w_start = max(0.0, cluster_start - dynamic_radius)
    w_end = min(duration, cluster_end + dynamic_radius)
    windows.append([w_start, w_end])

# Merge overlapping windows
merged = []
for ws, we in sorted(windows, key=lambda x: (x[0], x[1])):
    if not merged:
        merged.append([ws, we])
        continue
    
    ps, pe = merged[-1]
    merged.append([ws, we])

windows = merged
print(f"[Phase 2] Generated {len(windows)} windows: {windows}")
