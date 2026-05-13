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

