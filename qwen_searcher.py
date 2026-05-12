"""
QwenTStarSearcher: VLM-based temporal video search using Qwen2-VL.

This module extends TStarSearcher to use Vision-Language Models (VLMs)
for temporal grounding instead of traditional object detection.
"""

import numpy as np
import torch
import gc
import re
import json
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

from videomind.TStar.interface_searcher import TStarSearcher
from videomind.dataset.utils import smart_resize


class DummyHeuristic:
    """Placeholder heuristic for VLM-based search (no object detection needed)."""

    def __init__(self):
        self.texts = []
        self.detections_inbatch = []

    def reparameterize_object_list(self, target_objects, cue_objects):
        pass

    def bbox_visualization(self, images, detections_inbatch):
        return images


class QwenTStarSearcher(TStarSearcher):
    """
    VLM-based temporal searcher using Qwen2-VL for video grounding.

    Instead of object detection, this class uses a Vision-Language Model
    to score frame grids based on natural language queries.
    """

    def __init__(self, qwen_model, processor, device, **kwargs):
        """
        Initialize QwenTStarSearcher with VLM components.

        Args:
            qwen_model: Qwen2-VL model instance
            processor: Qwen2-VL processor for input preparation
            device: Device for model inference (cuda/cpu)
            **kwargs: Additional arguments passed to TStarSearcher
        """
        kwargs['heuristic'] = DummyHeuristic()
        super().__init__(**kwargs)

        self.qwen_model = qwen_model
        self.processor = processor
        self.device = device

        # Override search budget from kwargs
        real_budget = kwargs.get('search_budget', 128)
        self.search_budget = real_budget

        # VLM prompt for temporal grounding
        self.search_prompt = (
            "You are a visual temporal search agent for video grounding. "
            "The image is a grid of uniformly sampled video frames. "
            "Each cell is labeled with a red index from 0 to __MAX_IDX__. "
            "Query: '__TARGETS__'. "
            "Return exactly 4 relevant cell indices that best help localize the event timing. "
            "If the query is about an action or state change, prefer cells where it is happening. "
            "If the query is about a static object or scene, prefer cells most informative for answering the query. "
            "Return ONLY one compact single-line JSON object with key \"candidates\". "
            "\"candidates\" must contain exactly 4 objects. "
            "Each object must contain \"frame_index\" (integer from 0 to __MAX_IDX__) and "
            "\"confidence\" (float from 0.0 to 1.0). "
            "Do not use markdown fences. "
            "Do not output more than 4 candidates. "
            "Frame indices do not need to be consecutive."
        )

    @torch.inference_mode()
    def imageGridScoreFunction(self, images, output_dir, image_grids):
        """
        Score image grids using VLM instead of object detection.

        Args:
            images: List of grid images
            output_dir: Output directory (unused)
            image_grids: Grid dimensions (rows, cols)

        Returns:
            Tuple of (confidence_maps, detected_objects_maps)
        """
        grid_rows, grid_cols = image_grids
        expected_cells = grid_rows * grid_cols

        confidence_maps = []
        detected_objects_maps = []

        for img_array in images:
            if img_array is None or img_array.size == 0:
                print("[T* Warning] Empty image array received.")
                confidence_maps.append(np.zeros((grid_rows, grid_cols), dtype=np.float32))
                detected_objects_maps.append([[] for _ in range(expected_cells)])
                continue

            # Prepare grid image with cell indices
            grid_img = Image.fromarray(img_array).convert('RGB')
            new_h, new_w = smart_resize(grid_img.height, grid_img.width)
            grid_img = grid_img.resize((new_w, new_h))

            draw = ImageDraw.Draw(grid_img)
            try:
                font = ImageFont.truetype("arialbd.ttf", 480)
            except IOError:
                font = ImageFont.load_default()

            # Draw cell indices on grid
            w = grid_img.width // grid_cols
            h = grid_img.height // grid_rows
            for i in range(expected_cells):
                r, c = divmod(i, grid_cols)
                x, y = c * w, r * h
                draw.text(
                    (x + 15, y + 15),
                    str(i),
                    fill=(255, 0, 0),
                    font=font,
                    stroke_width=4,
                    stroke_fill=(255, 255, 255)
                )

            # Prepare VLM input
            targets_str = ", ".join(self.target_objects)
            prompt_text = (
                self.search_prompt
                .replace("__MAX_IDX__", str(expected_cells - 1))
                .replace("__TARGETS__", targets_str)
            )

            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': grid_img},
                    {'type': 'text', 'text': prompt_text}
                ]
            }]

            # Run VLM inference
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[grid_img], return_tensors='pt').to(self.device)

            output_ids = self.qwen_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256
            )
            output_ids = output_ids[0, inputs.input_ids.size(1):]
            response = self.processor.decode(output_ids, skip_special_tokens=True)

            # Clean up GPU memory
            del inputs, output_ids
            torch.cuda.empty_cache()

            # Parse VLM response
            conf_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            det_obj_map = [[] for _ in range(expected_cells)]

            try:
                parsed = self._parse_vlm_response(response, expected_cells)

                if len(parsed) == 0:
                    raise ValueError(f"No valid candidates parsed from response: {response}")

                # Apply rank-based scoring with decay
                used = set()
                rank_decay = [1.0, 0.75, 0.5, 0.3]

                for rank, (idx, conf) in enumerate(parsed):
                    if idx in used:
                        continue
                    used.add(idx)

                    # Combine confidence and rank decay
                    score = conf * rank_decay[min(rank, len(rank_decay) - 1)]

                    r, c = divmod(idx, grid_cols)
                    conf_map[r, c] = max(conf_map[r, c], score)
                    det_obj_map[idx].extend(self.target_objects)

                    # Spatial diffusion to neighboring cells
                    for nr in range(max(0, r - 1), min(grid_rows, r + 2)):
                        for nc in range(max(0, c - 1), min(grid_cols, c + 2)):
                            if nr == r and nc == c:
                                continue
                            conf_map[nr, nc] = max(conf_map[nr, nc], score * 0.4)

            except Exception as e:
                print(f"[T* Error] Parsing failed: {response}. Error: {e}")

            confidence_maps.append(conf_map)
            detected_objects_maps.append(det_obj_map)

            del grid_img
            gc.collect()

        return np.stack(confidence_maps), detected_objects_maps

    def _parse_vlm_response(self, response: str, expected_cells: int) -> list:
        """
        Parse VLM JSON response with multiple fallback strategies.

        Args:
            response: Raw VLM output string
            expected_cells: Maximum valid cell index

        Returns:
            List of (frame_index, confidence) tuples
        """
        cleaned = response.strip().replace("```json", "").replace("```", "").strip()
        parsed = []

        # Strategy 1: Parse full JSON structure
        try:
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match is not None:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                candidates = result.get("candidates", [])
                if isinstance(candidates, list):
                    for cand in candidates[:4]:
                        idx = int(cand.get("frame_index", 0))
                        conf = float(cand.get("confidence", 0.0))
                        idx = max(0, min(expected_cells - 1, idx))
                        conf = max(0.0, min(1.0, conf))
                        parsed.append((idx, conf))
        except Exception:
            parsed = []

        # Strategy 2: Regex extraction of frame_index and confidence pairs
        if len(parsed) == 0:
            pair_pattern = re.findall(
                r'"frame_index"\s*:\s*(\d+)\s*,\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
                cleaned
            )
            for idx_str, conf_str in pair_pattern[:4]:
                idx = int(idx_str)
                conf = float(conf_str)
                idx = max(0, min(expected_cells - 1, idx))
                conf = max(0.0, min(1.0, conf))
                parsed.append((idx, conf))

        # Strategy 3: Extract single frame_index and confidence
        if len(parsed) == 0:
            single_idx = re.search(r'"frame_index"\s*:\s*(\d+)', cleaned)
            single_conf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
            if single_idx and single_conf:
                idx = int(single_idx.group(1))
                conf = float(single_conf.group(1))
                idx = max(0, min(expected_cells - 1, idx))
                conf = max(0.0, min(1.0, conf))
                parsed.append((idx, conf))

        return parsed

    def verify_and_remove_target(self, frame_sec, detected_objects, confidence_threshold):
        """
        Disable target verification for VLM-based search.

        VLM operates on natural language queries rather than discrete object detection,
        so traditional verification is not applicable.
        """
        return False

    def pop_frames(self, video_path=None, num_samples=8):
        """
        Extract top frames using peak suppression with temporal smoothing.

        Args:
            video_path: Path to video file (unused, kept for interface compatibility)
            num_samples: Number of frames to extract

        Returns:
            Tuple of (empty_list, time_stamps)
        """
        scores = np.array(self.score_distribution, dtype=np.float32)

        if np.sum(scores) <= 0:
            print("\n[T* Info] No relevant frames found. Returning empty.")
            return [], []

        # Store original scores for debugging
        original_scores = scores.copy()

        # Apply temporal smoothing to enforce continuity
        scores = gaussian_filter1d(scores, sigma=1.5)

        # Debug output
        diff = np.abs(scores - original_scores).mean()
        print(f"\n[T* Debug] Gaussian smoothing applied. Mean absolute difference: {diff:.4f}")

        # Update score distribution for visualization
        self.score_distribution = scores.tolist()

        # Peak suppression: select top peaks with spatial separation
        picked = []
        work_scores = scores.copy()
        suppress_radius = 1

        while len(picked) < min(num_samples, len(work_scores)):
            idx = int(np.argmax(work_scores))
            if work_scores[idx] <= 0:
                break

            picked.append(idx)

            # Suppress neighboring peaks
            l = max(0, idx - suppress_radius)
            r = min(len(work_scores), idx + suppress_radius + 1)
            work_scores[l:r] = 0.0

        picked = sorted(picked)

        # Convert to timestamps
        fps = getattr(self, 'fps', 1.0)
        time_stamps = [float(idx) / fps for idx in picked]

        print(f"\n[T* Info] Peak-suppressed extraction selected timestamps: {time_stamps}")
        return [], time_stamps

    def create_image_grid(self, frames, grid_rows, grid_cols):
        """
        Create image grid with automatic padding/truncation for mismatched frame counts.

        Args:
            frames: List of frame images
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns

        Returns:
            Combined image grid
        """
        expected_count = grid_rows * grid_cols
        actual_count = len(frames)

        if actual_count == expected_count:
            return super().create_image_grid(frames, grid_rows, grid_cols)

        # Handle edge cases
        if actual_count == 0:
            print(f"\n[T* Warning] Extracted 0 frames! Generating black dummy grid.")
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frames = [dummy_frame for _ in range(expected_count)]

        elif actual_count < expected_count:
            padding_needed = expected_count - actual_count
            print(f"\n[T* Warning] Only got {actual_count} frames, padding {padding_needed} black frames.")
            dummy_frame = np.zeros_like(frames[0])
            frames.extend([dummy_frame] * padding_needed)

        elif actual_count > expected_count:
            print(f"\n[T* Warning] Got {actual_count} frames, truncating to {expected_count}.")
            frames = frames[:expected_count]

        return super().create_image_grid(frames, grid_rows, grid_cols)
