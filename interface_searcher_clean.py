"""
TStarSearcher: Base class for temporal video search using dynamic sampling.
"""

import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline


class TStarSearcher:
    """
    Base class for performing keyframe search in videos using dynamic sampling.

    This class implements a coarse-to-fine search strategy that adaptively
    samples video frames based on detection confidence scores.
    """

    def __init__(
        self,
        video_path: str,
        heuristic: object,
        target_objects: List[str],
        cue_objects: List[str],
        search_nframes: int = 8,
        image_grid_shape: Tuple[int, int] = (4, 4),
        search_budget: float = 0.1,
        output_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
        object2weight: Optional[dict] = None,
    ):
        """
        Initialize TStarSearcher with video properties and configuration.

        Args:
            video_path: Path to the input video file
            heuristic: Detection interface (e.g., YOLO, VLM)
            target_objects: Primary objects to detect
            cue_objects: Contextual objects to aid detection
            search_nframes: Number of keyframes to search for
            image_grid_shape: Grid dimensions (rows, cols) for tiling
            search_budget: Fraction of frames to process (capped at 1000)
            output_dir: Directory for saving outputs
            confidence_threshold: Detection confidence threshold
            object2weight: Mapping of object names to detection weights
        """
        self.video_path = video_path
        self.target_objects = target_objects
        self.cue_objects = cue_objects
        self.search_nframes = search_nframes
        self.image_grid_shape = image_grid_shape
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}
        self.fps = 1  # Sampling rate: 1 frame per second

        # Initialize video properties
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        self.raw_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = total_frames / self.raw_fps
        cap.release()

        # Adjust total frame number based on sampling rate
        self.total_frame_num = int(self.duration * self.fps)
        self.remaining_targets = target_objects.copy()
        self.search_budget = min(1000, int(self.total_frame_num * search_budget))

        # Initialize score distributions and histories
        self.score_distribution = np.zeros(self.total_frame_num) + 1e-6
        self.non_visiting_frames = np.ones(self.total_frame_num)
        self.P = np.ones(self.total_frame_num) * self.confidence_threshold * 0.3

        self.P_history = []
        self.Score_history = []
        self.non_visiting_history = []
        self.image_grid_iters = []
        self.detect_annotot_iters = []
        self.detect_bbox_iters = []

        # Set detection interface
        self.heuristic = heuristic
        self.heuristic.reparameterize_object_list(target_objects, cue_objects)

        for obj in target_objects:
            self.object2weight[obj] = 1.0
        for obj in cue_objects:
            self.object2weight[obj] = 0.5

    # ==================== Detection Methods ====================

    def imageGridScoreFunction(
        self,
        images: List[np.ndarray],
        output_dir: Optional[str],
        image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        """
        Run object detection on image grids and map detections to grid cells.

        Args:
            images: List of grid images
            output_dir: Directory to save results (optional)
            image_grids: Grid dimensions (rows, cols)

        Returns:
            Tuple of (confidence_maps, detected_objects_maps)
            - confidence_maps: shape (num_images, grid_rows, grid_cols)
            - detected_objects_maps: List of detected objects per grid cell
        """
        if not images:
            return np.array([]), []

        grid_rows, grid_cols = image_grids
        grid_height = images[0].shape[0] / grid_rows
        grid_width = images[0].shape[1] / grid_cols

        confidence_maps = []
        detected_objects_maps = []

        for image in images:
            detections = self.heuristic.inference_detector(
                images=[image],
                use_amp=False
            )

            confidence_map = np.zeros((grid_rows, grid_cols))
            detected_objects_map = [[] for _ in range(grid_rows * grid_cols)]

            for detection in detections:
                for bbox, label, confidence in zip(
                    detection.xyxy, detection.class_id, detection.confidence
                ):
                    object_name = self.heuristic.texts[label][0]
                    weight = self.object2weight.get(object_name, 0.5)
                    adjusted_confidence = confidence * weight

                    x_min, y_min, x_max, y_max = bbox
                    box_center_x = (x_min + x_max) / 2
                    box_center_y = (y_min + y_max) / 2

                    grid_x = int(box_center_x // grid_width)
                    grid_y = int(box_center_y // grid_height)
                    grid_x = min(grid_x, grid_cols - 1)
                    grid_y = min(grid_y, grid_rows - 1)

                    cell_index = grid_y * grid_cols + grid_x
                    confidence_map[grid_y, grid_x] = max(
                        confidence_map[grid_y, grid_x], adjusted_confidence
                    )
                    detected_objects_map[cell_index].append(object_name)

            confidence_maps.append(confidence_map)
            detected_objects_maps.append(detected_objects_map)

        return np.stack(confidence_maps), detected_objects_maps

    # ==================== Frame Reading Methods ====================

    def read_frame_batch(
        self, video_path: str, frame_indices: List[int]
    ) -> Tuple[List[int], np.ndarray]:
        """
        Read a batch of frames from video with fallback for corrupted videos.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to read

        Returns:
            Tuple of (safe_indices, frames)
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        max_safe_idx = max(0, len(vr) - 1)
        safe_indices = [min(idx, max_safe_idx) for idx in frame_indices]

        try:
            frames = vr.get_batch(safe_indices).asnumpy()
            del vr
            return safe_indices, frames
        except Exception as e:
            print(f"\n[Warning] Decord batch read failed. Using frame-by-frame fallback...")
            frames = []

            try:
                fallback_frame = vr[0].asnumpy()
            except:
                fallback_frame = np.zeros((224, 224, 3), dtype=np.uint8)

            for idx in safe_indices:
                try:
                    frame = vr[idx].asnumpy()
                    frames.append(frame)
                    fallback_frame = frame
                except Exception:
                    frames.append(fallback_frame)

            del vr
            return safe_indices, np.stack(frames)

    def create_image_grid(
        self, frames: List[np.ndarray], rows: int, cols: int
    ) -> np.ndarray:
        """
        Combine frames into a single image grid.

        Args:
            frames: List of frame images
            rows: Number of grid rows
            cols: Number of grid columns

        Returns:
            Combined image grid
        """
        if len(frames) != rows * cols:
            raise ValueError(
                f"Frame count ({len(frames)}) does not match grid dimensions ({rows}x{cols})"
            )

        resized_frames = [cv2.resize(frame, (200, 95)) for frame in frames]
        grid_rows = [
            np.hstack(resized_frames[i * cols:(i + 1) * cols]) for i in range(rows)
        ]
        return np.vstack(grid_rows)

    # ==================== Score Update Methods ====================

    def score_image_grids(
        self, images: List[np.ndarray], image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        """
        Generate confidence maps and detected objects for image grids.

        Args:
            images: List of grid images
            image_grids: Grid dimensions (rows, cols)

        Returns:
            Tuple of (confidence_maps, detected_objects_maps)
        """
        return self.imageGridScoreFunction(images, self.output_dir, image_grids)

    def store_score_distribution(self):
        """Save current probability distribution and histories."""
        self.P_history.append(copy.deepcopy(self.P).tolist())
        self.Score_history.append(copy.deepcopy(self.score_distribution).tolist())
        self.non_visiting_history.append(copy.deepcopy(self.non_visiting_frames).tolist())

    def update_top_25_with_window(
        self,
        frame_confidences: List[float],
        sampled_frame_indices: List[int],
        window_size: int = 2
    ):
        """
        Update score distribution for high-confidence frames and their neighbors.

        Args:
            frame_confidences: Confidence scores for sampled frames
            sampled_frame_indices: Corresponding frame indices
            window_size: Number of neighboring frames to update
        """
        if len(frame_confidences) == 0:
            return

        top_threshold = np.percentile(frame_confidences, 85)
        top_indices = [
            frame_idx for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences)
            if confidence >= top_threshold and confidence > 0
        ]

        for frame_idx in top_indices:
            center_score = float(self.score_distribution[frame_idx])
            for offset in range(-window_size, window_size + 1):
                neighbor_idx = frame_idx + offset
                if 0 <= neighbor_idx < len(self.score_distribution):
                    decay = 1.0 if offset == 0 else 0.5
                    self.score_distribution[neighbor_idx] = max(
                        self.score_distribution[neighbor_idx],
                        center_score * decay
                    )

    def spline_keyframe_distribution(
        self,
        non_visiting_frames: np.ndarray,
        score_distribution: np.ndarray,
        video_length: int
    ) -> np.ndarray:
        """
        Generate probability distribution using spline interpolation.

        Args:
            non_visiting_frames: Array indicating unvisited frames
            score_distribution: Current score distribution
            video_length: Total number of frames

        Returns:
            Normalized probability distribution
        """
        visited_indices = np.array([
            idx for idx, visited in enumerate(non_visiting_frames) if visited == 0
        ])

        if len(visited_indices) == 0:
            return np.ones(video_length) / video_length

        observed_scores = np.array([score_distribution[idx] for idx in visited_indices])
        spline = UnivariateSpline(visited_indices, observed_scores, s=0.5)

        all_frames = np.arange(video_length)
        spline_scores = spline(all_frames)

        # Apply sigmoid for smoothing
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        adjusted_scores = np.maximum(1 / video_length, spline_scores)
        p_distribution = sigmoid(adjusted_scores)
        p_distribution /= p_distribution.sum()

        return p_distribution

    def update_frame_distribution(
        self,
        sampled_frame_indices: List[int],
        confidence_maps: np.ndarray,
        detected_objects_maps: List[List[List[str]]]
    ) -> Tuple[List[float], List[List[str]]]:
        """
        Update frame distribution based on detection results.

        Args:
            sampled_frame_indices: Indices of sampled frames
            confidence_maps: Confidence maps from detection
            detected_objects_maps: Detected objects per grid cell

        Returns:
            Tuple of (frame_confidences, frame_detected_objects)
        """
        confidence_map = confidence_maps[0]
        detected_objects_map = detected_objects_maps[0]
        grid_rows, grid_cols = self.image_grid_shape

        frame_confidences = []
        frame_detected_objects = []

        for idx, _ in enumerate(sampled_frame_indices):
            row = idx // grid_cols
            col = idx % grid_cols
            frame_confidences.append(confidence_map[row, col])
            frame_detected_objects.append(detected_objects_map[idx])

        # Mark frames as visited and update scores
        for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences):
            self.non_visiting_frames[frame_idx] = 0
            self.score_distribution[frame_idx] = confidence

        self.update_top_25_with_window(frame_confidences, sampled_frame_indices)
        self.P = self.spline_keyframe_distribution(
            self.non_visiting_frames,
            self.score_distribution,
            len(self.score_distribution)
        )
        self.store_score_distribution()

        return frame_confidences, frame_detected_objects

    # ==================== Sampling Methods ====================

    def sample_frames(self, num_samples: int) -> Tuple[List[int], List[np.ndarray]]:
        """
        Sample frames based on current probability distribution.

        Args:
            num_samples: Number of frames to sample

        Returns:
            Tuple of (sampled_frame_secs, resized_frames)
        """
        if num_samples > self.total_frame_num:
            num_samples = self.total_frame_num

        if not self.Score_history:
            # Initial uniform sampling
            interval = self.total_frame_num // num_samples
            sampled_frame_secs = np.arange(0, self.total_frame_num, interval)[:num_samples]
            if len(sampled_frame_secs) < num_samples:
                sampled_frame_secs = np.append(sampled_frame_secs, self.total_frame_num - 1)
        else:
            # Adaptive sampling based on probability distribution
            _P = (self.P + num_samples / self.total_frame_num) * self.non_visiting_frames
            _P = np.maximum(_P, 1e-8)
            _P = _P ** 1.5

            if _P.sum() == 0 or np.count_nonzero(_P) < num_samples:
                print("[Warning] Insufficient non-zero entries, using uniform distribution.")
                _P = np.ones(self.total_frame_num, dtype=np.float32) * self.non_visiting_frames
                _P = np.maximum(_P, 1e-8)

            _P /= _P.sum()

            sampled_frame_secs = np.random.choice(
                self.total_frame_num,
                size=num_samples,
                replace=False,
                p=_P
            )

        sampled_frame_indices = [int(sec * self.raw_fps / self.fps) for sec in sampled_frame_secs]
        indices, frames = self.read_frame_batch(self.video_path, sampled_frame_indices)
        resized_frames = [cv2.resize(frame, (200 * 4, 95 * 4)) for frame in frames]

        return sampled_frame_secs.tolist(), resized_frames

    def pop_frames(
        self, video_path: str, num_samples: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract top frames based on score distribution.

        Args:
            video_path: Path to video file
            num_samples: Number of frames to extract

        Returns:
            Tuple of (frames, time_stamps)
        """
        _P = self.score_distribution / self.score_distribution.sum()

        sampled_frame_secs = np.random.choice(
            self.total_frame_num, size=num_samples, replace=False, p=_P
        )
        sampled_frame_secs.sort()
        time_stamps_secs = [sec / self.fps for sec in sampled_frame_secs]

        frame_indices_in_video = [sec * self.raw_fps / self.fps for sec in time_stamps_secs]
        indices, frames = self.read_frame_batch(video_path, frame_indices_in_video)

        return frames, time_stamps_secs

    # ==================== Verification Methods ====================

    def verify_and_remove_target(
        self,
        frame_sec: int,
        detected_objects: List[str],
        confidence_threshold: float,
    ) -> bool:
        """
        Verify detection and remove target if confirmed.

        Args:
            frame_sec: Frame timestamp (in sampled seconds)
            detected_objects: Detected objects in the frame
            confidence_threshold: Threshold for confirmation

        Returns:
            True if target found and removed, else False
        """
        for target in list(self.remaining_targets):
            if target in detected_objects:
                frame_idx = int(frame_sec * self.raw_fps / self.fps)
                _, frame = self.read_frame_batch(self.video_path, [frame_idx])
                resized_frame = cv2.resize(frame[0], (200 * 3, 95 * 3))

                conf_map, det_obj_map = self.score_image_grids([resized_frame], (1, 1))
                single_confidence = conf_map[0, 0, 0]
                single_detected_objects = det_obj_map[0][0]
                self.score_distribution[frame_sec] = single_confidence

                self.image_grid_iters.append([resized_frame])
                self.detect_annotot_iters.append(
                    self.heuristic.bbox_visualization(
                        images=[resized_frame],
                        detections_inbatch=self.heuristic.detections_inbatch
                    )
                )
                self.detect_bbox_iters.append(self.heuristic.detections_inbatch)

                if target in single_detected_objects and single_confidence > confidence_threshold:
                    self.remaining_targets.remove(target)
                    print(f"[Info] Found target '{target}' at frame {frame_idx}, score {single_confidence:.2f}")
                    return True
        return False

    # ==================== Visualization Methods ====================

    def plot_single_iteration_score(
        self,
        save_path: Optional[str] = None,
        gt_spans: Optional[List] = None,
        selected_timestamps: Optional[List[float]] = None,
        iteration_name: str = "Current Iteration"
    ):
        """
        Plot score distribution with ground truth and selected anchors.

        Args:
            save_path: Path to save the plot
            gt_spans: Ground truth time spans
            selected_timestamps: Selected anchor timestamps
            iteration_name: Name for the iteration
        """
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.labelsize": 14,
            "font.size": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        })

        time_axis = np.linspace(0, self.duration, len(self.score_distribution))
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            time_axis, self.score_distribution,
            color="#1f77b4", linewidth=2.5, label=iteration_name
        )
        ax.fill_between(time_axis, self.score_distribution, alpha=0.2, color="#1f77b4")

        if gt_spans:
            spans_to_plot = [gt_spans] if isinstance(gt_spans[0], (int, float)) else gt_spans
            for i, sp in enumerate(spans_to_plot):
                label = "Ground Truth" if i == 0 else ""
                ax.axvspan(sp[0], sp[1], color="#2ca02c", alpha=0.3, label=label)

        if selected_timestamps:
            for i, ts in enumerate(selected_timestamps):
                label = "Selected Anchors" if i == 0 else ""
                ax.axvline(x=ts, color="#d62728", linestyle="--", linewidth=2, label=label)

        ax.set_xlabel("Time (s)", fontweight='bold')
        ax.set_ylabel("Relevance Score", fontweight='bold')
        ax.set_xlim([0, self.duration])

        max_score = max(self.score_distribution) if len(self.score_distribution) > 0 else 1.0
        ax.set_ylim(bottom=0.0, top=max_score * 1.25)

        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=save_path.split('.')[-1], bbox_inches='tight')
            print(f"[Info] Plot saved to {save_path}")

    # ==================== Main Search Logic ====================

    def search(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform keyframe search using object detection and dynamic sampling.

        This method iteratively samples frames, runs detection, updates the
        score distribution, and verifies targets until the search budget is
        exhausted or all targets are found.

        Returns:
            Tuple of (frames, timestamps)
            - frames: List of keyframe images
            - timestamps: List of corresponding timestamps in seconds
        """
        print(f"\n[T* Search] Starting object detection-based search")
        print(f"  Video duration: {self.duration:.1f}s")
        print(f"  Search budget: {self.search_budget} frames")
        print(f"  Target objects: {self.target_objects}")
        print(f"  Remaining targets: {self.remaining_targets}")

        iteration = 0

        while self.remaining_targets and self.search_budget > 0:
            iteration += 1
            print(f"\n[Iteration {iteration}]")

            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols

            # Sample frames based on current probability distribution
            sampled_frame_secs, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid

            # Create grid image
            grid_image = self.create_image_grid(frames, grid_rows, grid_cols)

            # Run object detection on grid
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )

            # Store visualization history
            self.image_grid_iters.append([grid_image])
            self.detect_annotot_iters.append(
                self.heuristic.bbox_visualization(
                    images=[grid_image],
                    detections_inbatch=self.heuristic.detections_inbatch
                )
            )
            self.detect_bbox_iters.append(self.heuristic.detections_inbatch)

            # Update frame distribution based on detection results
            frame_confidences, frame_detected_objects = self.update_frame_distribution(
                sampled_frame_indices=sampled_frame_secs,
                confidence_maps=confidence_maps,
                detected_objects_maps=detected_objects_maps
            )

            # Verify and remove found targets
            for frame_sec, detected_objects in zip(sampled_frame_secs, frame_detected_objects):
                self.verify_and_remove_target(
                    frame_sec=frame_sec,
                    detected_objects=detected_objects,
                    confidence_threshold=self.confidence_threshold,
                )

            print(f"  Sampled {len(sampled_frame_secs)} frames")
            print(f"  Max confidence: {max(frame_confidences):.3f}")
            print(f"  Remaining targets: {self.remaining_targets}")
            print(f"  Remaining budget: {self.search_budget}")

        # Extract top-k keyframes
        print(f"\n[T* Search] Extracting keyframes...")
        k_frames, time_stamps = self.pop_frames(
            video_path=self.video_path,
            num_samples=self.search_nframes
        )

        print(f"[T* Search] Complete! Found {len(time_stamps)} keyframes")

        return k_frames, time_stamps
