"""
Complete End-to-End Example: T* Search + Temporal Grounding

This script demonstrates the full 4-phase pipeline:
1. Anchor Localization (T* Search with VLM)
2. Adaptive Window Generation (Clustering + Expansion)
3. Grounder Inference (Window-based prediction)
4. Aggregation & Deduplication (Final results)
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from TStar.qwen_searcher_minimal import QwenTStarSearcher


def compute_temporal_iou(span1, span2):
    """
    Compute temporal Intersection over Union (IoU) between two time spans.

    Args:
        span1: [start, end] in seconds
        span2: [start, end] in seconds

    Returns:
        IoU value between 0.0 and 1.0
    """
    s1, e1 = span1
    s2, e2 = span2

    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)

    return intersection / union if union > 0 else 0.0


def phase1_anchor_localization(video_path, query, duration, model, processor, device):
    """
    Phase 1: Use T* Search with VLM to locate temporal anchors.

    Args:
        video_path: Path to video file
        query: Natural language query
        duration: Video duration in seconds
        model: Qwen2-VL model
        processor: Qwen2-VL processor
        device: Device for inference

    Returns:
        List of anchor timestamps in seconds
    """
    print("\n" + "="*80)
    print("PHASE 1: ANCHOR LOCALIZATION")
    print("="*80)

    searcher = QwenTStarSearcher(
        video_path=video_path,
        target_objects=[query],
        qwen_model=model,
        processor=processor,
        device=device,
        search_nframes=min(150, int(duration / 3)),
        image_grid_shape=(4, 4),
        search_budget=int(duration * 1.0),
    )

    # Run iterative search (5 iterations by default)
    for iteration in range(5):
        print(f"\n[Iteration {iteration + 1}/5]")
        frame_secs, frames = searcher.sample_frames(num_samples=16)
        grid_img = searcher.create_image_grid(frames, rows=4, cols=4)
        conf_maps, det_objs = searcher.score_image_grids([grid_img], (4, 4))
        searcher.update_frame_distribution(frame_secs, conf_maps, det_objs)

    # Extract anchors
    _, time_stamps = searcher.pop_frames(num_samples=8)

    print(f"\n[Phase 1 Complete] Found {len(time_stamps)} anchors: {time_stamps}")

    return time_stamps, searcher


def phase2_window_generation(time_stamps, duration):
    """
    Phase 2: Generate adaptive windows from anchors.

    Args:
        time_stamps: List of anchor timestamps
        duration: Video duration in seconds

    Returns:
        List of [start, end] windows
    """
    print("\n" + "="*80)
    print("PHASE 2: ADAPTIVE WINDOW GENERATION")
    print("="*80)

    if len(time_stamps) == 0:
        print("[Warning] No anchors found, using full video as window")
        return [[0.0, duration]]

    time_stamps = sorted(time_stamps)

    # Duration-adaptive parameters
    tau_safe = min(max(duration * 0.05, 3.0), 20.0)
    base_radius = min(max(duration * 0.025, 3.0), 25.0)

    print(f"[Config] Clustering threshold: {tau_safe:.1f}s")
    print(f"[Config] Base expansion radius: {base_radius:.1f}s")

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

    print(f"\n[Clustering] Found {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {cluster} (span: {cluster[-1] - cluster[0]:.1f}s)")

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

    print(f"\n[Expansion] Generated {len(windows)} windows:")
    for i, (ws, we) in enumerate(windows):
        print(f"  Window {i+1}: [{ws:.1f}s, {we:.1f}s] (length: {we-ws:.1f}s)")

    # Merge overlapping windows
    merged = []
    for ws, we in sorted(windows, key=lambda x: (x[0], x[1])):
        if not merged:
            merged.append([ws, we])
            continue

        ps, pe = merged[-1]
        merged.append([ws, we])

    print(f"\n[Merging] After merging: {len(merged)} windows:")
    for i, (ws, we) in enumerate(merged):
        print(f"  Window {i+1}: [{ws:.1f}s, {we:.1f}s] (length: {we-ws:.1f}s)")

    print(f"\n[Phase 2 Complete] Final windows: {merged}")

    return merged


def phase3_grounder_inference(video_path, query, windows, grounder):
    """
    Phase 3: Run grounder on each window.

    Args:
        video_path: Path to video file
        query: Natural language query
        windows: List of [start, end] windows
        grounder: Temporal grounding model

    Returns:
        List of predictions with confidence scores
    """
    print("\n" + "="*80)
    print("PHASE 3: GROUNDER INFERENCE")
    print("="*80)

    all_predictions = []

    for i, (w_start, w_end) in enumerate(windows):
        print(f"\n[Window {i+1}/{len(windows)}] Processing [{w_start:.1f}s, {w_end:.1f}s]")

        # Run grounder on this window
        predictions = grounder.predict(
            video_path=video_path,
            query=query,
            window=[w_start, w_end],
            top_k=10
        )

        print(f"  Generated {len(predictions)} predictions")
        all_predictions.extend(predictions)

    print(f"\n[Phase 3 Complete] Total predictions: {len(all_predictions)}")

    return all_predictions


def phase4_aggregation(all_predictions, top_k=5):
    """
    Phase 4: Aggregate and deduplicate predictions.

    Args:
        all_predictions: List of predictions from all windows
        top_k: Number of final predictions to return

    Returns:
        List of top-k deduplicated predictions
    """
    print("\n" + "="*80)
    print("PHASE 4: AGGREGATION & DEDUPLICATION")
    print("="*80)

    if len(all_predictions) == 0:
        print("[Warning] No predictions to aggregate")
        return []

    # Sort by confidence
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    print(f"\n[Sorting] Top-10 by confidence:")
    for i, pred in enumerate(all_predictions[:10]):
        print(f"  {i+1}. [{pred['span'][0]:.1f}s, {pred['span'][1]:.1f}s] "
              f"conf={pred['confidence']:.3f}")

    # Deduplicate based on temporal IoU
    final_predictions = []
    iou_threshold = 0.75

    for pred in all_predictions:
        is_duplicate = False

        for final_pred in final_predictions:
            iou = compute_temporal_iou(pred['span'], final_pred['span'])
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            final_predictions.append(pred)

        if len(final_predictions) >= top_k:
            break

    print(f"\n[Deduplication] After removing duplicates (IoU > {iou_threshold}):")
    for i, pred in enumerate(final_predictions):
        print(f"  {i+1}. [{pred['span'][0]:.1f}s, {pred['span'][1]:.1f}s] "
              f"conf={pred['confidence']:.3f}")

    print(f"\n[Phase 4 Complete] Final {len(final_predictions)} predictions")

    return final_predictions


def main():
    """Main function demonstrating the complete pipeline."""
    # Configuration
    video_path = "path/to/video.mp4"
    query = "person opening a door"
    duration = 120.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("T* SEARCH + TEMPORAL GROUNDING PIPELINE")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Query: {query}")
    print(f"Duration: {duration}s")
    print(f"Device: {device}")

    # Load VLM
    print("\n[Loading] Qwen2-VL model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Phase 1: Anchor Localization
    time_stamps, searcher = phase1_anchor_localization(
        video_path, query, duration, model, processor, device
    )

    # Visualize
    searcher.plot_single_iteration_score(
        save_path="phase1_anchors.png",
        selected_timestamps=time_stamps,
        iteration_name="T* Search Result"
    )
    print("\n[Visualization] Saved to phase1_anchors.png")

    # Phase 2: Window Generation
    windows = phase2_window_generation(time_stamps, duration)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
