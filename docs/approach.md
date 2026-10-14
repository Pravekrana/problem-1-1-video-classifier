# Video classifier — architecture and evaluation

**Objective:** Reproducible, pretrained Kinetics-400 action recognition. This is a baseline, not a fine-tuned VideoMME model. VideoMME measures video question answering; its score cannot be equated with action-classification accuracy.

**Pipeline:** OpenCV video decoding → uniformly sample 16 frames → resize shorter side to 128 → center crop 128×128 → normalize RGB with Kinetics statistics → torchvision pretrained R3D-18 → softmax top-5 labels → JSON results.

**Trade-offs:** R3D-18 (~33M parameters) provides a simple 3D convolutional baseline but is not a tiny edge model. Full sequential decoding is portable but O(video length) and stores decoded frames, unsuitable for very long videos. Uniform sampling can miss brief events. Kinetics labels may not represent surveillance, wildlife or gameplay accurately. The initial weight download requires internet (~127 MB). CPU and CUDA supported.

**Evaluation:** Provide representative videos and a CSV with columns `video,label` where labels exactly match Kinetics categories. Run `python -m src.main --input path/to/videos --labels labels.csv`. Output reports top-1/top-5 accuracy for matched videos and per-video wall-clock latency (decode, preprocess, inference), excluding model initialization. Use held-out domain-specific labels, fixed random seed, and report hardware, p50/p95 latency, memory and confusion matrix for serious benchmarking. No real-world accuracy numbers are claimed here; the synthetic video only checks execution.

**Future work:** Decode-time frame sampling, temporal ensemble, domain fine-tuning, ONNX/quantization, calibrated confidence, streaming memory and domain-specific evaluation.
