# 01 PROJECT INVENTORY

## Complete File Inventory

### Python Modules

| Directory | File | Lines | Purpose |
|-----------|------|-------|---------|
| models/ | __init__.py | 31 | Public API exports |
| models/ | factory.py | 751 | ModelFactory, backbone zoo, all model creation |
| models/ | classification.py | 122 | BrainTumorClassifier3D (unused 3D CNN) |
| models/ | segmentation.py | 398 | 3D U-Net (UNet3D) |
| models/ | tumor_aware.py | 410 | TumorAwareMultiTaskNet (proposed model) |
| models/ | attention.py | 97 | CBAM3D (3D attention modules) |
| models/ | initialization.py | 71 | Kaiming/TruncNormal init |
| training/ | classification_trainer.py | 1094 | Unified config-driven trainer |
| training/ | segmentation_trainer.py | 764 | 3D U-Net BraTS trainer |
| training/ | self_supervised_trainer.py | 699 | SimCLR pretraining |
| training/ | multitask_trainer.py | 279 | Multi-task TumorAware trainer |
| training/ | training_manager.py | 652 | Persistent stop/resume manager |
| datasets/ | __init__.py | — | Module init |
| datasets/ | classification.py | 560 | Classification dataset discovery + splitting |
| datasets/ | segmentation.py | 628 | Patch-based BraTS segmentation dataset |
| datasets/ | brats.py | 844 | BraTS NIfTI discovery + patient-level split |
| datasets/ | dataloaders.py | 201 | Shared DataLoader utilities |
| datasets/ | preprocessing.py | 159 | z-score MRI normalization |
| datasets/ | splitting.py | 392 | Patient-level split manifest management |
| losses/ | __init__.py | — | Module init |
| losses/ | classification.py | 136 | CrossEntropy + label smoothing |
| losses/ | segmentation.py | 256 | DiceLoss3D + Dice/CE combo |
| evaluation/ | __init__.py | — | Module init |
| evaluation/ | classification_evaluator.py | 579 | Test-set evaluator |
| evaluation/ | classification_visualizer.py | — | Classification result plots |
| evaluation/ | comprehensive_metrics.py | 657 | 30+ metrics all backbones |
| evaluation/ | explainability.py | 552 | Grad-CAM heatmaps |
| evaluation/ | segmentation_evaluator.py | 528 | Sliding-window BraTS eval |
| evaluation/ | advanced_robustness.py | 354 | 9 corruption types + MRRS |
| evaluation/ | calibration.py | 342 | ECE, Brier, MCE, reliability diagrams |
| evaluation/ | uncertainty.py | 258 | MC Dropout uncertainty |
| evaluation/ | cross_dataset_analysis.py | 192 | Domain shift analysis |
| evaluation/ | statistical_validation.py | — | Statistical tests |
| evaluation/ | efficiency_analysis.py | — | Latency/FPS benchmarks |
| evaluation/ | ablation_runner.py | — | Ablation experiments |
| evaluation/ | densenet121_evaluator.py | — | DenseNet121-specific evaluator |
| evaluation/ | generate_metric_figures.py | — | Metric visualization |
| evaluation/ | gradcam_localization.py | — | Grad-CAM localization |
| engine/ | __init__.py | — | Module init |
| engine/ | inference.py | 1330 | InferenceEngine |
| engine/ | checkpoint.py | 349 | CheckpointManager |
| engine/ | pipeline.py | 821 | BrainTumorPipeline |
| engine/ | report.py | 382 | Report generator |
| engine/ | visualization.py | 801 | MRI visualization engine |
| inference/ | __init__.py | — | Module init |
| inference/ | predict.py | 128 | CLI single-case inference |
| inference/ | pipeline.py | — | Pipeline inference entry |
| utils/ | __init__.py | — | Module init |
| utils/ | device.py | 67 | CUDA/CPU device selection |
| utils/ | metrics.py | 147 | Dice/IoU/Accuracy metrics |
| utils/ | seed.py | 36 | Deterministic seeding |
| utils/ | logging.py | 58 | Application-wide logging |
| tests/ | test_models.py | 169 | Model factory + backbone tests |
| tests/ | test_trainer.py | 295 | Unified trainer end-to-end |
| tests/ | test_training_manager.py | 118 | Training manager tests |
| tests/ | test_datasets.py | 121 | Dataset discovery + splitting |
| tests/ | test_engine.py | 166 | Inference engine tests |
| tests/ | test_evaluator.py | 85 | Evaluator tests |
| tests/ | test_explainability.py | 187 | Grad-CAM tests |
| tests/ | test_metrics.py | 84 | Metric utility tests |
| tests/ | test_self_supervised.py | 182 | SimCLR module tests |

### Configuration Files

| File | Purpose |
|------|---------|
| configs/config.yaml | Project meta-config |
| configs/classification.yaml | CNN-2D baseline config |
| configs/cnn2d_improved.yaml | CNN-2D improved config |
| configs/densenet121.yaml | DenseNet121 baseline |
| configs/resnet50.yaml | ResNet50 baseline |
| configs/efficientnet_v2_s.yaml | EfficientNetV2-S baseline |
| configs/convnext_tiny.yaml | ConvNeXt-Tiny baseline |
| configs/swin_t.yaml | Swin Transformer-Tiny baseline |
| configs/vit_b_16.yaml | ViT-B/16 baseline |
| configs/proposed_model.yaml | TumorAwareMultiTaskNet config |
| configs/dataset.yaml | BraTS + classification dataset config |
| configs/paths.yaml | All project paths |
| configs/training.yaml | Global training settings |
| configs/segmentation.yaml | 3D U-Net segmentation config |
| configs/self_supervised.yaml | SimCLR pretraining config |

### Checkpoints

| Path | Status |
|------|--------|
| checkpoints/classification/cnn2d/best.pt | EXISTS |
| checkpoints/classification/cnn2d/last.pt | EXISTS |
| checkpoints/classification/densenet121/best.pt | EXISTS |
| checkpoints/classification/densenet121/last.pt | EXISTS |
| checkpoints/classification/resnet50/best.pt | EXISTS |
| checkpoints/classification/resnet50/last.pt | EXISTS |
| checkpoints/classification/efficientnet_v2_s/best.pt | EXISTS |
| checkpoints/classification/efficientnet_v2_s/last.pt | EXISTS |
| checkpoints/classification/convnext_tiny/best.pt | EXISTS |
| checkpoints/classification/convnext_tiny/last.pt | EXISTS |
| checkpoints/classification/swin_t/best.pt | EXISTS |
| checkpoints/classification/swin_t/last.pt | EXISTS |
| checkpoints/classification/vit_b_16/best.pt | EXISTS |
| checkpoints/classification/vit_b_16/last.pt | EXISTS |
| checkpoints/segmentation/unet3d/best/unet3d_segmentation.pt | EXISTS |
| checkpoints/segmentation/unet3d/last/unet3d_segmentation.pt | EXISTS |
| checkpoints/self_supervised/densenet121/best.pt | EXISTS |
| checkpoints/self_supervised/densenet121/last.pt | EXISTS |

### Scripts

| Script | Purpose |
|--------|---------|
| scripts/train_classification.py | Classification training CLI |
| scripts/train_segmentation.py | Segmentation training CLI |
| scripts/train_self_supervised.py | SSL pretraining CLI |
| scripts/manage_training.py | Training manager CLI |
| scripts/training_progress.py | Progress monitoring |
| scripts/run_research_experiments.py | Full 7-phase experiment pipeline |
| scripts/run_multi_seed.py | Multi-seed validation |
| scripts/run_ablation.py | Ablation experiments |
| scripts/run_all.py | Run all experiments |
| scripts/run_smoke_test.py | Quick smoke test |
| scripts/run_remaining.py | Run remaining experiments |
| scripts/run_post_training.py | Post-training analysis |
| scripts/run_gradcam.py | Grad-CAM generation |
| scripts/generate_tables.py | 12 research tables |
| scripts/generate_figures.py | 16 research figures |
| scripts/generate_gradcam_figures.py | Grad-CAM figure generation |
| scripts/generate_paper.py | IEEE LaTeX paper generation |
| scripts/export_all_outputs.py | Consolidated output export |
| scripts/aggregate_results.py | Result aggregation |
| scripts/rebuild_results.py | Result rebuilding |
| scripts/predict_classification.py | Classification prediction |
| scripts/evaluate_kaggle_full.py | Kaggle dataset evaluation |
| scripts/evaluate_segmentation.py | Segmentation evaluation |
| scripts/test_new_dataset.py | New dataset testing |
| scripts/validate_brats.py | BraTS data validation |
