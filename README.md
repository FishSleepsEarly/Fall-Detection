
# Project Documentation

## File Structure
```
project/
├── checkpoints/
│   └── sam2.1_hiera_tiny.pt
├── configs/
│   └── sam2.1_hiera_t.yaml
├── scripts/
│   ├── frame_manager.py
│   └── SAM2.py
├── data/
│   ├── train/
│   │   ├── fall/
│   │   │   ├── mask_videos/
│   │   │   ├── masks/
│   │   │   ├── point_frames/
│   │   │   ├── point_mask_videos/
│   │   │   ├── point_videos/
│   │   │   ├── points/
│   │   │   ├── raw_frames/
│   │   │   └── raw_videos/
│   │   ├── not_fall/
│   │   │   ...
│   ├── test/
│   │   ├── fall/
│   │   │   ...
│   │   ├── not_fall/
│   │   │   ...


```
