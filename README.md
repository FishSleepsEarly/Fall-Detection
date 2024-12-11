# Project Documentation

## File Structure

The project directory is organized as follows:

```
project/
├── checkpoints/
│   └── sam2.1_hiera_tiny.pt           # Pre-trained model checkpoint
├── configs/
│   └── sam2.1_hiera_t.yaml            # Configuration file for the model
├── scripts/
│   ├── frame_manager.py               # Frame management script
│   └── SAM2.py                        # Main script for SAM2 operations
├── data/
│   ├── train/
│   │   ├── fall/
│   │   │   ├── mask_videos/           # Masked video data for "fall" category
│   │   │   ├── masks/                 # Masks corresponding to frames
│   │   │   ├── point_frames/          # Frames with annotated key points
│   │   │   ├── point_mask_videos/     # Masked videos with key points
│   │   │   ├── point_videos/          # Videos with key points
│   │   │   ├── points/                # Annotated key points
│   │   │   ├── raw_frames/            # Raw extracted frames
│   │   │   └── raw_videos/            # Original raw videos
│   │   ├── not_fall/
│   │   │   ...                        # Data structure mirrors "fall" category
│   ├── test/
│   │   ├── fall/                      # Test data for "fall" category
│   │   │   ...
│   │   ├── not_fall/                  # Test data for "not_fall" category
│   │   │   ...
```

---

## Instructions for Setup and Execution
The data, model checkpoints, and source code can be downloaded from the following link: [example.com](https://example.com)

### Prerequisites

Ensure that the necessary dependencies are installed in your environment. These may include Python and relevant libraries such as PyTorch, OpenCV, and others required for the scripts. Refer to the `requirements.txt` if provided.

### Steps to Set Up the Project

1. **Download and Extract the Code**  
   After downloading the project, extract the ZIP file. The extracted folder should contain the following items:  
   - `checkpoints/`  
   - `configs/`  
   - `data/`  
   - `scripts/`  
   - `README.md`  

2. **Testing the Trained Model**  
   To evaluate the pre-trained model on the test set:  
   - Open a terminal.  
   - Navigate to the `scripts/` directory:  
     ```bash
     cd scripts
     ```  
   - Execute the following command to load the model checkpoint:  
     ```bash
     python load_model.py
     ```  
   - This script loads the checkpoint file `epoch=9-step=170.ckpt` and evaluates it on the test dataset.

3. **Training a New Model**  
   To train a new model from scratch:  
   - Open a terminal and navigate to the `scripts/` directory:  
     ```bash
     cd scripts
     ```  
   - Run the training script:  
     ```bash
     python expirenment.py
     ```  
   - The newly trained model will be saved under:  
     ```
     scripts/lightning_logs/version_n/checkpoints/
     ```  
     Here, `n` refers to the latest version of the model.

---

## Notes

- Ensure that the data structure in `data/` follows the correct organization, as shown above.
- Modify the configuration file `sam2.1_hiera_t.yaml` in the `configs/` directory to customize the model settings if necessary.

For additional information or troubleshooting, refer to the inline comments within the scripts.

---
