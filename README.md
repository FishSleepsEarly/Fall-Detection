The source code (only) can be downloaded from the following link: [https://drive.google.com/file/d/1lsAVv05lQzorBcA6qsWYSvwmLc-N4p5j/view?usp=drive_link](https://drive.google.com/file/d/1lsAVv05lQzorBcA6qsWYSvwmLc-N4p5j/view?usp=drive_link)

The data, model checkpoints, and source code can be downloaded from the following link: [https://drive.google.com/file/d/1Wh9le71yHOWgbesZMSdT5ZtLg2HjeMEj/view?usp=drive_link](https://drive.google.com/file/d/1Wh9le71yHOWgbesZMSdT5ZtLg2HjeMEj/view?usp=drive_link)

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
│   ├── database.py                    # Database management script
│   ├── expirenment.py                 # Script for training the model
│   ├── frame_manager.py               # Frame management script
│   ├── load_model.py                  # Script to load and evaluate the model
│   ├── net.py                         # Network architecture implementation
│   └── SAM2.py                        # Main script for SAM2 operations
├── data/
│   ├── train/
│   │   ├── fall/
│   │   │   ├── angles/                # Joint angle data
│   │   │   ├── mask_videos/           # Masked video data for "fall" category
│   │   │   ├── masks/                 # Masks corresponding to frames
│   │   │   ├── points/                # Annotated key points
│   │   │   ├── raw_frames/            # Raw extracted frames
│   │   │   └── raw_videos/            # Original raw videos
│   │   ├── not_fall/
│   │   │   ├── angles/                # Joint angle data
│   │   │   ├── mask_videos/           # Masked video data for "not_fall" category
│   │   │   ├── masks/                 # Masks corresponding to frames
│   │   │   ├── points/                # Annotated key points
│   │   │   ├── raw_frames/            # Raw extracted frames
│   │   │   └── raw_videos/            # Original raw videos
│   ├── test/
│   │   ├── fall/                      # Test data for "fall" category
│   │   │   ...
│   │   ├── not_fall/                  # Test data for "not_fall" category
│   │   │   ...
│   ├── val/
│   │   ├── fall/                      # Validation data for "fall" category
│   │   │   ...
│   │   ├── not_fall/                  # Validation data for "not_fall" category
│   │   │   ...
├── paper/
│   ├── MM811_Report.pdf               # Final project report (PDF format)
│   └── MM811_Report.tex               # Report source file (LaTeX format)
```

---

## Instructions for Setup and Execution

### Prerequisites

Ensure that the necessary dependencies are installed in your environment. These may include Python and relevant libraries such as PyTorch, OpenCV, and others required for the scripts. Refer to the `requirements.txt` if provided.

### Steps to Set Up the Project

1. **Download and Extract the Code**  
   After downloading the project, extract the ZIP file. The extracted folder should contain the following items:  
   - `checkpoints/`  
   - `configs/`  
   - `data/`  
   - `scripts/`  
   - `paper/`  
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

4. **Accessing the Project Report**  
   The final project report can be found in the `paper/` folder:  
   - PDF format: `paper/MM811_Report.pdf`  
   - LaTeX source: `paper/MM811_Report.tex`

4. **Out Source Code Citation**  
   All out source codes used are citated in our paper.
---

## Notes

- Ensure that the data structure in `data/` follows the correct organization, as shown above.
- Modify the configuration file `sam2.1_hiera_t.yaml` in the `configs/` directory to customize the model settings if necessary.

For additional information or troubleshooting, refer to the inline comments within the scripts.

---

## Authors

- Siming Hua
- Letian Xu
- Qilong Yu
