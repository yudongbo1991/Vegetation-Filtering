## Requirements

- **OS**: Linux (recommended for compatibility)
- **Python**: 3.8
- **PyTorch**: 2.1.0+cu118

> We have tested the code under the above environment. Other versions may work but are not guaranteed.

---

## Data Processing

1. Put the training data (e.g., `hunluan2-1w.txt`) into the `/data` folder.

2. Run the following scripts **in sequence**:

   ```bash
   python sample.py
   python normals.py
   python 2D_pos.py
   python 1D_seq3w_in.py

3. The generated folder `hunluan2-1w-2d-3sw-3h-15` contains the processed data for training.
The same procedure applies to generating evaluation data.

---

## Model
Our model is located in the `/model` folder.
Pre-trained weights: PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1-hl1w-normaldiv.pth

---

## Training
   ```bash
   python train3_128.py \
    --data_folder="data/hunluan2-1w-2d-3sw-3h-15" \
    --model="PTM_128_ms2_3dmamba_xyz_cb_aw1"


The trained weight file will be saved automatically.

---

## Evaluation
   ```bash
   python evaluate3_128.py \
    --data_folder="data/hunluan2-32w-2d-3sw-3h-15" \
    --model="PTM_128_ms2_3dmamba_xyz_cb_aw1"


Note: Make sure to specify the evaluation dataset path in --data_folder. The filtered point cloud file will be saved automatically.
