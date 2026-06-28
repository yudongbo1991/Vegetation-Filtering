Requirements
Linux is recommended for compatibility reasons.
We have tested on Python 3.8 with PyTorch 2.1.0+cu118.

Data Process
Put the training data (taking hunluan2-1w. txt as an example) into the/data folder
Run "python sample.py", "python noramls.py", "python 2D_pos.py", and "python 1D_seq3w_in.py" in sequence.
The generated "hunluan2-1w-2d-3sw-3h-15" folder contains the data used for training.
The same applies to the method of generating evaluation data.

Model
Our model is in the /model folder.
The trained parameter weights. psh file is "PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1-hl1w-normaldiv.pth"

Train&Evaluate
Run "python train3_128.py --data_folder="data/hunluan2-1w-2d-3sw-3h-15" --model="PTM_128_ms2_3dmamba_xyz_cb_aw1".py"
The program will automatically save the trained weight file.

Run "python evaluate3_128.py --data_folder="data/hunluan2-32w-2d-3sw-3h-15" --model="PTM_128_ms2_3dmamba_xyz_cb_aw1".py"
Please note that the file path here is the file path of the evaluation dataset.
The program will automatically save the filtered point cloud file.
