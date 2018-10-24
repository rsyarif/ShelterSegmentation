## Input / Output structure

Currently this code requires the following separate folders:


shelterdata/180505_v1/input/train/ #train data here

shelterdata/180505_v1/input/test/ #test data here

shelterdata/180505_v1/internal/checkpoints/ #code will create folders and files based on model names

shelterdata/180505_v1/internal/npy #temporary files to store train/test names

shelterdata/180505_v1/internal/prev_checkpoints_to_load/ #place weights to load here if training using existing weights

shelterdata/180505_v1/output/ #output file/folders based on model name will be stored here