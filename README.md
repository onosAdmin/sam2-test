This repository is based on the instructions on:

https://github.com/facebookresearch/segment-anything-2

https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb


## Donwload this repository
```
git clone https://github.com/onosAdmin/sam2-test.git
cd sam2-test

```


## Download SAM2
```
git clone https://github.com/facebookresearch/segment-anything-2
cd segment-anything-2

```
## Create the virtual env and activate it

```
python3 -m venv env

source env/bin/activate
```


## Install SAM2

```
pip install -e .

pip install matplotlib opencv-python
```

Edit the script mask_test.py
replacing your image path in:
image_path = 'img0.jpg'


## Download SAM2 checkpoints
```
cd checkpoints
./download_ckpts.sh
cd ..
```

## Move test files from previous folder to SAM2 folder
```
cp ../test.py .
cp ../mask_test.py .
```

## Change the used checkpoints and config based on the checkpoints used inside the test.py and mask_test.py scripts
```
sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

```


## Run the test
```
python test.py
```
