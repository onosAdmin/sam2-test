This repository is based on the instructions on:

https://github.com/facebookresearch/segment-anything-2

https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb


## Start

## Download SAM2
```
git clone https://github.com/facebookresearch/segment-anything-2

```
## Create the virtual env and activate it

```
cd segment-anything-2
python3 -m venv env

source env/bin/activate
```

## Download SAM2 checkpoints
```
cd checkpoints
./download_ckpts.sh

```
## Install SAM2

```
pip install -e .

pip install matplotlib opencv-python
```

Edit the script mask_test.py
replacing your image path in:
image_path = 'photo.jpg'




## Change the used checkpoints and config based on the checkpoints used
```
sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

```


## Run the test
```
python test.py
```
