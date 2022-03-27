#!/usr/bin/bash

python ./train.py -c "1" -d "rgb" -g 2
python ./train.py -c "2" -d "rgb" -g 2
python ./train.py -c "3" -d "rgb" -g 2
python ./train.py -c "4" -d "rgb" -g 2

python ./train.py -c "1" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"
python ./train.py -c "2" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"
python ./train.py -c "3" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"
python ./train.py -c "4" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"

python ./train.py -c "1" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"
python ./train.py -c "2" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"
python ./train.py -c "3" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"
python ./train.py -c "4" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"

python ./predict.py -c "1" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"
python ./predict.py -c "2" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"
python ./predict.py -c "3" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"
python ./predict.py -c "4" -d "rgbd" -g 2 -depth "estimated_depth_fcrn"

python ./predict.py -c "1" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"
python ./predict.py -c "2" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"
python ./predict.py -c "3" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"
python ./predict.py -c "4" -d "rgbd" -g 2 -depth "estimated_depth_pix2pix"