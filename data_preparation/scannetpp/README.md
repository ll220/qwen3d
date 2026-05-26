# How to make SN++ work with our code

1. Download SN++ dataset from their official repo. You can use multiprocessiing enabled files
from this repo instead of the original ones: `download_scannetpp.py`, `download_scannetpp.yml` 
2. Run processing to extract RGBD frames. You will use their github repo
but you can replace their `prepare_iphone_data.py` and their `prepare_iphone_data.yml`
with our versions in this directory. Then run this from the scannetpp github repo:
`python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml`

3. Run `python process_snpp_rgbd_to_scannet.py` for converting SN++ RGB-D to our scannet-like format

4 Make jsons in coco format using scannetpp_to_coco.py
5. Process the mesh pc and labels using the following

```
python data_preparation/scannetpp/scannetpp_processing.py preprocess \
    --data_dir /data/group_data/katefgroup-ssd/datasets/scannetpp_full \
    --save_dir  /data/group_data/katefgroup/language_grounding/mask3d_processed/scannetpp \
    --modes ["validation",]
```

