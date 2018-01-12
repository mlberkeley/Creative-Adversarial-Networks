export CUDA_VISIBLE_DEVICES=1
python3 download_and_convert_data.py  \
    --dataset_name "wikiart" \
    --dataset_dir "/data/wikiart-records" \
    --input_dataset_dir "/data/wikiart/"
