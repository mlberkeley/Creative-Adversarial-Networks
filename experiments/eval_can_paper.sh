# export CUDA_VISIBLE_DEVICES=0 # edit this if you want to limit yourself to GPU
export PYTHONPATH="slim/:$PYTHONPATH"
python3 main.py \
--epoch 25 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 16 \
--sample_size 16 \
--input_height 256 \
--output_height 256 \
--lambda_val 1.0 \
--smoothing 1.0 \
--use_resize True \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--load_dir "logs/can_paper"
--crop False \
--visualize False \
--can True \
--train False
