export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--epoch 25 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 32 \
--sample_size 72 \
--input_height 128 \
--output_height 128 \
--lambda_val 0.0 \
--smoothing 1.0 \
--use_resize True \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--visualize False \
--can True \
--train
