export CUDA_VISIBLE_DEVICES=1
python3 main.py \
--epoch 25 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 32 \
--sample_size 256 \
--input_height 128 \
--output_height 128 \
--lambda_val 1.0 \
--smoothing 0.9 \
--use_resize False \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--visualize False \
--can False \
--train
