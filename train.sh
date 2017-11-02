export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--epoch 25 \
--learning_rate .0002 \
--beta 0.5 \
--batch_size 36 \
--input_height 64 \
--output_height 64 \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--visualize False \
--can False \
--train
