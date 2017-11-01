export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--epoch 25 \
--learning_rate .0004 \
--beta 0.5 \
--batch_size 64 \
--input_height 128 \
--output_height 128 \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--visualize False \
--can True \
--train
