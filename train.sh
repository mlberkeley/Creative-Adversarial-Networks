export CUDA_VISIBLE_DEVICES=1
python3 main.py \
--epoch 10 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 64 \
--sample_size 64 \
--input_height 28 \
--output_height 28 \
--dataset mnist \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--visualize False \
--can True \
--train
