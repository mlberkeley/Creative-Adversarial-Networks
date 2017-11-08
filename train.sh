export CUDA_VISIBLE_DEVICES=1
python3 main.py \
--epoch 25 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 25 \
--sample_size 12 \
--input_height 28 \
--output_height 28 \
--lambda_val 1.0 \
--smoothing 0.9 \
--dataset mnist \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--visualize False \
--can True \
--train
