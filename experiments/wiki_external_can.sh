# trains gan with an outside can network instead of having the discriminator learn style classification
export PYTHONPATH="slim/:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
python3 main.py \
--epoch 25 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 16 \
--sample_size 72 \
--input_height 128 \
--output_height 128 \
--lambda_val 1.0 \
--smoothing 1.0 \
--use_resize True \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--crop False \
--visualize False \
--use_s3 False \
--can True \
--train \
--style_net_checkpoint "slim/logs/wikiart/inception_resnet_v2/all/bs=16,lr=0.0001,epochs=100/smol_adam_fixedLR"
