
python3 main.py \
--epoch 5 \
--learning_rate .0002 \
--beta 0.5 \
--batch_size 6 \
--input_height 128 \
--output_height 128 \
--dataset wikiart \
--input_fname_pattern */*/.jpg \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--visualize False \
--train

-
