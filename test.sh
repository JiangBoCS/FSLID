python3 ./test.py --arch FS --batch_size 1 --gpu '0' \
    --input_dir /input/ \
    --gt_dir /GT/ \
    --result_dir /Ours_15/ \
    --weights /model_best_K60_G15.pth \
    --embed_dim 64 --FS 1