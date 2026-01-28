is_finetune=0
is_transfer=1

dset_finetune='test_norm'
context_points=192
batch_size=64
is_half=1

e_layers=3
d_layers=3

patch_len=48
stride=48

revin=1

model_type='LightGTS_4M'
pretrained_model='./checkpoints/LightGTS_4M.pth'

n_epochs_finetune=20
n_epochs_freeze=10


random_seed=2021


mkdir -p "logs/Forecasting/$model_type/$dset_finetune"

for train_file in data/test_norm/train_norm_*.csv
do
    data_id="$(basename "$train_file")"
    data_id="${data_id#train_norm_}"
    data_id="${data_id%.csv}"

    mkdir -p "logs/Forecasting/$model_type/$dset_finetune/$data_id"

    for target_points in 8 16 32 64 96
    do
        python -u zero_shot.py \
        --is_finetune $is_finetune \
        --is_transfer $is_transfer \
        --dset_finetune $dset_finetune \
        --data_id $data_id \
        --time_col_name Cycle \
        --unit_col_name UnitNumber \
        --is_half $is_half \
        --context_points $context_points \
        --target_points $target_points \
        --batch_size $batch_size \
        --patch_len $patch_len\
        --stride $stride\
        --revin 1 \
        --e_layers $e_layers\
        --d_layers $d_layers\
        --n_heads 8 \
        --d_model 256 \
        --d_ff 512\
        --dropout 0.2\
        --head_drop 0.2 \
        --n_epochs_finetune $n_epochs_finetune\
        --n_epochs_freeze $n_epochs_freeze\
        --lr 1e-4 \
        --finetuned_model_id 1\
        --pretrained_model $pretrained_model\
        --model_type $model_type\  >"logs/Forecasting/$model_type/$dset_finetune/$data_id/percentage$is_half"_finetune$is_finetune"_context$context_points"_target$target_points.log
    done

    python -u zero_shot.py \
    --is_finetune $is_finetune \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --data_id $data_id \
    --time_col_name Cycle \
    --unit_col_name UnitNumber \
    --is_half $is_half \
    --context_points 50 \
    --target_points 1 \
    --batch_size $batch_size \
    --patch_len $patch_len\
    --stride $stride\
    --revin 1 \
    --e_layers $e_layers\
    --d_layers $d_layers\
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --n_epochs_finetune $n_epochs_finetune\
    --n_epochs_freeze $n_epochs_freeze\
    --lr 1e-4 \
    --finetuned_model_id 1\
    --pretrained_model $pretrained_model\
    --model_type $model_type\  >"logs/Forecasting/$model_type/$dset_finetune/$data_id/percentage$is_half"_finetune$is_finetune"_context50"_target1.log
done
