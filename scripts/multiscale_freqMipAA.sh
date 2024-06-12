#gpu6,1
test_name=freqmipaa_multiscale_blender

GPU_NUM=1
adding_mask=(7000)
adjust_lr=(20000000)

for data in lego chair drums ficus mic ship hotdog materials ; do 
   
    echo "############################################################"
    echo "${data}  start"
    echo "############################################################"
    CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py \
        --n_threads=4 \
        --config=configs/blender/mip_tensor_freq_vm.yaml \
        --exp_name="${test_name}/${data}" \
        --model.scale_types="cylinder_radius" \
        --model.learnable_mask \
        --model.train_freq_domain \
        --dataset.data_dir=/home/dataset/nerf/nerf_synthetic/$data \
        --model.force_seperate \
        --model.force_mask \
        --training.n_iters=40000 \
        --training.vis_every=20000 \
        --training.adding_mask "${adding_mask[@]}" \
        --training.adjust_mask_lr "${adjust_lr[@]}" 
    echo "=============================================================="
    echo "$data end"
    echo "=============================================================="

done

