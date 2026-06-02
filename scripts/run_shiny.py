import os

data_base_path = '/data/dataset/njfan/dataset_gaussian/Ref-NeRF/refnerf'
out_base_path = 'output/Ref-NeRF/refnerf'
log_base_path = 'logs/Ref-NeRF/refnerf'
gpu_id=0

for scene in os.listdir(data_base_path):
    source_path = os.path.join(data_base_path, scene)
    model_path = os.path.join(out_base_path, scene)
    log_path = os.path.join(log_base_path, scene)
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cmd = f'rm -rf {model_path}/*'
    print(cmd)
    os.system(cmd)

    if scene == "coffee":
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {source_path} -m {model_path} --eval --iterations 15000 --run_dim 64 --albedo_lr 0.002 > "{log_path}/train" 2>&1'
    elif scene == "ball":
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {source_path} -m {model_path} --eval --iterations 15000 --run_dim 64 > "{log_path}/train" 2>&1'
    elif scene == "car":
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {source_path} -m {model_path} --eval --run_dim 64 > "{log_path}/train" 2>&1'
    else:
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {source_path} -m {model_path} --eval --iterations 20000 --run_dim 64 > "{log_path}/train" 2>&1'
    print(cmd)
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {model_path} --dataset shiny > "{log_path}/render" 2>&1'
    print(cmd)
    os.system(cmd)