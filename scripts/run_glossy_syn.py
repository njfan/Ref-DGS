import os

data_base_path = '/data/dataset/njfan/dataset_gaussian/Glossy/GlossySynthetic'
out_base_path = 'output/Glossy/GlossySynthetic'
log_base_path = 'logs/Glossy/GlossySynthetic'
gpu_id=1

for scene in os.listdir(data_base_path):
    source_path = os.path.join(data_base_path, scene)
    model_path = os.path.join(out_base_path, scene)
    log_path = os.path.join(log_base_path, scene)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cmd = f'rm -rf {model_path}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {source_path} -m {model_path} --eval --run_dim 64 --albedo_bias 2 --albedo_lr 0.0005 > "{log_path}/train" 2>&1'
    print(cmd)
    os.system(cmd)


    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {model_path} --dataset glossy > "{log_path}/render" 2>&1'
    print(cmd)
    os.system(cmd)