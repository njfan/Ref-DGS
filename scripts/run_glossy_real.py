import os

data_base_path = '/data/dataset/njfan/dataset_gaussian/Glossy/GlossyReal'
out_base_path = 'output/Glossy/GlossyReal'
log_base_path = 'logs/Glossy/GlossyReal'
gpu_id=3

env_scope_center={"bear":"0.005 1.689 2.665", "bunny":"0.339 1.310 1.897", "coral":"0.025 1.281 1.669", "maneki":"-0.015 1.278 1.798", "vase":"0.085 1.858 1.591"}
env_scope_radius={"bear":"0.952", "bunny":"1.143", "coral":"0.878", "maneki":"1.101", "vase":"1.3"}

for scene in os.listdir(os.path.join(data_base_path)):
    source_path = os.path.join(data_base_path, scene)
    if not os.path.isdir(source_path):
        continue
    model_path = os.path.join(out_base_path, scene)
    log_path = os.path.join(log_base_path, scene)
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cmd = f'rm -rf {model_path}/*'
    print(cmd)
    os.system(cmd)

    cmd = f' CUDA_VISIBLE_DEVICES={gpu_id} python train-real.py -s {source_path} -m {model_path} --eval --vggt_weight 0.025 --vggt_until_iter 5000 --run_dim 64 --albedo_bias 2 --albedo_lr 0.0005 -r 4 --env_scope_center {env_scope_center[scene]} --env_scope_radius {env_scope_radius[scene]} --init_until_iter 1500 --xyz_axis 0.0 1.0 2.0 > "{log_path}/train" 2>&1'
    print(cmd)
    os.system(cmd)

    cmd = f' CUDA_VISIBLE_DEVICES={gpu_id} python render-real.py -m {model_path} > "{log_path}/render" 2>&1'
    print(cmd)
    os.system(cmd)