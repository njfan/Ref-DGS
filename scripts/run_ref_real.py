import os

data_base_path = '/data/dataset/njfan/dataset_gaussian/Ref-NeRF/ref_real'
out_base_path = 'output/Ref-NeRF/ref_real'
log_base_path = 'logs/Ref-NeRF/ref_real'
gpu_id=2

env_scope_center={"gardenspheres":"-0.2270 1.9700 1.7740", "sedan":"-0.032 0.808 0.751", "toycar":"0.6810 0.8080 4.4550"}
env_scope_radius={"gardenspheres":"0.974", "sedan":"2.138", "toycar":"2.707"}
reso={"gardenspheres":"4", "sedan":"8", "toycar":"4"}
init_until_iter={"gardenspheres":"1500", "sedan":"700", "toycar":"1500"}
xyz_axis={"gardenspheres":"2.0 1.0 0.0", "sedan":"2.0 1.0 0.0", "toycar":"0.0 2.0 1.0"}
vggt_until_iter={"gardenspheres":"10000", "sedan":"0", "toycar":"10000"}

for scene in os.listdir(os.path.join(data_base_path)):
    source_path = os.path.join(data_base_path, scene)
    model_path = os.path.join(out_base_path, scene)
    log_path = os.path.join(log_base_path, scene)
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cmd = f'rm -rf {model_path}/*'
    print(cmd)
    os.system(cmd)

    # iteration=15,000: geometry and novel view synthesis are good enough.
    # iteration=25,000: rendering quality improves further.
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train-real.py -s {source_path} -m {model_path} --eval --iteration 15000 --vggt_until_iter {vggt_until_iter[scene]} --run_dim 64 --albedo_bias 2 --albedo_lr 0.0005 -r {reso[scene]} --env_scope_center {env_scope_center[scene]} --env_scope_radius {env_scope_radius[scene]} --init_until_iter {init_until_iter[scene]} --xyz_axis {xyz_axis[scene]} > "{log_path}/train" 2>&1'
    print(cmd)
    os.system(cmd)

    cmd = f' CUDA_VISIBLE_DEVICES={gpu_id} python render-real.py -m {model_path} > "{log_path}/render" 2>&1'
    print(cmd)
    os.system(cmd)