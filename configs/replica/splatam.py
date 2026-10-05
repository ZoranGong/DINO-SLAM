import os

scenes = ["room0", "room1", "room2",
          "office0", "office1", "office2",
          "office3", "office4"]

seed = int(os.environ["SEED"]) if "SEED" in os.environ else 0
scene_name = scenes[int(os.environ["SCENE_NUM"]) if "SCENE_NUM" in os.environ else 0] 

name = 'Replica'
config = dict(
    vis_base_dir = f'output/splatam_{name}_seed{seed}/{scene_name}',
    seed = seed,
    primary_device='cuda',
    use_wandb = True,
    wandb = dict(
        name = name,
        project_name = "GauS_SLAM_Replica",
    ),
          
    render = dict(
        method = '3dgs',
        use_sa = True, 
        use_weight_norm = False, 
        enable_exposure = False,
        eps = 1e-6,
        depth_far = 1e2,
        depth_near = 1e-2,
    ),
    
    frontend = dict(
        map_every = 1,
        keyframe_every = 5, 
        num_tracking_iters = 40,
        num_mapping_iters = 60,
        num_overlap_frames = 24,
        additional_densify = False, 
    ), 

    densify=dict(
        use_edge_growth = False,
        densify_interval=10,
        method='splatam',
        sil_thres=0.5,
        dep_thres=0.1,
        opacity_cuil=0.05,
        scale_cuil=5e-4,
        scale_max=0.1,
        num_addpts=4000000,
        percent_dense=0.01,
        densify_grad_threshold=0.0002,
        extent=2,
    ),
    
    loss = dict(
        ignore_outliners = True,
        use_normal_loss = False,
        silmask_th = 0.99,
        tracking = dict(
            color=0.5,
            depth=1.0,
            normal = 0,
            dist = 0,
        ),
        mapping = dict(
            color=0.5,
            depth=1.0,
            normal = 0,
            dist = 0,
        ),
    ),
    data=dict(
        dataset_name = "Replica",
        meshdir="./data/Replica/cull_replica_mesh",
        basedir="./data/Replica",
        gradslam_data_cfg="./configs/data/replica.yaml",
        sequence=scene_name,
        desired_image_height=340 * 2,
        desired_image_width=600 * 2,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    gaussians=dict(
        gaussian_distribution = "isotropic", # ["isotropic", "anisotropic"]
        training_args=dict(
            xyz_lr= 0.0001,
            feature_lr=0.0025,
            rgb_lr=0.0025,
            rotation_lr= 0.001,
            opacity_lr=0.05,
            scaling_lr=0.001,
        ),
    ),
    cameras=dict(
        adam_betas = (0.7, 0.99),
        frontend_lr=dict(
            cam_rot_lr_init =  0.0004, # 0.0004,  # / 100
            cam_rot_lr_final = 0.00004,
            cam_rot_lr_max_step = 40, 
            cam_trans_lr_init =  0.002, # 0.002, # / 100
            cam_trans_lr_final = 0.0002,
            cam_trans_lr_max_step = 40,
            exposure_lr_init = 0.001,
            exposure_lr_final = 0.0001,
            exposure_lr_max_step = 60,
        ),
    ),
    viz=dict(
        viz_w=600,  
        viz_h=340,
        view_scale=2,
    ),
    eval=dict(
        save_renders = False, 
        eval_mesh = True, 
        save_mesh = True, 
        mesh_interval = 5, 
        voxel_size = 0.01,
    ),
)