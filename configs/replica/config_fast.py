import os

# scenes = ["room0", "room1", "room2",
#           "office0", "office1", "office2",
#           "office3", "office4"]
scenes = ["room0"]


seed = int(os.environ["SEED"]) if "SEED" in os.environ else 0
scene_name = scenes[int(os.environ["SCENE_NUM"]) if "SCENE_NUM" in os.environ else 0] 
exp = int(os.environ["EXP"]) if "EXP" in os.environ else 0

name = "Fast_Replica"
h = 340 * 2
w = 600 * 2
trans_lr_base = 0.002
rot_lr_base = 0.0004
num_tracking_iter = 40
num_mapping_iters = 30
num_ba_iters = 40
localmap_max_frames = 80

config = dict(
    vis_base_dir = f'output/{name}_exp{exp}_seed{seed}/{scene_name}',
    seed = seed,
    primary_device='cuda',
    use_wandb = False,
    wandb = dict(
        name = name,
        project_name = "DINO_SLAM_Replica",
    ),
        
    render = dict(
        method = '2dgs',
        use_sa = True, 
        use_weight_norm = True, 
        enable_exposure = False,
        eps = 1e-6,
        depth_far = 1e2,
        depth_near = 1e-2,
    ),
    
    frontend = dict(
        num_tracking_iters = num_tracking_iter,
        num_mapping_iters = num_mapping_iters,
        converged_th = 5e-4,
        tau_k = 0.15, # The proportion of new observed scene, used for keyframe selection.
        tau_l = h * w * 1.5,  # Maximum number of Gaussians in the local map
        max_frames = localmap_max_frames, # Maximum number of frames in the local map
        vel_pose_init  = True, 
        enable_retracking = False, 
        additional_densify = False, 
        max_pending_submaps = 2, # the frontend waits when more submaps are queued for the backend
        congestion_tau_k_gain = 1.0, # tau_k *= (1 + gain * waiting submaps)
        congestion_max_frames_gain = 0.5, # max_frames *= (1 + gain * waiting submaps), capped at 2x
        congestion_start = 1, # number of waiting submaps tolerated before throttling
    ), 
    
    backend = dict(
        num_ba_iters = num_ba_iters,
        num_frame_saved = localmap_max_frames // 4,
        num_covis_submaps = 10,
        sleep_time = 0.1, # Prevent continuous backend operations from degrading frontend efficiency.
        
        mesh_vis = False,
        render_vis = False, 
        common_vis = True, 
        gs_densify = False,
        random_process = True, # random optimization when backend is not busy
        final_refinement = -1,
        offload_frames = True, # saved-frame RGB-D on GPU only for the current and covisible submaps
        mem_cleanup_interval = 50, # empty the CUDA cache every N mapping iterations
        min_ba_iters = num_ba_iters // 4, # lower bound of the task budget per submap under congestion
        congestion_gain = 1.0, # task budget = num_ba_iters / (1 + gain * waiting submaps)
        preempt_keep_ratio = 0.25, # fraction of queued tasks kept when a submap preempts
        preempt_pending_th = 2, # waiting submaps needed to preempt the task list
    ),
    
    densify=dict(
        use_edge_growth = False,
        densify_interval=20,
        method='splatam',
        sil_thres=0.6,
        edge_thres=0.4,
        dep_thres=0.1,
        opacity_cuil=0.05,
        scale_cuil=5e-4,
        scale_max=0.1,
        num_addpts= h * w,
        percent_dense=0.01,
        densify_grad_threshold=0.0002,
        extent=2,
    ),
    
    loss = dict(
        ignore_outliners = False,
        use_normal_loss = False,
        silmask_th = 0.90, # 
        tracking = dict(
            color=0.5,
            depth=1.0,
            normal = 0,
        ),
        mapping = dict(
            color=0.5,
            depth=1.0,
            normal = 0,
            dist = 0.1,
            feat = 1.0,
            feat_consist = 0.5,
            feat_ae = 1.0,
        ),
    ),
    # Scene Geometry Encoder / GeoDINO features 
    sge = dict(
        enable = True,
        dino_model = 'dinov2_vitb14', 
        dino_weights = None, 
        dino_input_height = 448,
        d_model = 128,
        num_heads = 4,
        feat_dim = 16, 
        head_hidden = 128,
        depth_scale = 5.0, 
        lr = 1e-3, 
        warmup_lr = 3e-3, 
        N_stage1 = num_ba_iters, 
        N_warmup = 50, 
        N_joint = -1, 
        loss_res_scale = 2, 
        mask_alpha_th = 0.5,
        frontend_init = True, 
        dino_cache_size = 16,
        dino_fp16 = True,
        feat_render_checkpoint = False, 
        feat_render_scale = 0.25, 
        sge_every_max = 4, 
        feat_grad_to_geometry = False, 
        feat_loss_ratio = 0.5, 
        grad_clip = 1.0, 
    ),
    data=dict(
        dataset_name="Replica",
        meshdir="/media/ziren/Data/ziren/data/Replica",
        basedir="/media/ziren/Data/ziren/data/Replica",
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
        gaussian_distribution = "anisotropic", # ["isotropic", "anisotropic"]
        training_args=dict(
            xyz_lr= 0.0001,
            feature_lr=0.0025,
            rgb_lr=0.0025,
            rotation_lr= 0.001,
            opacity_lr=0.05,
            scaling_lr=0.001,
            feat_lr=0.0025,
        ),

    ),
    cameras=dict(
        adam_betas = (0.7, 0.99),
        frontend_lr=dict(
            cam_rot_lr_init =  rot_lr_base,
            cam_rot_lr_final = rot_lr_base / 5,
            cam_rot_lr_max_step = num_tracking_iter, 
            cam_trans_lr_init =  trans_lr_base,
            cam_trans_lr_final = trans_lr_base / 5,
            cam_trans_lr_max_step = num_tracking_iter,
            exposure_lr_init = 0.005,
            exposure_lr_final = 0.0001,
            exposure_lr_max_step = 60,
        ),
        backend_lr=dict(
            cam_rot_lr_init = rot_lr_base / 4,
            cam_rot_lr_final = 0,
            cam_rot_lr_max_step = 2 * num_ba_iters,
            cam_trans_lr_init = trans_lr_base / 4,
            cam_trans_lr_final = 0,
            cam_trans_lr_max_step = 2 * num_ba_iters,
            exposure_lr_init = 0.005,
            exposure_lr_final = 0.0001,
            exposure_lr_max_step = 60,
        )
    ),
    viz=dict(
        viz_w=600,  
        viz_h=340,
        view_scale=2,
        mesh_every=5,
        gen_animation = False, 
        video_freq=30,
        cam_loc = ([[ 1.0, -0.0,  0.0,-3.08],
                    [-0.0, -1.0, -0.0, 1.14],
                    [ 0.0, -0.0, -1.0, 5.83],
                    [ 0.0,  0.0,  0.0, 1.0 ]]),
    ),
    eval=dict(
        save_renders = False, 
        eval_mesh = False, 
        save_mesh = False, 
        mesh_interval = 5, 
        voxel_size = 0.01,
    ),
)