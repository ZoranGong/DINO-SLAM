import os

# scenes = ["freiburg1_desk", "freiburg1_desk2", "freiburg1_room", "freiburg2_xyz", "freiburg3_long_office_household"]
scenes = ["freiburg1_desk"]

seed = int(os.environ["SEED"]) if "SEED" in os.environ else 0
# scene_name = scenes[int(os.environ["SCENE_NUM"])] if "SCENE_NUM" in os.environ else 0
scene_name = scenes[0]
exp = int(os.environ["EXP"]) if "EXP" in os.environ else 0
name = 'TUM_RGBD'

# ablation study
wo_keyframe = (exp == 5) # E
wo_localmap = (exp == 6) # F
wo_random_optimization = (exp == 7) # G
wo_final_refinement = (exp == 8) # H

h = 480
w = 640
trans_lr_base = 0.004
rot_lr_base = 0.001
num_tracking_iter = 120
num_ba_iters = 120
localmap_max_frames = 40

config = dict(
    vis_base_dir = f'output/{name}_exp{exp}_seed{seed}/{scene_name}',
    seed = seed,
    primary_device='cuda',
    use_wandb = False,
    wandb = dict(
        name = name,
        project_name = "GauS_SLAM_tumrgbd",
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
        num_mapping_iters = localmap_max_frames,
        converged_th = -1,
        tau_k = 0.01  if not wo_keyframe else -1, # The proportion of new observed scene, used for keyframe selection.
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
        num_covis_submaps = 30,
        sleep_time = 0.1, # Prevent continuous backend operations from degrading frontend efficiency.
        
        mesh_vis = False,
        render_vis = False, 
        common_vis = True, 
        gs_densify = False,
        random_process = not wo_random_optimization, # random optimization when backend is not busy
        final_refinement = -1  if not wo_final_refinement else 0,
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
        opacity_cuil=0.05,
        scale_cuil=5e-4,
        scale_max=0.1,
        num_addpts=480 * 640,
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
            dist = 0,
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
    
    # Scene Geometry Encoder / GeoDINO features (slam/SGE.py)
    sge = dict(
        enable = True,
        dino_model = 'dinov2_vitb14', # torch.hub model of facebookresearch/dinov2
        dino_weights = None, # optional local checkpoint
        dino_input_height = 448,
        d_model = 128,
        num_heads = 4,
        feat_dim = 16, # per-Gaussian feature dimension
        head_hidden = 128,
        depth_scale = 5.0, # depth normalization (m)
        lr = 1e-3, # SGE + MLPs learning rate (joint stage)
        warmup_lr = 3e-3, # SGE + MLPs learning rate (warm-up)
        N_stage1 = num_ba_iters, # backend mapping iterations before SGE starts
        N_warmup = 50, # SGE-only iterations, Gaussians frozen
        N_joint = -1, # joint iterations, -1 = until the end
        loss_res_scale = 2, # feature losses at DINO patch grid x this scale
        mask_alpha_th = 0.5,
        frontend_init = True, # initialize new Gaussians' features with a frozen SGE copy in the frontend
        dino_cache_size = 16,
        dino_fp16 = True,
        feat_render_checkpoint = False, # checkpoint the feature passes to save VRAM
        feat_render_scale = 0.25, # raster size of the feature passes relative to the image
        sge_every_max = 4, # under congestion the SGE losses are used every (1 + waiting submaps) iterations at most
        feat_grad_to_geometry = False, # feature losses update the geometry as well
        feat_loss_ratio = 0.5, # feature loss is scaled to at most this fraction of the RGB-D loss
        grad_clip = 1.0, # gradient norm clip of SGE + MLPs
    ),
    data=dict(
        dataset_name='tum',
        basedir="/media/ziren/Data/ziren/data/TUM",
        gradslam_data_cfg=f"./configs/data/TUM/{scene_name}.yaml",
        sequence=f"rgbd_dataset_{scene_name}",
        desired_image_height=480,
        desired_image_width=640,
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
            cam_rot_lr_init = rot_lr_base, # 0.0004,  # / 100
            cam_rot_lr_final = rot_lr_base / 10,
            cam_rot_lr_max_step = num_tracking_iter,
            cam_trans_lr_init = trans_lr_base, # 0.002, # / 100
            cam_trans_lr_final = trans_lr_base / 10,
            cam_trans_lr_max_step = num_tracking_iter,
            exposure_lr_init = 0.0001,
            exposure_lr_final = 0.00001,
            exposure_lr_max_step = 100,
        ),
        backend_lr=dict(
            cam_rot_lr_init = rot_lr_base / 2, # 0.0004,  # / 100
            cam_rot_lr_final = rot_lr_base / 10,
            cam_rot_lr_max_step = 2*num_ba_iters,
            cam_trans_lr_init = trans_lr_base / 2, # 0.002, # / 100
            cam_trans_lr_final = trans_lr_base / 10,
            cam_trans_lr_max_step = 2*num_ba_iters,
            exposure_lr_init = 0.0001,
            exposure_lr_final = 0.00001,
            exposure_lr_max_step = 100,
        )
    ),
    viz=dict(
        viz_w=600,  
        viz_h=340,
        view_scale=2,
        mesh_every=5,
        gen_animation = False, 
        video_freq=30,
    ),
    eval=dict(
        save_renders = False, 
        eval_mesh = False, 
        save_mesh = False, 
        mesh_interval = 5, 
        voxel_size = 0.01,
    ),
)