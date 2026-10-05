
from open3d_ui import Vis_Render, Vis_Mesh
from scene import Gaussians, LocalMap, Localmaps, Frame
import torch 
from render import Renderer_mapping, Renderer_tracking, Renderer_view
from tqdm import tqdm
from slam.Loss import get_loss, weight_losses
import random
from utils.common_utils import build_quaternion, build_rotation
from slam.Densify import prune_gaussians
from slam.SGE import SGESystem, sge_enabled
import torch.multiprocessing as mp
import os
import gc
import matplotlib.pyplot as plt
from queue import Queue

import evo
import copy
from evo.core import metrics
from evo.core.trajectory import PosePath3D
import matplotlib.pyplot as plt
import evo.tools.plot
import time 
import wandb

class Backend():
    
    def __init__(self, config, wandb_run):
        self.config = config
        self.map = Gaussians(config)
        self.cur_lmid = -1
        self.local_maps = Localmaps(config)
        self.fix_vel = torch.eye(4).cuda().float()
        self.num_ba_iters = config['backend']['num_ba_iters']
        self.num_covis_submaps = config['backend']['num_covis_submaps']
        self.enable_random_process = config['backend']['random_process']
        self.mesh_vis = config['backend']['mesh_vis']
        self.render_vis = config['backend']['render_vis']
        self.enable_exposure = config['render']['enable_exposure']
        self.gs_densify = config['backend']['gs_densify']
        self.use_wandb = config['use_wandb']
        self.wandb_run = wandb_run
        
        if self.mesh_vis:
            self.vis_mesh = Vis_Mesh(config)
        if self.render_vis:
            self.vis_render = Vis_Render(config, os.path.join(config['vis_base_dir'], "backend"))

        self.task_queue = Queue()
        self.random_process_localmaps_idxs = []
        self.ape_rec = []
        self.totalpts_rec = []
        self.mapping_iter = 0

        # SGE (Group A), backend only
        self.use_sge = sge_enabled(config)
        self._sge = None
        sge_cfg = config.get('sge', {})
        self.sge_every_max = sge_cfg.get('sge_every_max', 4)
        self.feat_loss_ratio = sge_cfg.get('feat_loss_ratio', 0.5)
        self.sge_grad_clip = sge_cfg.get('grad_clip', 1.0)

        # saved-frame RGB-D stays on the GPU only for the active (current + covisible) submaps
        self.offload_frames = config['backend'].get('offload_frames', True)
        self.mem_cleanup_interval = config['backend'].get('mem_cleanup_interval', 50)
        self.active_lm_idxs = set()

        # congestion control: task budget and preemption depend on the number of waiting submaps
        self.min_ba_iters = config['backend'].get('min_ba_iters', max(4, self.num_ba_iters // 4))
        self.congestion_gain = config['backend'].get('congestion_gain', 1.0)
        self.preempt_keep_ratio = config['backend'].get('preempt_keep_ratio', 0.25)
        self.preempt_pending_th = config['backend'].get('preempt_pending_th', 2)
        self.congestion_level = 0

    def pending_submaps(self):
        return 0

    def congestion(self):
        return max(self.congestion_level, self.pending_submaps())

    def ba_budget(self):
        n = int(self.num_ba_iters / (1.0 + self.congestion_gain * self.congestion()))
        return max(self.min_ba_iters, n)

    def sge_every(self):
        return min(1 + self.congestion(), self.sge_every_max)

    def shrink_task_queue(self, keep_ratio):
        tasks = []
        while not self.task_queue.empty():
            tasks.append(self.task_queue.get())
        for t in tasks[:int(len(tasks) * keep_ratio)]:
            self.task_queue.put(t)

    def _select_frame(self, localmap_idx):
        select_lm: LocalMap = self.local_maps[localmap_idx]
        select_fid = random.choice(select_lm.saved_idxs)
        select_f: Frame = select_lm.frames[select_fid]
        select_f.to_device('cuda')
        return select_lm, select_fid, select_f

    def _after_task(self, localmap_idx):
        if self.offload_frames and localmap_idx not in self.active_lm_idxs:
            self.local_maps[localmap_idx].offload('cpu')
        self._mem_cleanup()

    def _set_active_submaps(self, idxs):
        self.active_lm_idxs = set(idxs)
        if self.offload_frames:
            for i, lm in enumerate(self.local_maps):
                lm.offload('cuda' if i in self.active_lm_idxs else 'cpu')

    def _mem_cleanup(self, force=False):
        if force or (self.mem_cleanup_interval > 0 and self.mapping_iter % self.mem_cleanup_interval == 0):
            gc.collect()
            torch.cuda.empty_cache()

    @property
    def sge(self) -> SGESystem:
        if self.use_sge and self._sge is None:
            self._sge = SGESystem(self.config, trainable=True)
        return self._sge

    def save_sge(self, path):
        if self.use_sge and self._sge is not None:
            self._sge.save(path)
    
    def re_tracking(self, localmap_idx):
        """
        tracking from the first frame of local map when tracking lost
        """
        num_tracking_iters = self.config['frontend']['num_tracking_iters'] * 2
        for iter in range(num_tracking_iters):
            select_lm, select_fid, select_f = self._select_frame(localmap_idx)
            select_lm.transform.optimizer.zero_grad(set_to_none = True)
            fix_w2c = select_lm.get_frame_w2c(select_fid) # select_f.get_w2c.detach() @ select_kf.get_w2c
            fix_exposure = select_lm.get_frame_exposure(select_fid)             
            render_pkg = Renderer_tracking(self.config, 
                                            self.map, 
                                            select_f, 
                                            fix_w2c=fix_w2c, 
                                            fix_exposure=fix_exposure)
            loss = get_loss(self.config, 
                            render_pkg, 
                            select_f, 
                            tracking=True)
            loss.backward()
            
            with torch.no_grad():
                select_lm.transform.optimizer.step()
                select_lm.transform.update_learning_rate()
        self._after_task(localmap_idx)

    def tracking(self, localmap_idx):
        select_lm, select_fid, select_f = self._select_frame(localmap_idx)
        select_lm.transform.optimizer.zero_grad(set_to_none = True)
        fix_w2c = select_lm.get_frame_w2c(select_fid)
        fix_exposure = select_lm.get_frame_exposure(select_fid)             
        
        render_pkg = Renderer_tracking(self.config, 
                                        self.map, 
                                        select_f, 
                                        fix_w2c=fix_w2c, 
                                        fix_exposure=fix_exposure)
        loss = get_loss(self.config, render_pkg, select_f, tracking=True)
        loss.backward()
        
        with torch.no_grad():
            select_lm.transform.optimizer.step()
            select_lm.transform.update_learning_rate()
          
    def mapping(self, localmap_idx):
        """
        stage1: original mapping, warmup: SGE only (Gaussians frozen), joint: Gaussians + SGE
        """
        select_lm, select_fid, select_f = self._select_frame(localmap_idx)
        self.map.optimizer.zero_grad(set_to_none=True)
        if self.enable_exposure:
            select_lm.exposure.optimizer.zero_grad(set_to_none=True)

        stage = self.sge.stage(self.mapping_iter) if self.use_sge else SGESystem.STAGE1
        use_sge = stage == SGESystem.WARMUP or (stage == SGESystem.JOINT and self.mapping_iter % self.sge_every() == 0)
        if use_sge:
            self.sge.zero_grad()

        fix_w2c = select_lm.get_frame_w2c(select_fid) # select_f.get_w2c.detach() @ select_kf.get_w2c
        fix_exposure = select_lm.get_frame_exposure(select_fid)
        render_pkg = Renderer_mapping(self.config, self.map, select_f,
                                    fix_w2c=fix_w2c,
                                    fix_exposure=fix_exposure,
                                    render_feat=use_sge)
        weights = self.config['loss']['mapping']
        feat_scale = 1.0
        if stage == SGESystem.WARMUP:
            extra_losses = self.sge.losses((select_lm.lmid, select_f.time_idx), select_f.gt_color, select_f.gt_depth, render_pkg)
            loss = weight_losses(extra_losses, weights)
        else:
            loss = get_loss(self.config, render_pkg, select_f)
            if use_sge:
                extra_losses = self.sge.losses((select_lm.lmid, select_f.time_idx), select_f.gt_color, select_f.gt_depth, render_pkg)
                feat_loss = weight_losses(extra_losses, weights)
                # the feature loss contributes at most feat_loss_ratio of the RGB-D loss
                if self.feat_loss_ratio > 0 and feat_loss.requires_grad:
                    feat_scale = min(1.0, (self.feat_loss_ratio * loss.detach() / feat_loss.detach().clamp_min(1e-8)).item())
                loss = loss + feat_scale * feat_loss
        if loss.requires_grad:
            loss.backward()
        with torch.no_grad():
            if use_sge and self.sge_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.sge.trainable_parameters(), self.sge_grad_clip)
            if stage == SGESystem.WARMUP:
                self.map.optimizer.zero_grad(set_to_none=True)
                if self.enable_exposure:
                    select_lm.exposure.optimizer.zero_grad(set_to_none=True)
                self.sge.optimizer_step()
            else:
                if self.gs_densify:
                    self.map.add_densification_stats(render_pkg)

                self.map.optimizer.step()
                select_lm.mapping_times += 1
                if self.enable_exposure and select_lm.mapping_times > 120:
                    select_lm.exposure.optimizer.step()
                    select_lm.exposure.update_learning_rate()
                if stage == SGESystem.JOINT:
                    self.sge.optimizer_step()

            if use_sge:
                self.sge.step_counter()
                if self.use_wandb and self.sge.sge_iter % 20 == 0:
                    self.wandb_run.log({**{f"sge/{k}": float(v) for k, v in extra_losses.items()}, "sge/feat_scale": feat_scale})

            self.mapping_iter += 1
            if self.gs_densify and stage != SGESystem.WARMUP and (self.mapping_iter + 1) % self.config['densify']['densify_interval'] == 0:
                self.map.densify_and_prune()
    
    def ba(self, localmap_idx):
        select_lm, select_fid, select_f = self._select_frame(localmap_idx)
        self.map.optimizer.zero_grad(set_to_none=True)
        select_lm.transform.optimizer.zero_grad(set_to_none=True)
        if self.enable_exposure:
            select_lm.exposure.optimizer.zero_grad(set_to_none=True)
        
        fix_w2c = select_lm.get_frame_w2c(select_fid)
        fix_exposure = select_lm.get_frame_exposure(select_fid)   
        
        render_pkg = Renderer_mapping(self.config, self.map, select_f, 
                                        fix_w2c=fix_w2c,
                                        fix_exposure=fix_exposure)
        loss = get_loss(self.config, render_pkg, select_f)
        loss.backward()
        with torch.no_grad():
            self.map.add_densification_stats(render_pkg)
            
            self.map.optimizer.step()
            select_lm.transform.optimizer.step()
            select_lm.transform.update_learning_rate()
            if self.enable_exposure:
                select_lm.exposure.optimizer.step()
                select_lm.exposure.update_learning_rate()
            
    @torch.no_grad()
    def transfer_map_params(self, map_params, transfer):
        map_params['xyz'] = (transfer[:3, :3] @ map_params['xyz'].T  + transfer[:3, 3:]).T
        map_params['rotation'] = build_quaternion(torch.matmul(transfer[None, :3, :3], build_rotation(map_params['rotation'])))
        return map_params
    
    def final_refine(self):
        iters = self.config['backend']['final_refinement']
        # default iters = num_frames
        if iters == -1:
            iters = self.local_maps[-1].frames[-1].time_idx
        for i in tqdm(range(iters), total=iters, desc="final_refine"):
            localmap_idx = random.choice(list(range(len(self.local_maps))))
            self.mapping(localmap_idx)
            self._after_task(localmap_idx)
            if self.gs_densify and self.mapping_iter > 100 and self.mapping_iter % 50 == 0:
                self.map.densify_and_prune()
                
    def process(self):
        """
        Processes task queue commands
        When queue is empty, choose a random local map for mapping if random_process is enabled
        """
        if not self.task_queue.empty():
            cmd = self.task_queue.get()
            if cmd[0] == "prune":
                prune_gaussians(self.config, self.map)
            elif cmd[0] == "tracking":
                localmap_idx = cmd[1]
                self.tracking(localmap_idx)
                self._after_task(localmap_idx)
            elif cmd[0] == "mapping":
                localmap_idx = cmd[1]
                self.mapping(localmap_idx)
                self._after_task(localmap_idx)
            elif cmd[0] == "ba":
                localmap_idx = cmd[1]
                self.ba(localmap_idx)
                self._after_task(localmap_idx)
        elif self.enable_random_process and len(self.local_maps) > 0:
            localmap_idx = random.choice(list(range(len(self.local_maps))))
            self.task_queue.put(("mapping", localmap_idx))
        
    def process_localmap(self, lm: LocalMap, multi_process=True):
        """
        Main processing pipeline of the Backend
        Backend process one local map at a time
        """
        self.local_maps.add_localmap(lm)
        self.cur_lmid = self.cur_lmid + 1
        map_params = lm.local_map_params
        lm.local_map_params = None 
        if self.cur_lmid == 0:
            initial_w2kf = torch.eye(4, dtype=torch.float32, device='cuda')
        else:
            last_lm: LocalMap = self.local_maps[self.cur_lmid - 1]
            initial_w2kf = last_lm.get_frame_w2c(-1).detach()
        
        if not lm.tracking_ok:
            print("backend global tracking for local tracking lost")
            lm.start_optimizer(initial_w2kf, self.config['cameras']['frontend_lr'])
            self.re_tracking(self.cur_lmid)
            initial_w2kf = lm.get_w2c
            
        lm.start_optimizer(initial_w2kf, self.config['cameras']['backend_lr'])
        
        num_ba_iters = self.ba_budget()
        self.congestion_level = 0
        if self.use_sge:
            self.sge.prefetch([((lm.lmid, lm.frames[i].time_idx), lm.frames[i].gt_color) for i in lm.saved_idxs])

        # first KeyFrame
        if self.cur_lmid == 0:
            self.map.create_params(map_params)
            self._set_active_submaps({0})
            for i in range(num_ba_iters):
                self.task_queue.put(("mapping", 0))
        else:
            map_params = self.transfer_map_params(map_params, torch.linalg.inv(lm.get_w2c.detach()) @ lm.ref2f0)
            map_params['opacity'] = torch.min(map_params['opacity'], self.map.inverse_opacity_activation(0.01 * torch.ones_like(map_params['opacity'])))
            self.map.add_params(map_params)
            self.random_process_localmaps_idxs = self.local_maps.query_covisable(self.cur_lmid, self.num_covis_submaps)
            self._set_active_submaps(set(self.random_process_localmaps_idxs) | {self.cur_lmid})
            # Merge localmap
            merge_set = self.random_process_localmaps_idxs[:max(1, self.num_covis_submaps // 2)]
            for i in range(num_ba_iters):
                self.task_queue.put(("mapping", random.choice(merge_set)))
            self.task_queue.put(("prune", None))
            
            # Performing tracking and mapping alternately
            for i in range(num_ba_iters // 2):
                self.task_queue.put(("tracking", self.cur_lmid))
            for i in range(num_ba_iters):
                self.task_queue.put(("mapping", random.choice(self.random_process_localmaps_idxs)))
            for i in range(num_ba_iters):
                self.task_queue.put(("tracking", random.choice(self.random_process_localmaps_idxs)))
            
        del map_params
        self._mem_cleanup(force=True)

        # sync
        if not multi_process:
             while not self.task_queue.empty():
                 self.process()

        if self.use_wandb:
            self.wandb_run.log({"cur_lmid":self.cur_lmid, "Backend_numpts":self.map.size})         

    def update_vis(self, ):
        if self.mesh_vis and len(self.local_maps) > 0:  
            lm:LocalMap = self.local_maps[-1]
            
            for i, frame in enumerate(lm.frames[:-1]):
                frame: Frame 
                if frame.time_idx % 5 == 0:
                    fix_w2c = lm.get_frame_w2c(i)
                    render_pkg = Renderer_view(self.config, self.map, fix_w2c)
                    render_color = torch.clamp(render_pkg['render_color'], 0.0, 1.0)
                    render_depth = render_pkg['render_depth']
                    self.vis_mesh.update_frame(render_color, render_depth, fix_w2c, frame.gt_w2c, frame.time_idx)
        
        if self.render_vis and len(self.local_maps) > 0:
            lm:LocalMap = self.local_maps[-1]
            
            for i, frame in enumerate(lm.frames[:-1]):
                frame: Frame 
                self.vis_render.update_frame(self.map, lm.get_frame_w2c(i), frame.frame_type, frame.time_idx)

    
    def update_common_visualization(self, ):
        if self.config['backend']['common_vis']:
            w2cs = self.local_maps.get_w2cs()
            gt_w2cs = self.local_maps.get_gt_w2cs()
            notbad = [id for id in range(len(gt_w2cs)) if not (torch.isnan(gt_w2cs[id]).any() or torch.isinf(gt_w2cs[id]).any())]
            w2cs = [w2cs[id] for id in notbad] # w2cs[notbad]
            gt_w2cs = [gt_w2cs[id] for id in notbad] #  gt_w2cs[notbad]
            vis_base_dir = self.config['vis_base_dir']
            os.makedirs(vis_base_dir, exist_ok=True)
            if len(w2cs) > 3:
                pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in w2cs]
                gt_pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in gt_w2cs]
                traj_ref = PosePath3D(poses_se3=gt_pose_w2cs)
                traj_est = PosePath3D(poses_se3=pose_w2cs)
                traj_est_aligned = copy.deepcopy(traj_est)
                traj_est_aligned.align(traj_ref, correct_scale=False,)
                pose_relation = metrics.PoseRelation.translation_part
                data = (traj_ref, traj_est_aligned)
                ape_metric = metrics.APE(pose_relation)
                ape_metric.process_data(data)
                ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
                ape_stats = ape_metric.get_all_statistics()
                
                fig = plt.figure()
                plot_mode = evo.tools.plot.PlotMode.xy
                ax = evo.tools.plot.prepare_axis(fig, plot_mode)
                ax.set_title(f"ATE RMSE: {ape_stat}")
                self.ape_rec.append(ape_stat)
                evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
                evo.tools.plot.traj_colormap(
                    ax,
                    traj_est_aligned,
                    ape_metric.error,
                    plot_mode,
                    min_map=ape_stats["min"],
                    max_map=ape_stats["max"],
                )            
                ax.legend()
                plt.savefig(os.path.join(f"{vis_base_dir}", "evo_2dplot.png"), dpi=90)
                plt.close()

                if self.use_wandb:
                    self.wandb_run.log({"cur_lmid":self.cur_lmid, "APE":ape_stat})
                    evo_2dplot_figname = str(os.path.join(f"{vis_base_dir}", "evo_2dplot.png"))
                    self.wandb_run.log({"evo_2dplot": [wandb.Image(evo_2dplot_figname)]})
            
            self.totalpts_rec.append(self.map.size)
            plt.plot(range(len(self.totalpts_rec)), self.totalpts_rec)
            plt.savefig(f"{vis_base_dir}/backend_numpts.png")
            plt.close()
            
            differ_rec = []
            for j in range(len(w2cs)):
                differ_rec.append(torch.norm((w2cs[j] @ torch.linalg.inv(gt_w2cs[j]))[:3, 3]).item())
            plt.plot(range(len(differ_rec)), differ_rec)
            plt.savefig(f"{vis_base_dir}/trackloss.png")
            plt.close()
            
            plt.plot(range(len(self.ape_rec)), self.ape_rec)
            plt.savefig(f"{vis_base_dir}/ape.png")
            plt.close()


class mp_Backend(Backend):
    def __init__(self, config, to_backend: mp.Queue, event, wandb_run):
        self.to_backend = to_backend
        self.event = event
        self.sleep_time = config['backend']['sleep_time']
        super().__init__(config, wandb_run)

    def pending_submaps(self):
        return self.to_backend.qsize()

    def run(self, ):
        should_finish = 0
        while True:
            
            if should_finish and self.task_queue.empty():
                print("Backend finished !!!")
                self.event.set()
                
                if self.mesh_vis:
                    self.vis_mesh.destroy()
                if self.render_vis:
                    self.vis_render.destory()
                
                break 
            # a waiting submap is admitted when the task list is empty, or right away
            # (after shrinking the task list) once preempt_pending_th submaps are waiting
            pending = self.pending_submaps()
            has_submap = pending > 0
            if has_submap and not self.task_queue.empty():
                if pending >= self.preempt_pending_th and not should_finish:
                    self.shrink_task_queue(self.preempt_keep_ratio)
                else:
                    has_submap = False
            if has_submap:
                self.update_vis()
                self.congestion_level = pending - 1

                lm_asap = self.to_backend.get()
                lm = copy.deepcopy(lm_asap)
                del lm_asap
                if isinstance(lm, str):
                    if lm == "finish":
                        should_finish = 1
                        continue 
                self.process_localmap(lm)
                self.update_common_visualization()
                
            self.process()
            if self.task_queue.empty():
                time.sleep(self.sleep_time)
    
    