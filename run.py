import argparse
import random
from time import time
from typing import Optional, Tuple
import sys
import os
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from thop import clever_format, profile
import cv2

import dataset.config as dataset_config
import dataset.evaluation as dataset_evaluation
import dataset.grasp as dataset_grasp
import dataset.graspnet_utils as dataset_graspnet_utils
import dataset.pc_dataset_tools as pc_dataset_tools
import dataset.utils as dataset_utils

from dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi,
                                get_thetas_widths)
from dataset.pc_dataset_tools import center2dtopc
from dataset.grasp import RectGraspGroup
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PatchMultiGraspNet


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve checkpoints robustly when launched outside RegionNormalizedGrasp.

    Priority:
      1) Absolute paths as-is
      2) Relative to current working directory
      3) Relative to the installed RegionNormalizedGrasp root (inferred from `dataset` package)
    """
    p = Path(checkpoint_path).expanduser()
    if p.is_absolute() and p.exists():
        return str(p)

    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return str(cwd_p)

    try:
        import dataset as _dataset_pkg  # top-level package installed from RegionNormalizedGrasp

        rng_root = Path(_dataset_pkg.__file__).resolve().parent.parent
        rng_p = (rng_root / p).resolve()
        if rng_p.exists():
            return str(rng_p)
    except Exception:
        pass

    # Fall back to the original value; torch.load will raise a helpful error.
    return checkpoint_path


def _load_pyrealsense2():
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyrealsense2 is required for the live RealSense demo. "
            "Install Intel RealSense SDK + python bindings."
        ) from e
    return rs


def _override_intrinsics(intrinsics_3x3: np.ndarray) -> None:
    """Override get_camera_intrinsic() across modules that imported it."""

    def _get_camera_intrinsic(_camera: str = "realsense"):
        return intrinsics_3x3

    dataset_config.get_camera_intrinsic = _get_camera_intrinsic
    dataset_evaluation.get_camera_intrinsic = _get_camera_intrinsic
    dataset_grasp.get_camera_intrinsic = _get_camera_intrinsic
    dataset_graspnet_utils.get_camera_intrinsic = _get_camera_intrinsic
    pc_dataset_tools.get_camera_intrinsic = _get_camera_intrinsic
    dataset_utils.get_camera_intrinsic = _get_camera_intrinsic


def capture_one_aligned_frame(
    *,
    serial: Optional[str],
    width: int,
    height: int,
    fps: int,
    warmup: int,
    timeout_ms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      - rgb (H, W, 3) uint8, RGB
      - depth_mm (H, W) float32, millimeters
      - intrinsics (3, 3) float32, for the aligned-to-color stream
      - depth_scale (float), meters per unit
    """
    rs = _load_pyrealsense2()

    pipeline = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)

    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(cfg)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        align = rs.align(rs.stream.color)

        for _ in range(max(0, warmup)):
            pipeline.wait_for_frames(timeout_ms)

        frames = pipeline.wait_for_frames(timeout_ms)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get both depth and color frames.")

        color = np.asanyarray(color_frame.get_data())  # RGB8
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)  # Z16 -> float
        depth_mm = depth_raw * depth_scale * 1000.0

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        intrinsics = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        return color, depth_mm.astype(np.float32), intrinsics, depth_scale
    finally:
        pipeline.stop()


class PointCloudHelper:
    def __init__(self, all_points_num: int, intrinsics_3x3: np.ndarray) -> None:
        self.all_points_num = all_points_num
        fx, fy = intrinsics_3x3[0, 0], intrinsics_3x3[1, 1]
        cx, cy = intrinsics_3x3[0, 2], intrinsics_3x3[1, 2]
        # NOTE: This repo uses a (W, H) tensor layout in demo.py, so we keep
        # the (1280, 720) meshgrid order consistent here.
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()

    def to_scene_points(self, rgbs: torch.Tensor, depths: torch.Tensor, include_rgb: bool = True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones((batch_size, self.all_points_num, feature_len), dtype=torch.float32).cuda()
        idxs = []
        masks = (depths > 0)
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        for i in range(batch_size):
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T
            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)), self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                idxs.append(cur_idxs)
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    def to_xyz_maps(self, depths: torch.Tensor):
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


def inference(
    view_points,
    rgbd,
    x,
    ori_rgb,
    ori_depth,
    *,
    anchornet,
    localnet,
    anchors,
    args,
    use_heatmap: bool = True,
    vis_heatmap: bool = True,
    vis_grasp: bool = True,
):
    eps = 1e-6
    with torch.no_grad():
        if use_heatmap:
            pred_2d, _ = anchornet(x)
            loc_map, cls_mask, theta_offset, height_offset, width_offset = anchor_output_process(
                *pred_2d, sigma=args.sigma
            )
            rect_gg = detect_2d_grasp(
                loc_map,
                cls_mask,
                theta_offset,
                height_offset,
                width_offset,
                ratio=args.ratio,
                anchor_k=args.anchor_k,
                anchor_w=args.hggd_anchor_w,
                anchor_z=args.anchor_z,
                mask_thre=args.heatmap_thres,
                center_num=args.center_num,
                grid_size=args.grid_size,
                grasp_nms=args.grid_size,
                reduce="max",
            )
            if rect_gg.size == 0:
                print("No 2d grasp found")
                return None
            if vis_heatmap:
                rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
                resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
                resized_rgb = np.array(resized_rgb.resize((args.input_w, args.input_h))) / 255.0
                depth_t = ori_depth.cpu().numpy().squeeze().T
                plt.subplot(131)
                plt.imshow(rgb_t)
                plt.subplot(132)
                plt.imshow(depth_t)
                plt.subplot(133)
                plt.imshow(loc_map.squeeze().T, cmap="jet")
                plt.tight_layout()
                plt.show()
        else:
            raise RuntimeError("Only heatmap-based center selection is supported in demo_realsense_live.py")

        valid_local_centers, _ = center2dtopc(
            [rect_gg],
            args.center_num,
            ori_depth,
            (args.input_w, args.input_h),
            append_random_center=False,
            is_training=False,
        )

        _, w, h = rgbd.shape
        t = torch.linspace(0, 1, args.patch_size, device="cuda", dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(t, t)
        grid_idxs = torch.stack([grid_x, grid_y], -1) - 0.5
        ratio = w / args.input_w
        centers_t = ratio * torch.from_numpy(rect_gg.centers).cuda()
        grid_idxs = grid_idxs[None].expand(len(centers_t), -1, -1, -1)

        intrinsics = dataset_config.get_camera_intrinsic()
        fx = intrinsics[0, 0]
        radius = torch.full((len(centers_t),), 0.10, device="cuda")
        radius *= 2 * fx / valid_local_centers[0][:, 2]
        radius *= args.anchor_w / 60.0

        grid_idxs = grid_idxs * radius[:, None, None, None]
        grid_idxs = grid_idxs + torch.flip(centers_t[:, None, None], [-1])
        grid_idxs = grid_idxs / torch.FloatTensor([(h - 1), (w - 1)]).cuda() * 2 - 1

        local_patches = F.grid_sample(
            rgbd[None].expand(len(centers_t), -1, -1, -1),
            grid_idxs,
            mode="nearest",
            align_corners=False,
        )
        local_patches = local_patches.permute(0, 3, 2, 1).contiguous()

        mask = (local_patches[..., -1:] > 0)
        patch_centers = valid_local_centers[0][:, None, None].expand(
            -1, args.patch_size, args.patch_size, -1
        )
        local_patches[..., 3:] -= mask * patch_centers
        local_patches[..., 3:] /= args.anchor_w / 1e3

        _, pred, offset, theta_cls, theta_offset, width_reg = localnet(local_patches)
        theta_cls = theta_cls.sigmoid().clip(eps, 1 - eps).detach().cpu().numpy().squeeze()
        theta_offset = theta_offset.clip(-0.5, 0.5).detach().cpu().numpy().squeeze()
        width_reg = width_reg.detach().cpu().numpy().squeeze()

        thetas, widths_6d = get_thetas_widths(
            theta_cls,
            theta_offset,
            width_reg,
            anchor_w=args.anchor_w,
            rotation_num=1,
        )

        pred_grasp, pred_6d_gg = detect_6d_grasp_multi(
            thetas,
            widths_6d,
            pred,
            offset,
            valid_local_centers,
            anchors,
            alpha=args.alpha * args.anchor_w / 60.0,
            k=args.local_k,
        )

        pred_gg, valid_mask = collision_detect(view_points[..., :3].squeeze(), pred_6d_gg, mode="graspnet")
        pred_grasp = pred_grasp[valid_mask]

        mask = (pred_gg.scores > 0.5)
        pred_gg = pred_gg[mask]
        pred_gg = pred_gg.nms()[:50]

        if vis_grasp:
            print("pred grasp num ==", len(pred_gg))
            grasp_geo = pred_gg.to_open3d_geometry_list(scale=args.anchor_w / 60)
            points = view_points[..., :3].cpu().numpy().squeeze()
            colors = view_points[..., 3:6].cpu().numpy().squeeze()
            vispc = o3d.geometry.PointCloud()
            vispc.points = o3d.utility.Vector3dVector(points)
            vispc.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([vispc] + grasp_geo)
        return pred_gg


def main():
    parser = argparse.ArgumentParser()

    # Model/checkpoint (defaults match demo.sh)
    parser.add_argument("--checkpoint-path", default="./checkpoints/realsense")
    parser.add_argument("--center-num", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=64, help="local patch grid size")

    # RealSense capture
    parser.add_argument("--serial", default=None, help="RealSense device serial (optional)")
    parser.add_argument("--rs-width", type=int, default=1280)
    parser.add_argument("--rs-height", type=int, default=720)
    parser.add_argument("--rs-fps", type=int, default=30)
    parser.add_argument("--rs-warmup", type=int, default=15)
    parser.add_argument("--rs-timeout-ms", type=int, default=5000)

    # paras from hggd (some are useless but kept for convenience)
    parser.add_argument("--input-h", type=int, default=360)
    parser.add_argument("--input-w", type=int, default=640)
    parser.add_argument("--sigma", type=int, default=10)
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--anchor-k", type=int, default=6)
    parser.add_argument("--hggd-anchor-w", type=float, default=75.0)
    parser.add_argument("--anchor-z", type=float, default=20.0)
    parser.add_argument("--grid-size", type=int, default=12)

    # pc
    parser.add_argument("--all-points-num", type=int, default=25600)

    # patch / net
    parser.add_argument("--alpha", type=float, default=0.02, help="grasp center crop range")
    parser.add_argument("--anchor-w", type=float, default=60.0)
    parser.add_argument("--anchor-num", type=int, default=7)

    # grasp detection
    parser.add_argument("--heatmap-thres", type=float, default=0.01)
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--local-thres", type=float, default=0.01)
    parser.add_argument("--rotation-num", type=int, default=1)

    # visualization
    parser.add_argument("--no-vis-heatmap", action="store_true")
    parser.add_argument("--no-vis-grasp", action="store_true")

    # others
    parser.add_argument("--random-seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()
    args.checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)

    # torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError("CUDA not available")

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Capture one aligned RGB-D frame
    rgb_u8, depth_mm, intrinsics_3x3, depth_scale = capture_one_aligned_frame(
        serial=args.serial,
        width=args.rs_width,
        height=args.rs_height,
        fps=args.rs_fps,
        warmup=args.rs_warmup,
        timeout_ms=args.rs_timeout_ms,
    )

    if (args.rs_width, args.rs_height) != (1280, 720):
        raise RuntimeError(
            f"This demo currently expects 1280x720 input for correct point-cloud mapping, "
            f"but got {args.rs_width}x{args.rs_height}. Please use --rs-width 1280 --rs-height 720."
        )

    # Override intrinsics across the codebase for this run
    _override_intrinsics(intrinsics_3x3)
    print("Using live RealSense intrinsics:")
    print(intrinsics_3x3)
    print(f"RealSense depth_scale (m/unit) = {depth_scale}")

    # Init the model
    anchornet = AnchorGraspNet(in_dim=4, ratio=args.ratio, anchor_k=args.anchor_k)
    localnet = PatchMultiGraspNet(args.anchor_num**2, theta_k_cls=6, feat_dim=args.embed_dim, anchor_w=args.anchor_w)
    x = torch.randn((48, 64, 64, 6), device="cuda")
    params_heat = sum(p.numel() for p in anchornet.parameters() if p.requires_grad)
    print(f"Heatmap Model params == {params_heat}")
    macs, params = clever_format(profile(localnet.cuda(), inputs=(x,)), "%.3f")
    print(f"RNGNet macs == {macs}  params == {params}")

    anchornet = anchornet.cuda().eval()
    localnet = localnet.cuda().eval()

    # Load checkpoint
    check_point = torch.load(args.checkpoint_path)
    anchornet.load_state_dict(check_point["anchor"])
    localnet.load_state_dict(check_point["local"])

    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {"gamma": basic_anchors, "beta": basic_anchors}
    anchors["gamma"] = check_point["gamma"]
    anchors["beta"] = check_point["beta"]
    print("Using saved anchors")
    print("-> loaded checkpoint %s " % (args.checkpoint_path))

    # Convert captured frame to the tensor layout used by demo.py
    ori_rgb_np = rgb_u8.astype(np.float32) / 255.0
    ori_depth_np = np.clip(depth_mm, 0, 1000).astype(np.float32)

    ori_rgb = torch.from_numpy(ori_rgb_np).permute(2, 1, 0)[None].to(device="cuda", dtype=torch.float32)
    ori_depth = torch.from_numpy(ori_depth_np).T[None].to(device="cuda", dtype=torch.float32)

    pc_helper = PointCloudHelper(all_points_num=args.all_points_num, intrinsics_3x3=intrinsics_3x3)

    view_points, idxs, masks = pc_helper.to_scene_points(ori_rgb, ori_depth, include_rgb=True)
    view_points = view_points.squeeze()
    xyzs = pc_helper.to_xyz_maps(ori_depth)
    rgbd = torch.cat([ori_rgb.squeeze(), xyzs.squeeze()], 0)

    # pre-process (keep demo.py behavior, including its (W,H) layout)
    rgb = F.interpolate(ori_rgb, (args.input_w, args.input_h))
    depth = F.interpolate(ori_depth[None], (args.input_w, args.input_h))[0]
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)
    x = torch.concat([depth[None], rgb], 1).to(device="cuda", dtype=torch.float32)

    # inference
    pred_gg = inference(
        view_points,
        rgbd,
        x,
        ori_rgb,
        ori_depth,
        anchornet=anchornet,
        localnet=localnet,
        anchors=anchors,
        args=args,
        use_heatmap=True,
        vis_heatmap=not args.no_vis_heatmap,
        vis_grasp=not args.no_vis_grasp,
    )

    # quick time test (optional, keep short)
    start = time()
    T = 10
    for _ in range(T):
        _ = inference(
            view_points,
            rgbd,
            x,
            ori_rgb,
            ori_depth,
            anchornet=anchornet,
            localnet=localnet,
            anchors=anchors,
            args=args,
            use_heatmap=True,
            vis_heatmap=False,
            vis_grasp=False,
        )
        torch.cuda.synchronize()
    print("avg time ==", (time() - start) / T * 1e3, "ms")


if __name__ == "__main__":
    main()

