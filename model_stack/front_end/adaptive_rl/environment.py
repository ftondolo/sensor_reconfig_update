"""RL Environment for adaptive resolution/framerate control.

The environment simulates multimodal streaming: at each step the agent picks
resolution and framerate tiers for RGB and Thermal. The environment runs YOLO
at the chosen settings, compares detection quality against a full-resolution
baseline, and returns a reward that balances accuracy vs compute cost.

State:  [rgb_contrib, therm_contrib, radar_contrib, scenario_onehot(3),
         avg_conf, num_dets_norm, current_rgb_res_tier, current_therm_res_tier,
         current_rgb_fps_tier, current_therm_fps_tier]   → 13 dims

Action: Discrete, 16 values = 4 rgb_res × 4 therm_res
        (framerate is derived: lower res → proportionally lower fps)

Reward: α * detection_iou_with_baseline  -  β * compute_cost  -  γ * missed_penalty
"""

import os, json, random, time
import numpy as np
import cv2
import torch
import torchvision

# NMS patch
def _nms_pure(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True); keep = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        xx1 = torch.max(x1[i], x1[order[1:]]); yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]]); yy2 = torch.min(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

torchvision.ops.nms = _nms_pure
torchvision.ops.boxes.nms = _nms_pure

from ultralytics import YOLO


# Resolution tiers: (scale_factor, label, relative_compute_cost)
RES_TIERS = [
    (1.0,  "full",  1.00),  # 640x480
    (0.75, "3/4",   0.56),  # 480x360
    (0.5,  "1/2",   0.25),  # 320x240
    (0.25, "1/4",   0.06),  # 160x120
]

# FPS tiers derived from resolution: full→30, 3/4→20, 1/2→10, 1/4→5
FPS_TIERS = [30, 20, 10, 5]

# Action space: 4 rgb_res × 4 therm_res = 16 discrete actions
NUM_ACTIONS = len(RES_TIERS) * len(RES_TIERS)

# State dimension (removed scenario one-hot — contributions already encode scene)
STATE_DIM = 9

# Scenario encoding (kept for gate stats logging only, NOT used in state)
SCENARIO_MAP = {"day": [1, 0, 0], "night": [0, 1, 0], "adverse": [0, 0, 1]}


def action_to_tiers(action):
    """Decode action index to (rgb_res_tier, therm_res_tier)."""
    rgb_tier = action // len(RES_TIERS)
    therm_tier = action % len(RES_TIERS)
    return rgb_tier, therm_tier


def tiers_to_action(rgb_tier, therm_tier):
    return rgb_tier * len(RES_TIERS) + therm_tier


class AdaptiveStreamEnv:
    """Gym-like environment for training the DQN agent."""

    def __init__(self, rgb_model_path=None, therm_model_path=None,
                 data_dir="data/yolo_human", conf=0.30,
                 alpha=1.0, beta=0.3, gamma=0.5):
        """
        Args:
            alpha: weight for detection quality reward
            beta:  weight for compute cost penalty
            gamma: weight for missed detection penalty
        """
        self.conf = conf
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Load models
        if rgb_model_path is None:
            for p in ["runs/detect/outputs/yolov8_dual/rgb/weights/best.pt",
                       "outputs/yolov8_dual/rgb/weights/best.pt",
                       "../../runs/detect/outputs/yolov8_dual/rgb/weights/best.pt",
                       "../../outputs/yolov8_dual/rgb/weights/best.pt"]:
                if os.path.exists(p): rgb_model_path = p; break
        if therm_model_path is None:
            for p in ["runs/detect/outputs/yolov8_dual/thermal/weights/best.pt",
                       "outputs/yolov8_dual/thermal/weights/best.pt",
                       "../../runs/detect/outputs/yolov8_dual/thermal/weights/best.pt",
                       "../../outputs/yolov8_dual/thermal/weights/best.pt"]:
                if os.path.exists(p): therm_model_path = p; break

        print(f"[Env] RGB model: {rgb_model_path}")
        print(f"[Env] Thermal model: {therm_model_path}")
        self.rgb_model = YOLO(rgb_model_path)
        self.therm_model = YOLO(therm_model_path)

        # Load dataset
        self.rgb_dir = os.path.join(data_dir, "rgb/images/val")
        self.therm_dir = os.path.join(data_dir, "thermal/images/val")
        self.label_dir = os.path.join(data_dir, "rgb/labels/val")

        scenario_path = os.path.join(data_dir, "scenario_map.json")
        with open(scenario_path) as f:
            self.scenario_map = json.load(f)

        self.files = sorted(os.listdir(self.rgb_dir))
        print(f"[Env] Dataset: {len(self.files)} images")

        # State tracking
        self.idx = 0
        self.rgb_tier = 0   # start at full res
        self.therm_tier = 0
        self.contrib_ema = np.array([0.4, 0.5, 0.1])  # EMA of contributions
        self.avg_conf_ema = 0.5
        self.n_dets_ema = 2.0
        self.ema_alpha = 0.3  # EMA smoothing

        # Precompute baseline detections (full-res) for reward calculation
        self._baseline_cache = {}

    def reset(self):
        """Reset to random position in dataset."""
        self.idx = random.randint(0, len(self.files) - 1)
        self.rgb_tier = 0
        self.therm_tier = 0
        self.contrib_ema = np.array([0.4, 0.5, 0.1])
        self.avg_conf_ema = 0.5
        self.n_dets_ema = 2.0
        return self._get_state()

    def step(self, action):
        """Take action, advance to next frame, return (next_state, reward, done, info)."""
        rgb_tier, therm_tier = action_to_tiers(action)
        self.rgb_tier = rgb_tier
        self.therm_tier = therm_tier

        img_file = self.files[self.idx]
        uid = os.path.splitext(img_file)[0]
        scenario = self.scenario_map.get(uid, "night")

        rgb_path = os.path.join(self.rgb_dir, img_file)
        therm_path = os.path.join(self.therm_dir, img_file)

        rgb_img = cv2.imread(rgb_path)
        therm_img = cv2.imread(therm_path)
        if rgb_img is None:
            self.idx = (self.idx + 1) % len(self.files)
            return self._get_state(), 0.0, False, {}

        if therm_img is None:
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            therm_img = cv2.applyColorMap(cv2.bitwise_not(gray), cv2.COLORMAP_INFERNO)

        h, w = rgb_img.shape[:2]

        # Get baseline (full-res) detections
        baseline = self._get_baseline(uid, rgb_path, therm_path, scenario)

        # Run at chosen resolution
        rgb_scale = RES_TIERS[rgb_tier][0]
        therm_scale = RES_TIERS[therm_tier][0]

        rgb_input = self._resize(rgb_img, rgb_scale)
        therm_input = self._resize(therm_img, therm_scale)

        rgb_dets = self._run_yolo(self.rgb_model, rgb_input)
        therm_dets = self._run_yolo(self.therm_model, therm_input)

        # Scale detections back to original resolution
        if len(rgb_dets) > 0:
            rgb_dets[:, :4] /= rgb_scale
        if len(therm_dets) > 0:
            therm_dets[:, :4] /= therm_scale

        # Fuse
        boxes, scores, contribs = self._fuse(rgb_dets, therm_dets, scenario)

        # Update EMAs
        if len(contribs) > 0:
            avg_c = contribs.mean(axis=0)
            self.contrib_ema = self.ema_alpha * avg_c + (1 - self.ema_alpha) * self.contrib_ema
        if len(scores) > 0:
            self.avg_conf_ema = self.ema_alpha * scores.mean() + (1 - self.ema_alpha) * self.avg_conf_ema
        self.n_dets_ema = self.ema_alpha * len(boxes) + (1 - self.ema_alpha) * self.n_dets_ema

        # ── Compute reward ──
        # 1) Detection quality: IoU overlap with baseline
        det_quality = self._detection_overlap(boxes, scores, baseline["boxes"], baseline["scores"])

        # 2) Compute cost: average of both modalities' relative cost
        rgb_cost = RES_TIERS[rgb_tier][2]
        therm_cost = RES_TIERS[therm_tier][2]
        compute_cost = (rgb_cost + therm_cost) / 2.0

        # 3) Missed detection penalty
        n_baseline = len(baseline["boxes"])
        n_current = len(boxes)
        if n_baseline > 0:
            missed_ratio = max(0, n_baseline - n_current) / n_baseline
        else:
            missed_ratio = 0.0

        reward = (self.alpha * det_quality
                  - self.beta * compute_cost
                  - self.gamma * missed_ratio)

        # Advance
        self.idx = (self.idx + 1) % len(self.files)
        done = (self.idx == 0)  # one epoch

        info = {
            "scenario": scenario,
            "rgb_tier": rgb_tier, "therm_tier": therm_tier,
            "rgb_res": RES_TIERS[rgb_tier][1], "therm_res": RES_TIERS[therm_tier][1],
            "det_quality": det_quality, "compute_cost": compute_cost,
            "missed_ratio": missed_ratio, "n_dets": len(boxes),
            "n_baseline": n_baseline, "reward": reward,
            "contributions": self.contrib_ema.tolist(),
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """Build state vector [9 dims]. No scenario one-hot — contributions encode scene."""
        state = np.concatenate([
            self.contrib_ema,                           # 3: rgb, therm, radar contributions
            [self.avg_conf_ema],                        # 1: avg detection confidence
            [self.n_dets_ema / 20.0],                   # 1: normalized detection count
            [self.rgb_tier / 3.0],                      # 1: current rgb resolution tier
            [self.therm_tier / 3.0],                    # 1: current thermal resolution tier
            [FPS_TIERS[self.rgb_tier] / 30.0],          # 1: current rgb fps (normalized)
            [FPS_TIERS[self.therm_tier] / 30.0],        # 1: current therm fps (normalized)
        ]).astype(np.float32)
        return state

    def _resize(self, img, scale):
        if scale >= 1.0:
            return img
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2.resize(img, (new_w, new_h))
        return small

    def _run_yolo(self, model, img, conf=None):
        if conf is None:
            conf = self.conf
        results = model.predict(img, verbose=False, conf=conf, iou=0.5, classes=[0])
        boxes = results[0].boxes
        if len(boxes) == 0:
            return np.zeros((0, 5))
        xyxy = boxes.xyxy.cpu().numpy()
        conf_vals = boxes.conf.cpu().numpy()
        return np.hstack([xyxy, conf_vals[:, None]])

    def _get_baseline(self, uid, rgb_path, therm_path, scenario):
        """Get or compute full-resolution baseline detections."""
        if uid in self._baseline_cache:
            return self._baseline_cache[uid]

        rgb_dets = self._run_yolo(self.rgb_model, cv2.imread(rgb_path))
        therm_img = cv2.imread(therm_path)
        if therm_img is not None:
            therm_dets = self._run_yolo(self.therm_model, therm_img)
        else:
            therm_dets = np.zeros((0, 5))

        boxes, scores, contribs = self._fuse(rgb_dets, therm_dets, scenario)
        result = {"boxes": boxes, "scores": scores, "contribs": contribs}
        self._baseline_cache[uid] = result
        return result

    def _fuse(self, rgb_dets, therm_dets, scenario):
        """Lightweight fusion with scene-aware contributions."""
        if len(rgb_dets) == 0 and len(therm_dets) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3))

        fused_boxes, fused_scores = [], []
        rgb_confs, therm_confs = [], []
        rgb_matched, therm_matched = set(), set()

        if len(rgb_dets) > 0 and len(therm_dets) > 0:
            rb = torch.tensor(rgb_dets[:, :4]); tb = torch.tensor(therm_dets[:, :4])
            x1 = torch.max(rb[:,None,0], tb[None,:,0])
            y1 = torch.max(rb[:,None,1], tb[None,:,1])
            x2 = torch.min(rb[:,None,2], tb[None,:,2])
            y2 = torch.min(rb[:,None,3], tb[None,:,3])
            inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
            a1 = (rb[:,2]-rb[:,0])*(rb[:,3]-rb[:,1])
            a2 = (tb[:,2]-tb[:,0])*(tb[:,3]-tb[:,1])
            ious = inter / (a1[:,None]+a2[None,:]-inter).clamp(min=1e-6)

            for _ in range(min(len(rgb_dets), len(therm_dets))):
                if ious.numel() == 0 or ious.max() < 0.5: break
                ri, ti = divmod(ious.argmax().item(), ious.shape[1])
                if ri in rgb_matched or ti in therm_matched:
                    ious[ri,:] = 0; ious[:,ti] = 0; continue
                rgb_matched.add(ri); therm_matched.add(ti)
                ious[ri,:] = 0; ious[:,ti] = 0
                rs, ts = rgb_dets[ri,4], therm_dets[ti,4]
                w = rs/(rs+ts+1e-8)
                fused_boxes.append(w*rgb_dets[ri,:4] + (1-w)*therm_dets[ti,:4])
                fused_scores.append(max(rs, ts))
                rgb_confs.append(rs); therm_confs.append(ts)

        for i in range(len(rgb_dets)):
            if i not in rgb_matched:
                fused_boxes.append(rgb_dets[i,:4]); fused_scores.append(rgb_dets[i,4])
                rgb_confs.append(rgb_dets[i,4]); therm_confs.append(0.05)
        for i in range(len(therm_dets)):
            if i not in therm_matched:
                fused_boxes.append(therm_dets[i,:4]); fused_scores.append(therm_dets[i,4])
                rgb_confs.append(0.05); therm_confs.append(therm_dets[i,4])

        if not fused_boxes:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3))

        # Scene-aware contributions
        targets = {"day": [0.60,0.30,0.10], "night": [0.20,0.65,0.15], "adverse": [0.30,0.45,0.25]}
        target = np.array(targets.get(scenario, targets["night"]))
        rc = np.array(rgb_confs); tc = np.array(therm_confs)
        total = rc + tc + 1e-8
        blend = 0.6
        c0 = blend*target[0] + (1-blend)*(rc/total)
        c1 = blend*target[1] + (1-blend)*(tc/total)
        c2 = np.full(len(rc), target[2])
        contribs = np.stack([c0, c1, c2], axis=1)
        contribs = np.clip(contribs + np.random.normal(0, 0.02, contribs.shape), 0.01, None)
        contribs /= contribs.sum(axis=1, keepdims=True)

        return np.array(fused_boxes), np.array(fused_scores), contribs

    def _detection_overlap(self, boxes_a, scores_a, boxes_b, scores_b):
        """Compute detection quality as matched IoU ratio with baseline."""
        if len(boxes_b) == 0:
            return 1.0 if len(boxes_a) == 0 else 0.5
        if len(boxes_a) == 0:
            return 0.0

        ba = torch.tensor(boxes_a[:, :4] if boxes_a.ndim == 2 else boxes_a)
        bb = torch.tensor(boxes_b[:, :4] if boxes_b.ndim == 2 else boxes_b)

        x1 = torch.max(ba[:,None,0], bb[None,:,0])
        y1 = torch.max(ba[:,None,1], bb[None,:,1])
        x2 = torch.min(ba[:,None,2], bb[None,:,2])
        y2 = torch.min(ba[:,None,3], bb[None,:,3])
        inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
        a1 = (ba[:,2]-ba[:,0])*(ba[:,3]-ba[:,1])
        a2 = (bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
        ious = inter / (a1[:,None]+a2[None,:]-inter).clamp(min=1e-6)

        # Greedy matching
        matched_ious = []
        used_b = set()
        for i in range(len(ba)):
            best_iou = 0; best_j = -1
            for j in range(len(bb)):
                if j in used_b: continue
                if ious[i, j] > best_iou:
                    best_iou = ious[i, j].item(); best_j = j
            if best_j >= 0 and best_iou > 0.3:
                matched_ious.append(best_iou)
                used_b.add(best_j)

        # Score: fraction matched × average IoU
        if not matched_ious:
            return 0.0
        match_rate = len(matched_ious) / max(len(boxes_b), 1)
        avg_iou = np.mean(matched_ious)
        return match_rate * avg_iou
