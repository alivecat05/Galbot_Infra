#!/usr/bin/env python3
"""WebSocket inference server for the Galbot dual-arm fold-clothes model."""

import argparse
import logging
import time

import numpy as np
import torch
from PIL import Image

from lda.model.framework.base_framework import baseframework
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
try:  # Supports both ``python file.py`` and package imports in tests.
    from .fold_clothes_protocol import (
        ACTION_KEYS, denormalize_actions, load_stats, normalize_state, state_distribution_report,
    )
except ImportError:
    from fold_clothes_protocol import (
        ACTION_KEYS, denormalize_actions, load_stats, normalize_state, state_distribution_report,
    )


ROOT = "/home/dwsu/WAM/LDA-1B"
# DEFAULT_CKPT = (
#     f"/home/dwsu/WAM/LDA-1B/results/Checkpoints/galbot_fold_clothes_v2_repeat4_40k/final_model/pytorch_model.pt"
# )
DEFAULT_CKPT = (
    f"/home/dwsu/WAM/LDA-1B/results/Checkpoints/galbot_fold_clothes_10hz_v1/final_model/pytorch_model.pt"
)
DEFAULT_NORM_STATS = f"{ROOT}/playground/fold_clothes/meta/stats_gr00t.json"
EMBODIMENT_ID = 27


class FoldClothesPolicy:
    def __init__(self, checkpoint, norm_stats, n_action_steps, seed, debug_dir=None):
        logging.info("Loading model from %s", checkpoint)
        started = time.time()
        self.model = baseframework.from_pretrained(checkpoint).to("cuda").eval()
        action_model = self.model.action_model
        self.stats = load_stats(norm_stats)
        self.n_action_steps = n_action_steps
        self.seed = seed
        self.debug_dir = debug_dir
        config = self.model.config.framework.action_model
        self.num_views = config.num_views
        self.obs_horizon = config.obs_horizon
        if self.num_views != 4 or config.state_dim != 16:
            raise ValueError(f"Expected 4 cameras and state_dim=16, got {self.num_views}, {config.state_dim}")
        expected_obs_merger_input = action_model.vision_encoder.config.hidden_size * (self.obs_horizon + 1)
        if action_model.obs_merger.in_features != expected_obs_merger_input:
            raise ValueError(
                f"obs_merger expects {action_model.obs_merger.in_features} features, but this 4-camera "
                f"checkpoint produces {expected_obs_merger_input}; do not apply the single-arm compatibility patch."
            )
        logging.info("Model loaded in %.0fs", time.time() - started)

    def predict_action(self, **msg):
        started = time.time()
        images = msg.get("images")
        expected_images = self.num_views * self.obs_horizon
        if not isinstance(images, (list, tuple)) or len(images) != expected_images:
            raise ValueError(
                f"images must contain {expected_images} RGB arrays in view-major time order "
                "(head_left, head_right, left_arm, right_arm; oldest to newest per view)"
            )
        pil_images = []
        for image in images:
            image = np.asarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Each image must have shape (H, W, 3), got {image.shape}")
            pil_images.append(Image.fromarray(image).resize((224, 224)))

        state = np.asarray(msg.get("state"), dtype=np.float32)
        if state.shape != (self.obs_horizon, 16):
            raise ValueError(f"state must have shape ({self.obs_horizon}, 16), got {state.shape}")
        example = {
            "image": pil_images,
            "lang": msg.get("lang", "fold clothes"),
            "state": normalize_state(msg.get("state"), self.stats),
            "embodiment_id": EMBODIMENT_ID,
            "history_action": None,
        }
        # The action head starts each diffusion rollout with ``torch.randn``. Reset
        # the CUDA RNG for each request so repeated observations keep one target.
        with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            normalized = self.model.predict_action(example)["normalized_actions"][0]
        actions = denormalize_actions(normalized[:self.n_action_steps], self.stats)
        joint_action = np.concatenate([actions[key] for key in ACTION_KEYS], axis=-1)
        below, above = state_distribution_report(state[-1], self.stats)
        arm_delta = np.abs(np.diff(joint_action[:, :14], axis=0))
        logging.info(
            "chunk: arm first-state-max=%.4frad, step-max=%.4frad, start-end-max=%.4frad, "
            "state clip=%d below/%d above",
            float(np.abs(joint_action[0, :14] - state[-1, :14]).max()),
            float(arm_delta.max()) if len(arm_delta) else 0.0,
            float(np.abs(joint_action[-1, :14] - joint_action[0, :14]).max()),
            len(below), len(above),
        )
        if below or above:
            logging.warning("State outside training q01/q99: below=%s above=%s", below, above)
        if self.debug_dir:
            timestamp = int(time.time() * 1000)
            np.savez_compressed(
                f"{self.debug_dir}/observation_{timestamp}.npz",
                **{f"image_{index}": image for index, image in enumerate(images)}, state=state,
                lang=example["lang"], joint_action=joint_action,
            )
        return {
            **actions,
            "joint_action": joint_action,
            "n_action_steps": self.n_action_steps,
            "latency_ms": (time.time() - started) * 1000,
        }


def main():
    parser = argparse.ArgumentParser(description="LDA dual-arm fold-clothes inference server")
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT)
    parser.add_argument("--norm_stats", default=DEFAULT_NORM_STATS)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="Fixed diffusion seed for deterministic control targets")
    parser.add_argument("--debug_dir", default='/home/dwsu/WAM/LDA-1B/results/Checkpoints/galbot_fold_clothes_v2_repeat4_40k/debug', help="Optional directory to save raw requests and predicted chunks")
    args = parser.parse_args()
    if args.n_action_steps < 1 or args.n_action_steps > 16:
        parser.error("--n_action_steps must be in [1, 16]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.debug_dir:
        import os
        os.makedirs(args.debug_dir, exist_ok=True)
    policy = FoldClothesPolicy(args.ckpt_path, args.norm_stats, args.n_action_steps, args.seed, args.debug_dir)
    WebsocketPolicyServer(policy, args.host, args.port, metadata={
        "model": "LDA-1B-galbot-dual-arm-fold-clothes-v1",
        "embodiment_id": EMBODIMENT_ID,
        "camera_order": ["head_left", "head_right", "left_arm", "right_arm"],
        "state_action_order": list(ACTION_KEYS),
        "state_dim": 16,
        "obs_horizon": policy.obs_horizon,
        "image_order": "view-major; oldest to newest per camera",
        "n_action_steps": args.n_action_steps,
        "seed": args.seed,
    }).serve_forever()


if __name__ == "__main__":
    main()
