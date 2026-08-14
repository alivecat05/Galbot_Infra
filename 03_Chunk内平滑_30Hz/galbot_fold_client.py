"""Archive profile v03: 30 Hz control with within-chunk smoothing."""

import argparse
import logging
import queue
import sys
import threading
import time

import numpy as np

from model_server.tools.websocket_policy_client import WebsocketClientPolicy

try:  # Supports both ``python file.py`` and package imports in tests.
    from .fold_clothes_protocol import ACTION_KEYS, ACTION_SLICES
except ImportError:
    from fold_clothes_protocol import ACTION_KEYS, ACTION_SLICES


CAMERA_ORDER = ("HEAD_LEFT_CAMERA", "HEAD_RIGHT_CAMERA", "LEFT_ARM_CAMERA", "RIGHT_ARM_CAMERA")
CAMERA_ALIASES = {"LEFT_WRIST_CAMERA": "LEFT_ARM_CAMERA", "RIGHT_WRIST_CAMERA": "RIGHT_ARM_CAMERA"}
ARM_GROUPS = ("left_arm", "right_arm")
GRIPPER_GROUPS = ("left_gripper", "right_gripper")
MODEL_GRIPPER_UNITS_PER_METER = 1000.0  # Training data stores millimetres (0--100); SDK uses metres.


def decode_rgb(compressed):
    import cv2

    image = cv2.imdecode(np.frombuffer(compressed["data"], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode Galbot RGB frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_transport_image(compressed, image_size, jpeg_quality):
    """Resize at the robot and send only model-resolution JPEG bytes."""
    import cv2

    image = cv2.imdecode(np.frombuffer(compressed["data"], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode Galbot RGB frame")
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    ok, payload = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        raise ValueError("Failed to JPEG-encode Galbot RGB frame")
    return {
        "__encoded_image__": True,
        "codec": "jpeg",
        "width": image_size,
        "height": image_size,
        "data": payload.tobytes(),
    }


def resolve_sensors(sensor_type, names):
    sensors = []
    for name in names:
        name = CAMERA_ALIASES.get(name, name)
        if not hasattr(sensor_type, name):
            raise ValueError(f"SensorType.{name} is unavailable in this SDK")
        sensors.append(getattr(sensor_type, name))
    return sensors


def get_observation(
    robot,
    sensors,
    lang,
    include_images=True,
    encoded_images=False,
    transport_image_size=224,
    transport_jpeg_quality=75,
):
    images = []
    if include_images:
        for sensor in sensors:
            frame = robot.get_rgb_data(sensor)
            if not frame:
                raise RuntimeError(f"No RGB frame from {sensor}")
            if encoded_images:
                images.append(
                    encode_transport_image(frame, transport_image_size, transport_jpeg_quality)
                )
            else:
                images.append(decode_rgb(frame))

    arms = []
    for group in ARM_GROUPS:
        arms.extend(joint.position for joint in robot.get_joint_states([group], []))
    widths_m = []
    for group in GRIPPER_GROUPS:
        state = robot.get_gripper_state(group)
        if state is None:
            raise RuntimeError(f"No width feedback from {group}")
        widths_m.append(state.width)
    state = np.asarray(arms + [width * MODEL_GRIPPER_UNITS_PER_METER for width in widths_m], dtype=np.float32)
    if state.shape != (16,):
        raise RuntimeError(f"Expected 16 state values (7+7+1+1), got {state.shape}")
    return {"images": images, "lang": lang, "state": state}


def update_observation_history(history, observation, obs_horizon):
    """Return the model's view-major image history and matching state history."""
    if obs_horizon < 1:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
    history = (history + [observation])[-obs_horizon:]
    history = [history[0]] * (obs_horizon - len(history)) + history
    if any(len(item["images"]) != len(CAMERA_ORDER) for item in history):
        raise ValueError("Each historical observation must contain four RGB images")
    return history, {
        "images": [item["images"][view] for view in range(len(CAMERA_ORDER)) for item in history],
        "lang": observation["lang"],
        "state": np.stack([item["state"] for item in history]),
    }


def capture_observation_history(
    robot,
    sensors,
    lang,
    obs_horizon,
    hz,
    encoded_images=False,
    transport_image_size=224,
    transport_jpeg_quality=75,
):
    history = []
    for index in range(obs_horizon):
        history, model_input = update_observation_history(
            history,
            get_observation(
                robot,
                sensors,
                lang,
                encoded_images=encoded_images,
                transport_image_size=transport_image_size,
                transport_jpeg_quality=transport_jpeg_quality,
            ),
            obs_horizon,
        )
        if index + 1 < obs_horizon:
            time.sleep(1 / hz)
    return history, model_input


class ObservationHistoryWorker:
    """Capture/encode model observations without disturbing the control cadence."""

    def __init__(
        self,
        robot,
        sensors,
        lang,
        obs_horizon,
        hz,
        history,
        model_input,
        encoded_images,
        transport_image_size,
        transport_jpeg_quality,
    ):
        self.robot = robot
        self.sensors = sensors
        self.lang = lang
        self.obs_horizon = obs_horizon
        self.period = 1.0 / hz
        self.history = history
        self.latest = model_input
        self.encoded_images = encoded_images
        self.transport_image_size = transport_image_size
        self.transport_jpeg_quality = transport_jpeg_quality
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        deadline = time.monotonic() + self.period
        while not self.stop.is_set():
            if self.stop.wait(max(0, deadline - time.monotonic())):
                break
            try:
                observation = get_observation(
                    self.robot,
                    self.sensors,
                    self.lang,
                    encoded_images=self.encoded_images,
                    transport_image_size=self.transport_image_size,
                    transport_jpeg_quality=self.transport_jpeg_quality,
                )
                with self.lock:
                    self.history, self.latest = update_observation_history(
                        self.history, observation, self.obs_horizon,
                    )
                    self.error = None
            except Exception as exc:
                self.error = exc
                logging.exception("Background observation capture failed")
            deadline = max(deadline + self.period, time.monotonic())

    def snapshot(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def split_action(row):
    return {key: np.asarray(row[start:end], dtype=np.float32)[None] for key, (start, end) in ACTION_SLICES.items()}


def make_plan(data):
    plan = np.concatenate([np.asarray(data[key], dtype=np.float32) for key in ACTION_KEYS], axis=-1)
    if plan.ndim != 2 or plan.shape[1] != 16 or len(plan) == 0:
        raise ValueError(f"Expected non-empty [steps, 16] action chunk, got {plan.shape}")
    return plan


def smooth_action_plan(plan, window):
    """Suppress high-frequency arm motion inside a model action chunk."""
    if window <= 1 or len(plan) <= 1:
        return plan
    radius = window // 2
    weights = np.arange(1, radius + 2, dtype=np.float32)
    weights = np.concatenate([weights, weights[-2::-1]])
    weights /= weights.sum()
    padded = np.pad(plan[:, :14], ((radius, radius), (0, 0)), mode="edge")
    smoothed = plan.copy()
    for index in range(len(plan)):
        smoothed[index, :14] = np.sum(
            padded[index:index + window] * weights[:, None], axis=0,
        )
    return smoothed


def apply_pending_plan(plan, pending, now, hz):
    """Start a completed chunk at its matching action time, never in the past."""
    if len(plan) or pending is None:
        return plan, pending, 0
    captured_at, data = pending
    candidate = make_plan(data)
    elapsed_steps = max(0, int((now - captured_at) * hz + 1e-3))
    return candidate[elapsed_steps:], None, elapsed_steps


class GripperGate:
    """Avoid reissuing noisy continuous policy outputs as physical gripper moves."""

    def __init__(self, deadband_m, min_interval_s, speed_mps, effort):
        self.deadband_m = deadband_m
        self.min_interval_s = min_interval_s
        self.speed_mps = speed_mps
        self.effort = effort
        self.last = {}

    def send(self, robot, target, now):
        for group in GRIPPER_GROUPS:
            width_m = float(target[group][0, 0] / MODEL_GRIPPER_UNITS_PER_METER)
            feedback = robot.get_gripper_state(group)
            if feedback is None:
                logging.warning("No width feedback from %s; skipping command", group)
                continue
            if feedback.is_moving:
                continue
            previous = self.last.get(group)
            if previous and now - previous[0] < self.min_interval_s:
                continue
            if previous and abs(width_m - previous[1]) < self.deadband_m:
                continue
            status = robot.set_gripper_command(group, width_m, self.speed_mps, self.effort, False)
            if getattr(status, "name", "SUCCESS") == "SUCCESS":
                self.last[group] = (now, width_m)
                logging.info("%s -> %.3fm", group, width_m)
            else:
                logging.warning("%s command returned %s", group, status)


class RobotControl:
    def __init__(self, robot, joint_command, args):
        self.robot = robot
        self.joint_command = joint_command
        self.grippers = GripperGate(
            args.gripper_deadband_m, args.gripper_min_interval_s,
            args.gripper_speed_mps, args.gripper_effort,
        )
        self.last_arm_target = None
        self.arm_filter_alpha = args.arm_filter_alpha
        self.max_arm_step_rad = args.max_arm_step_rad
        self.arm_deadband_rad = args.arm_deadband_rad
        self.reverse_confirm_steps = args.reverse_confirm_steps
        self.reverse_threshold_rad = args.reverse_threshold_rad
        self.arm_direction = np.zeros(14, dtype=np.int8)
        self.reverse_counts = np.zeros(14, dtype=np.int16)

    def reset_arm_target(self, arm_state):
        arm_state = np.asarray(arm_state, dtype=np.float32)
        if arm_state.shape != (14,):
            raise ValueError(f"Expected 14 arm joints, got {arm_state.shape}")
        self.last_arm_target = arm_state.copy()

    def filter_arm_target(self, desired):
        desired = np.asarray(desired, dtype=np.float32)
        if self.last_arm_target is None:
            return desired
        delta = desired - self.last_arm_target
        delta[np.abs(delta) < self.arm_deadband_rad] = 0.0
        direction = np.sign(delta).astype(np.int8)
        moving = np.abs(delta) >= self.reverse_threshold_rad
        reversing = moving & (self.arm_direction != 0) & (direction != self.arm_direction)
        self.reverse_counts[reversing] += 1
        self.reverse_counts[~reversing] = 0
        blocked = reversing & (self.reverse_counts < self.reverse_confirm_steps)
        delta[blocked] = 0.0
        confirmed = reversing & ~blocked
        same_direction = moving & ~reversing
        self.arm_direction[confirmed | same_direction] = direction[confirmed | same_direction]
        self.reverse_counts[confirmed] = 0
        delta *= self.arm_filter_alpha
        if self.max_arm_step_rad > 0:
            delta = np.clip(delta, -self.max_arm_step_rad, self.max_arm_step_rad)
        return self.last_arm_target + delta

    def send_arm_target(self, arm_target, target, now, send_grippers=True):
        arm_target = np.asarray(arm_target, dtype=np.float32)
        if arm_target.shape != (14,):
            raise ValueError(f"Expected 14 arm targets, got {arm_target.shape}")
        commands = [self.joint_command() for _ in arm_target]
        for command, position in zip(commands, arm_target):
            command.position = float(position)
        sent_target = np.asarray([command.position for command in commands], dtype=np.float64)
        assert np.array_equal(sent_target, arm_target.astype(np.float64)), (
            f"SDK joint target differs from filtered target: sent={sent_target}, target={arm_target}"
        )
        logging.debug("arm filtered target: %s", sent_target.tolist())
        status = self.robot.set_joint_commands(commands, list(ARM_GROUPS), [], 0.0)
        if getattr(status, "name", "SUCCESS") != "SUCCESS":
            logging.warning("arm command returned %s", status)
            return False
        self.last_arm_target = arm_target.copy()
        if send_grippers and self.grippers is not None:
            self.grippers.send(self.robot, target, now)
            logging.debug(
                "gripper target fidelity: exact; sent_target=%s",
                [float(target[group][0, 0] / MODEL_GRIPPER_UNITS_PER_METER) for group in GRIPPER_GROUPS],
            )
        return True

    def execute(self, target, current, now):
        desired = np.concatenate([target[group][0] for group in ARM_GROUPS])
        return self.send_arm_target(
            self.filter_arm_target(desired), target, now, send_grippers=True,
        )

    def execute_interpolated(self, target, duration_s, command_hz):
        """Linearly upsample one model action interval for smooth joint commands."""
        desired = np.concatenate([target[group][0] for group in ARM_GROUPS])
        endpoint = self.filter_arm_target(desired)
        start = endpoint.copy() if self.last_arm_target is None else self.last_arm_target.copy()
        subdivisions = max(1, int(round(duration_s * command_hz)))
        interval = duration_s / subdivisions
        deadline = time.monotonic()
        for index in range(subdivisions):
            alpha = (index + 1) / subdivisions
            arm_target = start + alpha * (endpoint - start)
            if not self.send_arm_target(
                arm_target,
                target,
                time.monotonic(),
                send_grippers=index == 0,
            ):
                return False
            deadline += interval
            time.sleep(max(0, deadline - time.monotonic()))
        return True


def blend_plan_start(plan, previous_arm_target, blend_steps):
    """Ramp into a new chunk to suppress target jumps at chunk boundaries."""
    if previous_arm_target is None or blend_steps <= 0:
        return plan
    blended = plan.copy()
    count = min(blend_steps, len(blended))
    for index in range(count):
        alpha = 1.0 if count == 1 else index / (count - 1)
        blended[index, :14] = (1.0 - alpha) * previous_arm_target + alpha * plan[index, :14]
    return blended


def wait_for_result(worker):
    """Wait without repeatedly resetting the robot's active position target."""
    while True:
        response = worker.poll()
        if response is not None:
            return response
        if worker.idle.is_set() and worker.error:
            raise worker.error
        time.sleep(0.01)


class AsyncPolicyWorker:
    """Keeps one websocket and retains only the newest observation/response."""

    def __init__(self, host, port):
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.idle = threading.Event()
        self.error = None
        self.client = None
        self.thread = threading.Thread(target=self._run, args=(host, port), daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.error:
            raise self.error

    @staticmethod
    def _put_latest(queue_, value):
        try:
            queue_.get_nowait()
        except queue.Empty:
            pass
        queue_.put_nowait(value)

    def _run(self, host, port):
        try:
            self.client = WebsocketClientPolicy(host, port)
            self.metadata = self.client.get_server_metadata()
        except Exception as exc:
            self.error = exc
            self.ready.set()
            return
        self.ready.set()
        self.idle.set()
        while not self.stop.is_set():
            try:
                captured_at, observation = self.requests.get(timeout=0.1)
                self.idle.clear()
                try:
                    response = self.client.predict_action({"type": "infer", **observation})
                    log_transport_metrics(self.client, response)
                    if response.get("ok"):
                        self._put_latest(self.results, (captured_at, response["data"]))
                        self.error = None
                    else:
                        self.error = RuntimeError(f"Inference error: {response.get('error')}")
                        logging.error("%s", self.error)
                finally:
                    self.idle.set()
            except queue.Empty:
                pass
            except Exception as exc:
                self.error = exc
                self.idle.set()
                logging.exception("Inference request failed")

    def request(self, observation):
        self._put_latest(self.requests, (time.monotonic(), observation))

    def poll(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def wait(self):
        """Wait for the in-flight inference while keeping worker failures visible."""
        while not self.stop.is_set():
            try:
                return self.results.get(timeout=0.1)
            except queue.Empty:
                if self.idle.is_set() and self.error:
                    raise self.error
        raise RuntimeError("Policy worker stopped before returning an action chunk")

    def close(self):
        self.stop.set()
        if self.client:
            self.client.close()
        self.thread.join(timeout=1)


def log_transport_metrics(client, response):
    metrics = getattr(client, "last_metrics", None)
    if not metrics:
        return
    server = response.get("server_timing_ms") or response.get("data", {}).get("timing_ms", {})
    logging.info(
        "Inference transport: %.2f MiB request, %.0fms round-trip "
        "(pack %.0fms, send %.0fms, server decode %.0fms, policy %.0fms, debug I/O %.0fms)",
        metrics["request_bytes"] / (1024 * 1024),
        metrics["round_trip_ms"],
        metrics["pack_ms"],
        metrics["send_ms"],
        server.get("decode_ms", 0),
        server.get("policy_ms", 0),
        server.get("debug_io_ms", 0),
    )


def add_arguments(parser, rtc):
    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--lang", default="fold clothes")
    parser.add_argument("--camera_sensors", default=",".join(CAMERA_ORDER))
    parser.add_argument("--hz", type=float, default=30.0, help="Model action frequency")
    parser.add_argument("--disable_grippers", action="store_true")
    parser.add_argument("--gripper_deadband_m", type=float, default=0.01)
    parser.add_argument("--gripper_min_interval_s", type=float, default=0.3)
    parser.add_argument("--gripper_speed_mps", type=float, default=0.05)
    parser.add_argument("--gripper_effort", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument(
        "--raw_image_transport", action="store_true",
        help="Send raw RGB arrays instead of the SDK's compressed camera frames",
    )
    parser.add_argument(
        "--transport_image_size", type=int, default=224,
        help="Resize each camera frame before transport; must match the model input size",
    )
    parser.add_argument(
        "--transport_jpeg_quality", type=int, default=75,
        help="JPEG quality used for model-resolution image transport",
    )
    parser.add_argument(
        "--prefetch_step", type=int, default=-1,
        help="Deprecated compatibility option; synchronous mode now replans before execution",
    )
    parser.add_argument(
        "--sync_execute_steps", type=int, default=0,
        help="Maximum fresh action steps executed before synchronous replanning; 0 uses all fresh steps",
    )
    parser.add_argument(
        "--chunk_blend_steps", type=int, default=3,
        help="Blend this many arm targets at action-chunk boundaries; 0 disables blending",
    )
    parser.add_argument(
        "--chunk_smoothing_window", type=int, default=5,
        help="Odd moving-average window for arm targets inside each action chunk",
    )
    parser.add_argument(
        "--arm_filter_alpha", type=float, default=0.25,
        help="Online arm low-pass coefficient; smaller is smoother and more delayed",
    )
    parser.add_argument(
        "--max_arm_step_rad", type=float, default=0.02,
        help="Maximum commanded change per arm joint in one control period; 0 disables",
    )
    parser.add_argument(
        "--arm_deadband_rad", type=float, default=0.0,
        help="Ignore arm target changes smaller than this many radians",
    )
    parser.add_argument(
        "--reverse_confirm_steps", type=int, default=1,
        help="Require this many consecutive opposite targets before reversing a joint",
    )
    parser.add_argument(
        "--reverse_threshold_rad", type=float, default=0.004,
        help="Apply reversal confirmation only above this joint error",
    )
    parser.add_argument(
        "--command_hz", type=float, default=30.0,
        help="Joint-command interpolation frequency; model actions remain at --hz",
    )
    if not rtc:
        parser.set_defaults(hz=30.0)


def create_robot(args):
    if args.sdk_root:
        sys.path.insert(0, args.sdk_root)
    try:
        from galbot_sdk.g1 import GalbotRobot, JointCommand, SensorType
    except ImportError as exc:
        raise RuntimeError("Cannot import galbot_sdk; set --sdk_root to the SDK directory") from exc
    sensors = resolve_sensors(SensorType, [name.strip() for name in args.camera_sensors.split(",")])
    if len(sensors) != 4:
        raise ValueError("--camera_sensors must name exactly four RGB cameras")
    robot = GalbotRobot()
    if not robot.init(set(sensors)):
        raise RuntimeError("GalbotRobot initialization failed")
    time.sleep(2)
    return robot, sensors, JointCommand


def run_sync(args, robot, sensors, controller):
    worker = AsyncPolicyWorker(args.server_ip, args.port)
    obs_horizon = int(worker.metadata.get("obs_horizon", 1))
    encoded_images = worker.metadata.get("encoded_images", False) and not args.raw_image_transport
    server_size = worker.metadata.get("transport_image_size")
    if isinstance(server_size, (list, tuple)) and len(server_size) == 2:
        if server_size[0] != server_size[1]:
            raise ValueError(f"Only square model input is supported, got {server_size}")
        args.transport_image_size = int(server_size[0])
    logging.info("Connected: %s", worker.metadata)
    logging.info("Image transport: %s", "224px JPEG" if encoded_images else "raw RGB")
    capture_started = time.monotonic()
    history, next_observation = capture_observation_history(
        robot,
        sensors,
        args.lang,
        obs_horizon,
        args.hz,
        encoded_images=encoded_images,
        transport_image_size=args.transport_image_size,
        transport_jpeg_quality=args.transport_jpeg_quality,
    )
    logging.info("Initial observation capture: %.0fms", (time.monotonic() - capture_started) * 1000)
    controller.reset_arm_target(next_observation["state"][-1, :14])
    observation_worker = ObservationHistoryWorker(
        robot,
        sensors,
        args.lang,
        obs_horizon,
        args.hz,
        history,
        next_observation,
        encoded_images,
        args.transport_image_size,
        args.transport_jpeg_quality,
    )
    worker.request(next_observation)
    try:
        for _ in range(args.max_steps):
            waited_at = time.monotonic()
            _, data = wait_for_result(worker)
            wait_ms = (time.monotonic() - waited_at) * 1000
            raw_plan = make_plan(data)
            current_observation = observation_worker.snapshot()
            controller.reset_arm_target(current_observation["state"][-1, :14])
            plan = smooth_action_plan(raw_plan, args.chunk_smoothing_window)
            plan = blend_plan_start(
                plan, controller.last_arm_target, args.chunk_blend_steps,
            )
            raw_step_max = float(np.abs(np.diff(raw_plan[:, :14], axis=0)).max()) if len(raw_plan) > 1 else 0.0
            smooth_step_max = float(np.abs(np.diff(plan[:, :14], axis=0)).max()) if len(plan) > 1 else 0.0
            execute_steps = len(plan) if args.sync_execute_steps == 0 else min(
                len(plan), args.sync_execute_steps,
            )
            logging.info(
                "Fresh action chunk: execute=%d/%d steps; waited %.0fms; "
                "arm step max %.4f -> %.4f rad",
                execute_steps, len(plan), wait_ms, raw_step_max, smooth_step_max,
            )
            for row in plan[:execute_steps]:
                controller.execute_interpolated(
                    split_action(row), 1.0 / args.hz, args.command_hz,
                )

            # Strict synchronous replanning: capture after the short action horizon,
            # then hold the last target until this single request returns.
            if observation_worker.error:
                raise observation_worker.error
            worker.request(observation_worker.snapshot())
    finally:
        observation_worker.close()
        worker.close()


def run_rtc(args, robot, sensors, controller):
    worker = AsyncPolicyWorker(args.server_ip, args.port)
    logging.info("Connected: %s", worker.metadata)
    obs_horizon = int(worker.metadata.get("obs_horizon", 1))
    encoded_images = worker.metadata.get("encoded_images", False) and not args.raw_image_transport
    server_size = worker.metadata.get("transport_image_size")
    if isinstance(server_size, (list, tuple)) and len(server_size) == 2:
        if server_size[0] != server_size[1]:
            raise ValueError(f"Only square model input is supported, got {server_size}")
        args.transport_image_size = int(server_size[0])
    logging.info("Image transport: %s", "224px JPEG" if encoded_images else "raw RGB")
    plan = np.empty((0, 16), dtype=np.float32)
    pending = None
    history = []
    try:
        for _ in range(args.max_steps):
            started = time.monotonic()
            history, observation = update_observation_history(
                history,
                get_observation(
                    robot,
                    sensors,
                    args.lang,
                    encoded_images=encoded_images,
                    transport_image_size=args.transport_image_size,
                    transport_jpeg_quality=args.transport_jpeg_quality,
                ),
                obs_horizon,
            )
            capture_ms = (time.monotonic() - started) * 1000
            if capture_ms > 500:
                logging.warning("Observation capture took %.0fms", capture_ms)
            worker.request(observation)
            response = worker.poll()
            if response is not None:
                try:
                    captured_at, data = response
                    make_plan(data)
                    pending = (captured_at, data)
                except (KeyError, TypeError, ValueError):
                    logging.exception("Discarding malformed action chunk")
            plan, pending, skipped = apply_pending_plan(plan, pending, time.monotonic(), args.hz)
            if skipped:
                latency_ms = response[1].get("latency_ms") if response is not None else None
                if len(plan):
                    logging.info(
                        "RTC skipped %d stale action steps (chunk age %.0fms, server %.0fms)",
                        skipped, skipped * 1000 / args.hz, latency_ms or 0,
                    )
                else:
                    logging.warning(
                        "RTC discarded an expired 16-step chunk (age >= %.0fms, server %.0fms); "
                        "use the non-RTC client or reduce inference below %.0fms",
                        1000 * 16 / args.hz, latency_ms or 0, 1000 * 16 / args.hz,
                    )
            if len(plan) and controller.execute(split_action(plan[0]), observation["state"], time.monotonic()):
                plan = plan[1:]
            time.sleep(max(0, 1 / args.hz - (time.monotonic() - started)))
    finally:
        worker.close()


def main(rtc=False):
    parser = argparse.ArgumentParser(description="Galbot fold-clothes RTC client" if rtc else "Galbot fold-clothes synchronous client")
    parser.add_argument("--sdk_root", default="/root/GalbotSDK1_7_1/LDA-1B")
    add_arguments(parser, rtc)
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be positive")
    if args.prefetch_step < -1:
        parser.error("--prefetch_step must be -1 or non-negative")
    if args.sync_execute_steps < 0:
        parser.error("--sync_execute_steps must be non-negative")
    if args.chunk_blend_steps < 0:
        parser.error("--chunk_blend_steps must be non-negative")
    if args.chunk_smoothing_window < 1 or args.chunk_smoothing_window % 2 == 0:
        parser.error("--chunk_smoothing_window must be a positive odd integer")
    if not 0 < args.arm_filter_alpha <= 1:
        parser.error("--arm_filter_alpha must be in (0, 1]")
    if args.max_arm_step_rad < 0:
        parser.error("--max_arm_step_rad must be non-negative")
    if args.arm_deadband_rad < 0:
        parser.error("--arm_deadband_rad must be non-negative")
    if args.reverse_confirm_steps < 1:
        parser.error("--reverse_confirm_steps must be positive")
    if args.reverse_threshold_rad < 0:
        parser.error("--reverse_threshold_rad must be non-negative")
    if args.command_hz < args.hz:
        parser.error("--command_hz must be greater than or equal to --hz")
    if args.transport_image_size < 32:
        parser.error("--transport_image_size must be at least 32")
    if not 1 <= args.transport_jpeg_quality <= 100:
        parser.error("--transport_jpeg_quality must be in [1, 100]")
    if not 0 < args.gripper_speed_mps <= 0.2:
        parser.error("G1 requires 0 < --gripper_speed_mps <= 0.2")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    robot, sensors, joint_command = create_robot(args)
    controller = RobotControl(robot, joint_command, args)
    logging.info(
        "Synchronous model actions at %.1fHz, interpolated commands at %.1fHz; "
        "chunk smooth=%d, low-pass=%.2f, max-step=%.3frad",
        args.hz,
        args.command_hz,
        args.chunk_smoothing_window,
        args.arm_filter_alpha,
        args.max_arm_step_rad,
    )
    try:
        (run_rtc if rtc else run_sync)(args, robot, sensors, controller)
    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
