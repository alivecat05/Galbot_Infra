"""Galbot G1 deployment client for the dual-arm fold-clothes policy."""

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


def resolve_sensors(sensor_type, names):
    sensors = []
    for name in names:
        name = CAMERA_ALIASES.get(name, name)
        if not hasattr(sensor_type, name):
            raise ValueError(f"SensorType.{name} is unavailable in this SDK")
        sensors.append(getattr(sensor_type, name))
    return sensors


def get_observation(robot, sensors, lang, include_images=True, encoded_images=False):
    images = []
    if include_images:
        for sensor in sensors:
            frame = robot.get_rgb_data(sensor)
            if not frame:
                raise RuntimeError(f"No RGB frame from {sensor}")
            if encoded_images:
                # The SDK already supplies JPEG/PNG bytes. Keep them compressed on the wire;
                # a capable policy server decodes them immediately before inference.
                images.append({"__encoded_image__": True, "data": bytes(frame["data"])})
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


def capture_observation_history(robot, sensors, lang, obs_horizon, hz, encoded_images=False):
    history = []
    for index in range(obs_horizon):
        history, model_input = update_observation_history(
            history,
            get_observation(robot, sensors, lang, encoded_images=encoded_images),
            obs_horizon,
        )
        if index + 1 < obs_horizon:
            time.sleep(1 / hz)
    return model_input


def split_action(row):
    return {key: np.asarray(row[start:end], dtype=np.float32)[None] for key, (start, end) in ACTION_SLICES.items()}


def make_plan(data):
    plan = np.concatenate([np.asarray(data[key], dtype=np.float32) for key in ACTION_KEYS], axis=-1)
    if plan.ndim != 2 or plan.shape[1] != 16 or len(plan) == 0:
        raise ValueError(f"Expected non-empty [steps, 16] action chunk, got {plan.shape}")
    return plan


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

    def execute(self, target, current, now):
        model_prediction = np.concatenate([target[group][0] for group in ARM_GROUPS])
        commands = [self.joint_command() for _ in model_prediction]
        for command, position in zip(commands, model_prediction):
            command.position = float(position)
        sent_target = np.asarray([command.position for command in commands], dtype=np.float64)
        assert np.array_equal(sent_target, model_prediction.astype(np.float64)), (
            f"SDK joint target differs from model prediction: sent={sent_target}, model={model_prediction}"
        )
        logging.debug("arm target fidelity: exact; sent_target=%s", sent_target.tolist())
        status = self.robot.set_joint_commands(commands, list(ARM_GROUPS), [], 0.0)
        if getattr(status, "name", "SUCCESS") != "SUCCESS":
            logging.warning("arm command returned %s", status)
            return False
        if self.grippers is not None:
            self.grippers.send(self.robot, target, now)
            logging.debug(
                "gripper target fidelity: exact; sent_target=%s",
                [float(target[group][0, 0] / MODEL_GRIPPER_UNITS_PER_METER) for group in GRIPPER_GROUPS],
            )
        return True


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
    server = response.get("server_timing_ms", {})
    logging.info(
        "Inference transport: %.2f MiB request, %.0fms round-trip "
        "(pack %.0fms, send %.0fms, server decode %.0fms, policy %.0fms)",
        metrics["request_bytes"] / (1024 * 1024),
        metrics["round_trip_ms"],
        metrics["pack_ms"],
        metrics["send_ms"],
        server.get("decode_ms", 0),
        server.get("policy_ms", 0),
    )


def add_arguments(parser, rtc):
    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--lang", default="fold clothes")
    parser.add_argument("--camera_sensors", default=",".join(CAMERA_ORDER))
    parser.add_argument("--hz", type=float, default=10.0, help="Training dataset frequency is 30 Hz")
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
        "--prefetch_step", type=int, default=-1,
        help="Synchronous mode: capture/send the next request at this chunk step; -1 uses the midpoint",
    )
    if not rtc:
        parser.set_defaults(hz=10.0)


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
    logging.info("Connected: %s", worker.metadata)
    logging.info("Image transport: %s", "SDK-compressed" if encoded_images else "raw RGB")
    next_observation = capture_observation_history(
        robot, sensors, args.lang, obs_horizon, args.hz, encoded_images=encoded_images,
    )
    worker.request(next_observation)
    try:
        for _ in range(args.max_steps):
            waited_at = time.monotonic()
            _, data = worker.wait()
            wait_ms = (time.monotonic() - waited_at) * 1000
            plan = make_plan(data)
            prefetch_step = len(plan) // 2 if args.prefetch_step < 0 else args.prefetch_step
            prefetch_step = min(prefetch_step, len(plan) - 1)
            logging.info(
                "Action chunk: %d steps (%.2fs at %.1fHz); waited %.0fms for prefetched result",
                len(plan), len(plan) / args.hz, args.hz, wait_ms,
            )
            for step, row in enumerate(plan):
                started = time.monotonic()
                current = get_observation(robot, sensors, args.lang, include_images=False)["state"]
                controller.execute(split_action(row), current, time.monotonic())
                if step == prefetch_step:
                    # Capture after part of the current plan has run, then infer the next
                    # chunk while the remaining robot commands are being consumed.
                    next_observation = capture_observation_history(
                        robot, sensors, args.lang, obs_horizon, args.hz,
                        encoded_images=encoded_images,
                    )
                    worker.request(next_observation)
                time.sleep(max(0, 1 / args.hz - (time.monotonic() - started)))
    finally:
        worker.close()


def run_rtc(args, robot, sensors, controller):
    worker = AsyncPolicyWorker(args.server_ip, args.port)
    logging.info("Connected: %s", worker.metadata)
    obs_horizon = int(worker.metadata.get("obs_horizon", 1))
    plan = np.empty((0, 16), dtype=np.float32)
    pending = None
    history = []
    try:
        for _ in range(args.max_steps):
            started = time.monotonic()
            history, observation = update_observation_history(
                history, get_observation(robot, sensors, args.lang), obs_horizon,
            )
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
    if not 0 < args.gripper_speed_mps <= 0.2:
        parser.error("G1 requires 0 < --gripper_speed_mps <= 0.2")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    robot, sensors, joint_command = create_robot(args)
    controller = RobotControl(robot, joint_command, args)
    logging.info("10Hz chunk execution; arm targets are sent exactly as model predicted")
    try:
        (run_rtc if rtc else run_sync)(args, robot, sensors, controller)
    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
