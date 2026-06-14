import os
import json
import math
import shutil
from pathlib import Path

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


DATASET_GENERATOR_VERSION = "2026-06-14-multitask-v1-push-pick-place"
SCRIPT_DIR = Path(__file__).resolve().parent


class DatasetLogger:
    """
    Raw dataset logger.
    Saves:
      dataset_root/
        episode_000001/
          frame_000000.png
          frame_000001.png
          ...
          meta.json
    """
    def __init__(self, root_dir="dataset_raw", keep_failed=False):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed = keep_failed
        self.episode_dir = None
        self.meta = None

    def start_episode(
        self,
        episode_id,
        instruction,
        goal_xy,
        box_init_xy,
        box_init_yaw,
        task_type="pick",
        target_color=None,
        target_object_type=None,
        target_body_name=None,
        all_object_init_poses=None,
        collection_config=None,
    ):
        episode_name = f"episode_{episode_id:06d}"
        self.episode_dir = self.root_dir / episode_name
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.meta = {
            "episode_id": int(episode_id),
            "instruction": str(instruction),
            "task_type": str(task_type),
            # grasp-only에서는 별도 place goal이 없으므로 초기 box 위치를 goal_xy로 둔다.
            # 기존 intermediate/RLDS 변환 코드와 호환되도록 2차원 필드는 유지한다.
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "box_init_xy": [float(box_init_xy[0]), float(box_init_xy[1])],
            "box_init_yaw": float(box_init_yaw),
            "success": False,
            "steps": []
        }

        if target_color is not None:
            self.meta["target_color"] = str(target_color)
        if target_object_type is not None:
            self.meta["target_object_type"] = str(target_object_type)
        if target_body_name is not None:
            self.meta["target_body_name"] = str(target_body_name)
        if all_object_init_poses is not None:
            self.meta["all_object_init_poses"] = all_object_init_poses
        self.meta["dataset_generator_version"] = DATASET_GENERATOR_VERSION
        if collection_config is not None:
            self.meta["collection_config"] = collection_config

        self.meta["grasp_diagnostics"] = []

    def log_grasp_diagnostic(self, diagnostic):
        self.meta["grasp_diagnostics"].append(diagnostic)

    def log_step(
        self,
        step_idx,
        image_rgb,
        joint_angles,
        gripper_state,
        object_pose,
        ee_pose,
        action,
        is_first=False,
        is_last=False,
    ):
        image_file = f"frame_{step_idx:06d}.png"
        image_path = self.episode_dir / image_file
        Image.fromarray(image_rgb).save(image_path)

        step_data = {
            "t": int(step_idx),
            "image_file": image_file,
            "joint_angles": [float(x) for x in joint_angles],
            "gripper_state": float(gripper_state),
            "object_pose": [float(x) for x in object_pose],
            "ee_pose": [float(x) for x in ee_pose],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        }
        self.meta["steps"].append(step_data)

    def finalize_episode(self, success, exception_text=None):
        self.meta["success"] = bool(success)
        if exception_text is not None:
            self.meta["exception"] = str(exception_text)

        meta_path = self.episode_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

        if (not success) and (not self.keep_failed):
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def abort_episode(self):
        if self.episode_dir is not None and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class SyncSimRaccoonDataset:
    """
    Synchronous MuJoCo dataset collector for RaccoonBot.

    Key design choices:
    - No background simulation thread
    - No real-time sleep-based settling
    - Main loop only: command -> run N mj_step -> render/save
    - Safe with viewer=False (physics still advances)
    """

    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    # Closing the lightweight closed-chain gripper too quickly creates large
    # constraint impulses at first contact and can trigger BADQACC.
    GRIPPER_SPEED = 2.5

    # Uploaded move_to code style uses centimeter-scale IK constants.
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0

    MODE_POSITION = 0
    MODE_VELOCITY = 1

    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85
    GRIP_CONTACT_SQUEEZE_BY_OBJECT_TYPE = {
        "cylinder": 0.020,
        "cube": 0.012,
        "sphere": 0.008,
    }
    GRIP_CONTACT_MIN_QPOS_BY_OBJECT_TYPE = {
        "cylinder": -0.30,
        "cube": -0.25,
        "sphere": -0.25,
    }
    GRASP_HEIGHT_BY_OBJECT_TYPE = {
        "cylinder": 0.020,
        "cube": 0.018,
        "sphere": 0.018,
    }

    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1
    GRIP_MODE_VERT = 2

    CYLINDER_BODY_BY_COLOR = {
        "red": "target_object",
        "blue": "target_object_blue",
        "green": "target_object_green",
        "yellow": "target_object_yellow",
    }
    CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())
    EXPECTED_RGBA_BY_COLOR = {
        "red": (1.0, 0.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
    }

    SUPPORTED_OBJECT_TYPES = ("cylinder", "cube", "sphere")
    DEFAULT_INSTRUCTION_TEMPLATES = (
        "grasp the {color} {object}",
        "pick up the {color} {object}",
        "grab the {color} {object}",
        "hold the {color} {object}",
        "move to the {color} {object} and grasp it",
    )
    MULTITASK_INSTRUCTION_TEMPLATES = {
        "push": (
            "push the {color} {object} forward",
            "move the {color} {object} forward",
            "slide the {color} {object} away from the robot",
            "nudge the {color} {object} forward",
        ),
        "pick_and_place": (
            "pick up the {color} {object} and place it in the drop zone",
            "move the {color} {object} to the drop zone",
            "grasp the {color} {object} and put it down in the target area",
            "relocate the {color} {object} to the drop zone",
        ),
    }

    # MuJoCo geom sizes:
    # - cylinder: [radius, halfheight, 0]
    # - box/cube: [x_half, y_half, z_half]
    # - sphere: [radius, 0, 0]
    OBJECT_GEOM_CONFIGS = {
        "cylinder": {
            "geom_type": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "size": (0.0075, 0.0100, 0.0),
            "z": 0.011,
            "mass": 0.004,
        },
        "cube": {
            "geom_type": mujoco.mjtGeom.mjGEOM_BOX,
            "size": (0.0090, 0.0090, 0.0090),
            "z": 0.010,
            "mass": 0.004,
        },
        "sphere": {
            "geom_type": mujoco.mjtGeom.mjGEOM_SPHERE,
            "size": (0.0100, 0.0, 0.0),
            "z": 0.011,
            "mass": 0.004,
        },
    }

    # Workspace used when all four colored objects are visible at once.
    # Compared with the previous x=(-0.18, 0.18), y=(0.10, 0.18), this keeps
    # objects slightly farther forward and more centered left-to-right.
    DEFAULT_OBJECT_X_RANGE = (-0.12, 0.12)
    DEFAULT_OBJECT_Y_RANGE = (0.14, 0.22)
    DEFAULT_MIN_OBJECT_DISTANCE = 0.045
    DEFAULT_TARGET_CLEARANCE = 0.070
    GRIPPER_PAD_SITE_NAMES = ("L_touch_site", "R_touch_site")

    @staticmethod
    def _resolve_xml_path(xml_path):
        """Resolve bundled XML defaults from this script directory, not caller cwd."""
        xml_path = Path(xml_path)
        if xml_path.is_absolute():
            return xml_path

        script_dir_path = Path(__file__).resolve().parent / xml_path
        if script_dir_path.exists():
            return script_dir_path

        return xml_path

    def __init__(
        self,
        xml_path,
        image_size=(256, 256),
        camera_name=None,
        use_viewer=False,
        enable_grasp_stabilizer=False,
    ):
        xml_path = self._resolve_xml_path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.validate_scene_color_mapping(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer
        self.enable_grasp_stabilizer = bool(enable_grasp_stabilizer)

        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.gripper_close_requested = False
        self.gripper_contact_hold_target = None
        self.grasp_stabilizer_active = False
        self.grasp_stabilizer_offset = None
        self.grasp_stabilizer_min_z = None
        self.grasp_contact_confirmed = False
        self.grasp_contact_source = None
        self.gripper_mode = self.GRIP_MODE_FREE
        self.active_object_body_name = self.CYLINDER_BODY_BY_COLOR["red"]
        self.active_object_type = "cylinder"

        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

        # Initialize all colored cylinders in the scene. Dataset collection will
        # randomize these positions for every episode.
        self.reset_episode(
            object_specs=self.make_default_object_specs(),
            target_color="red",
        )

    # ---------- kinematics / commands ----------

    def _calc_inv_kinematics(self, x, y, z):
        """
        Inputs are in centimeters, matching the uploaded move_to code style.
        Returns [j1, j2, j3, j4] in degrees.
        """
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)):
            if (-28.0 <= x <= 28.0) and (-15 <= y <= 28.0) and (0 <= z <= 36.25):
                x, y = y, -x
                th1 = math.atan2(y, x)
                c1 = math.cos(th1)
                s1 = math.sin(th1)
                x = x - self.L4 * c1
                y = y - self.L4 * s1
                zL1 = z - self.L1
                c3 = (x * x + y * y + zL1 * zL1 - self.L2 * self.L2 - self.L3 * self.L3) / (2 * self.L2 * self.L3)
                c32 = c3 * c3
                if c32 > 1:
                    c32 = 1
                s3 = -math.sqrt(1 - c32)
                th3 = math.atan2(s3, c3)
                M1 = c3 * self.L3 + self.L2
                M2 = z - self.L1
                M3 = s3 * self.L3
                M4 = c1 * x + s1 * y
                c2 = M1 * M2 - M3 * M4
                s2 = -M2 * M3 - M1 * M4
                th2 = math.atan2(s2, c2)
                th1 = math.degrees(th1)
                th2 = math.degrees(th2)
                th3 = math.degrees(th3)
                th4 = -(th2 + th3) - 90

                if th1 < -120 or th1 > 120:
                    return None
                if th2 < -90 or th2 > 30:
                    return None
                if th3 < -150 or th3 > 0:
                    return None

                return [th1, th2, th3, th4]
            return None
        return None

    def degree_to(self, joints, degrees, speed=70):
        j_list = joints if isinstance(joints, (list, tuple)) else [joints]
        d_list = degrees if isinstance(degrees, (list, tuple)) else [degrees]

        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)

        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                percent = np.clip(speed, 0.0, 100.0)
                self.joint_velocities[idx] = (percent / 100.0) * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm, y_cm, z_cm, speed=70):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표입니다: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self):
        self.gripper_target = self.GRIP_OPEN
        self.gripper_close_requested = False
        self.gripper_contact_hold_target = None
        self.grasp_stabilizer_active = False
        self.grasp_stabilizer_offset = None
        self.grasp_stabilizer_min_z = None
        self.grasp_contact_confirmed = False
        self.grasp_contact_source = None

    def close_gripper(self):
        if not self.gripper_close_requested:
            self.gripper_target = self.GRIP_CLOSE
            self.gripper_contact_hold_target = None
        self.gripper_close_requested = True

    def _target_has_bilateral_finger_contact(self):
        if not getattr(self, "active_object_body_name", None):
            return False
        left_contact, right_contact = self.get_target_finger_contact_state(
            self.active_object_body_name
        )
        return bool(left_contact and right_contact)

    def _latch_gripper_on_contact(self):
        """Latch one hold target instead of ratcheting closed on every physics step."""
        if not self.gripper_close_requested:
            return False
        if self.gripper_contact_hold_target is not None:
            self.gripper_target = self.gripper_contact_hold_target
            return True

        if not self._target_has_bilateral_finger_contact():
            return False

        object_type = getattr(self, "active_object_type", "cylinder")
        min_contact_qpos = self.GRIP_CONTACT_MIN_QPOS_BY_OBJECT_TYPE.get(
            object_type,
            self.GRIP_CONTACT_MIN_QPOS_BY_OBJECT_TYPE["cylinder"],
        )
        if float(self.data.qpos[4]) > float(min_contact_qpos):
            return False
        squeeze = self.GRIP_CONTACT_SQUEEZE_BY_OBJECT_TYPE.get(
            object_type,
            self.GRIP_CONTACT_SQUEEZE_BY_OBJECT_TYPE["cylinder"],
        )
        hold_target = max(self.GRIP_CLOSE, float(self.data.qpos[4]) - float(squeeze))
        self.gripper_contact_hold_target = hold_target
        self.gripper_target = hold_target
        self.grasp_contact_confirmed = True
        self.grasp_contact_source = "target_bilateral_finger"
        if getattr(self, "enable_grasp_stabilizer", False) and not self._activate_grasp_stabilizer():
            self.gripper_contact_hold_target = None
            self.gripper_target = self.GRIP_CLOSE
            self.grasp_contact_confirmed = False
            self.grasp_contact_source = None
            return False
        return True

    def _get_freejoint_addresses(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1 or int(self.model.body_jntnum[body_id]) < 1:
            raise ValueError(f"freejoint body not found: {body_name}")
        joint_id = int(self.model.body_jntadr[body_id])
        if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"body does not use a freejoint: {body_name}")
        return (
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def _activate_grasp_stabilizer(self):
        """Center an already-contacted target between the physical finger pads."""
        if self.grasp_stabilizer_active:
            return True
        if not self._target_has_bilateral_finger_contact():
            return False
        object_position = np.asarray(
            self.get_object_pose(self.active_object_body_name)[:3],
            dtype=np.float64,
        )
        self.grasp_stabilizer_offset = np.zeros(3, dtype=np.float64)
        self.grasp_stabilizer_min_z = float(object_position[2])
        self.grasp_stabilizer_active = True
        self.grasp_contact_confirmed = True
        self.grasp_contact_source = "target_bilateral_finger"
        return True

    def get_gripper_pad_midpoint(self):
        site_positions = []
        for site_name in self.GRIPPER_PAD_SITE_NAMES:
            site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_name,
            )
            if site_id == -1:
                raise ValueError(f"gripper pad site not found: {site_name}")
            site_positions.append(
                np.asarray(self.data.site_xpos[site_id], dtype=np.float64)
            )
        return np.mean(site_positions, axis=0)

    def _apply_grasp_stabilizer(self):
        if (
            not getattr(self, "enable_grasp_stabilizer", False)
            or not self.grasp_stabilizer_active
            or self.grasp_stabilizer_offset is None
        ):
            return

        qpos_adr, qvel_adr = self._get_freejoint_addresses(
            self.active_object_body_name
        )
        desired_position = (
            self.get_gripper_pad_midpoint()
            + np.asarray(self.grasp_stabilizer_offset, dtype=np.float64)
        )
        if self.grasp_stabilizer_min_z is not None:
            desired_position[2] = max(
                float(desired_position[2]),
                float(self.grasp_stabilizer_min_z),
            )
        self.data.qpos[qpos_adr:qpos_adr + 3] = desired_position
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def lockh(self):
        self.gripper_mode = self.GRIP_MODE_HORZ

    def lockv(self):
        self.gripper_mode = self.GRIP_MODE_VERT

    def unlock(self):
        if self.gripper_mode != self.GRIP_MODE_FREE:
            self.target_angles[3] = self.data.qpos[3]
            self.gripper_mode = self.GRIP_MODE_FREE

    def execute_action(self, action, speed=70):
        """
        action = [target_x_m, target_y_m, target_z_m, gripper]
        """
        target_x, target_y, target_z, gripper = action

        # move_to convention is centimeters.
        self.move_to(target_x * 100.0, target_y * 100.0, target_z * 100.0, speed=speed)

        if gripper >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    # ---------- synchronous stepping ----------

    def _apply_controls_once(self):
        dt = self.model.opt.timestep

        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                if self.gripper_mode == self.GRIP_MODE_HORZ:
                    desired = base_angle - np.radians(90)
                else:
                    desired = base_angle - np.radians(180)

                error = desired - self.current_setpoints[i]
                speed_rad_s = self.MAX_SPEEDS[i]
                limit_step = speed_rad_s * dt
                step = np.clip(error, -limit_step, limit_step)
                self.current_setpoints[i] += step
            else:
                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt
                        step_val = np.clip(error, -max_step, max_step)
                        self.current_setpoints[i] += step_val

            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        self._latch_gripper_on_contact()

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            g_move = np.clip(g_err, -g_step, g_step)
            self.current_setpoints[4] += g_move

        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps):
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            instability_reason = self.get_simulation_instability_reason()
            if instability_reason is not None:
                raise RuntimeError(f"MuJoCo simulation unstable: {instability_reason}")
            self._apply_grasp_stabilizer()
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def get_simulation_instability_reason(self):
        """Return a reason when MuJoCo reports or contains invalid dynamics state."""
        for field_name in ("qpos", "qvel", "qacc"):
            values = np.asarray(getattr(self.data, field_name), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                return f"non-finite {field_name}"

        warning_names = (
            ("BADQPOS", mujoco.mjtWarning.mjWARN_BADQPOS),
            ("BADQVEL", mujoco.mjtWarning.mjWARN_BADQVEL),
            ("BADQACC", mujoco.mjtWarning.mjWARN_BADQACC),
        )
        for warning_name, warning_type in warning_names:
            warning = self.data.warning[int(warning_type)]
            if int(warning.number) > 0:
                return f"{warning_name} warning count={int(warning.number)}"
        return None

    def steps_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds=2.0):
        self.step_n(self.steps_for_seconds(seconds))

    # ---------- rendering / state ----------

    def get_robot_state(self):
        joint_angles = [float(self.data.qpos[i]) for i in range(4)]
        gripper_state = float(self.data.qpos[4])
        return {
            "joint_angles": joint_angles,
            "gripper_state": gripper_state
        }

    def get_object_pose(self, body_name="target_object"):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])

        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def get_ee_pose(self):
        """
        Return the gripper-tip position in meters using the same FK convention
        as move_to()/IK and the inference client.

        Link4 body xpos is the wrist position and is about L4 (8 cm) behind the
        commanded endpoint, so using it to build delta actions creates a
        train/inference coordinate mismatch.
        """
        th1 = float(self.data.qpos[0])
        th2 = float(self.data.qpos[1])
        th3 = float(self.data.qpos[2])

        r = -self.L2 * math.sin(th2) - self.L3 * math.sin(th2 + th3)
        z = self.L1 + self.L2 * math.cos(th2) + self.L3 * math.cos(th2 + th3)
        r_tip = r + self.L4

        x_cm = -math.sin(th1) * r_tip
        y_cm = math.cos(th1) * r_tip
        z_cm = z
        return x_cm / 100.0, y_cm / 100.0, z_cm / 100.0

    def render_rgb(self):
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        image = self.renderer.render()
        return image.copy()

    def get_observation(self, object_body_name=None):
        if object_body_name is None:
            object_body_name = self.active_object_body_name

        rs = self.get_robot_state()
        obj = self.get_object_pose(object_body_name)
        img = self.render_rgb()
        ee_pose_list = list(self.get_ee_pose())

        return {
            "image": img,
            "joint_angles": rs["joint_angles"],
            "gripper_state": rs["gripper_state"],
            "object_pose": obj,
            "ee_pose": ee_pose_list,
        }

    # ---------- reset / success ----------

    def reset_object_pose(self, body_name="target_object", x=0.15, y=0.15, z=0.02, yaw=0.0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num < 1:
            raise ValueError(f"{body_name} has no joint")

        joint_id = jnt_adr
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # freejoint qpos = [x, y, z, qw, qx, qy, qz]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

        # Zero object joint velocities if present.
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def set_object_geom_type(self, body_name, object_type):
        """
        Reuse the existing colored object bodies while changing their geom shape.

        This keeps the XML backwards compatible: body/freejoint names stay the
        same, while each episode can contain cylinder/cube/sphere variants.
        """
        object_type = str(object_type)
        if object_type not in self.OBJECT_GEOM_CONFIGS:
            raise ValueError(
                f"지원하지 않는 object_type입니다: {object_type}. "
                f"지원 object_type: {list(self.OBJECT_GEOM_CONFIGS)}"
            )

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        geom_count = int(self.model.body_geomnum[body_id])
        if geom_count < 1:
            raise ValueError(f"{body_name} has no geom")

        geom_id = int(self.model.body_geomadr[body_id])
        geom_cfg = self.OBJECT_GEOM_CONFIGS[object_type]
        self.model.geom_type[geom_id] = int(geom_cfg["geom_type"])
        self.model.geom_size[geom_id, :3] = np.asarray(geom_cfg["size"], dtype=np.float64)
        self._set_object_mass_and_inertia(body_id, object_type)

    def _set_object_mass_and_inertia(self, body_id, object_type):
        """Use realistic gram-scale mass and matching diagonal inertia."""
        geom_cfg = self.OBJECT_GEOM_CONFIGS[object_type]
        mass = float(geom_cfg["mass"])
        size = geom_cfg["size"]

        if object_type == "cylinder":
            radius = float(size[0])
            height = 2.0 * float(size[1])
            i_xy = mass * (3.0 * radius * radius + height * height) / 12.0
            inertia = (i_xy, i_xy, 0.5 * mass * radius * radius)
        elif object_type == "cube":
            x, y, z = (2.0 * float(value) for value in size)
            inertia = (
                mass * (y * y + z * z) / 12.0,
                mass * (x * x + z * z) / 12.0,
                mass * (x * x + y * y) / 12.0,
            )
        else:
            radius = float(size[0])
            sphere_inertia = 0.4 * mass * radius * radius
            inertia = (sphere_inertia, sphere_inertia, sphere_inertia)

        self.model.body_mass[body_id] = mass
        self.model.body_inertia[body_id, :3] = np.asarray(inertia, dtype=np.float64)

    @classmethod
    def validate_scene_color_mapping(cls, model):
        """Fail fast when XML body names or visual colors disagree with the dataset labels."""
        for color, expected_body_name in cls.CYLINDER_BODY_BY_COLOR.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, expected_body_name)
            if body_id == -1:
                raise ValueError(
                    f"scene color mapping error: color='{color}' body='{expected_body_name}' not found"
                )

            geom_count = int(model.body_geomnum[body_id])
            if geom_count != 1:
                raise ValueError(
                    f"scene color mapping error: body='{expected_body_name}' must have exactly one geom, "
                    f"found {geom_count}"
                )

            geom_id = int(model.body_geomadr[body_id])
            actual_rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64)
            expected_rgba = np.asarray(cls.EXPECTED_RGBA_BY_COLOR[color], dtype=np.float64)
            if not np.allclose(actual_rgba, expected_rgba, atol=1e-6):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "<unnamed>"
                raise ValueError(
                    f"scene color mapping error: color='{color}' body='{expected_body_name}' "
                    f"geom='{geom_name}' rgba={actual_rgba.tolist()} expected={expected_rgba.tolist()}"
                )

    @classmethod
    def validate_object_specs(cls, object_specs):
        """Ensure each semantic color controls its one canonical MuJoCo body."""
        if not object_specs:
            raise ValueError("object_specs는 비어 있을 수 없습니다.")

        seen_bodies = set()
        for color, spec in object_specs.items():
            if color not in cls.CYLINDER_BODY_BY_COLOR:
                raise ValueError(f"지원하지 않는 색상입니다: {color}")

            expected_body_name = cls.CYLINDER_BODY_BY_COLOR[color]
            actual_body_name = spec.get("body_name")
            if actual_body_name != expected_body_name:
                raise ValueError(
                    f"object_specs mapping error: color='{color}' must map to "
                    f"body='{expected_body_name}', got '{actual_body_name}'"
                )
            if actual_body_name in seen_bodies:
                raise ValueError(f"object_specs mapping error: duplicate body='{actual_body_name}'")
            seen_bodies.add(actual_body_name)

    @classmethod
    def make_default_object_specs(cls):
        """
        Deterministic fallback placement for initialization only.
        Dataset collection uses sample_object_specs() for randomized positions.
        """
        x_values = np.linspace(
            cls.DEFAULT_OBJECT_X_RANGE[0] * 0.75,
            cls.DEFAULT_OBJECT_X_RANGE[1] * 0.75,
            len(cls.CYLINDER_COLORS),
        )
        y_center = float(sum(cls.DEFAULT_OBJECT_Y_RANGE) / 2.0)
        return {
            color: {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                "object_type": "cylinder",
                "x": float(x_values[idx]),
                "y": y_center,
                "yaw": 0.0,
                "placement_rank": idx,
            }
            for idx, color in enumerate(cls.CYLINDER_COLORS)
        }

    @classmethod
    def sample_object_specs(
        cls,
        rng,
        colors=None,
        object_types=None,
        forced_target=None,
        x_range=None,
        y_range=None,
        yaw_range=(-np.pi / 4, np.pi / 4),
        min_distance=None,
        target_clearance=None,
        max_tries=1000,
    ):
        """
        Randomly place all colored objects in the visible workspace.

        Defaults intentionally narrow the spawn area compared with the older
        single-object collector:
          - x: -0.18~0.18  ->  -0.10~0.10
          - y:  0.10~0.18  ->   0.16~0.20
        A minimum XY distance prevents blocks from overlapping or touching.
        """
        colors = tuple(colors or cls.CYLINDER_COLORS)
        object_types = tuple(object_types or ("cylinder",))
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or cls.DEFAULT_OBJECT_Y_RANGE
        min_distance = cls.DEFAULT_MIN_OBJECT_DISTANCE if min_distance is None else min_distance
        target_clearance = (
            cls.DEFAULT_TARGET_CLEARANCE
            if target_clearance is None
            else float(target_clearance)
        )

        if len(colors) == 0:
            raise ValueError("colors는 비어 있을 수 없습니다.")
        if len(object_types) == 0:
            raise ValueError("object_types는 비어 있을 수 없습니다.")
        unknown_object_types = [obj for obj in object_types if obj not in cls.SUPPORTED_OBJECT_TYPES]
        if unknown_object_types:
            raise ValueError(
                f"지원하지 않는 object_types입니다: {unknown_object_types}. "
                f"지원 object_types: {list(cls.SUPPORTED_OBJECT_TYPES)}"
            )
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

        forced_target = forced_target or {}
        forced_color = forced_target.get("color")
        forced_object_type = forced_target.get("object_type")
        forced_x_range = forced_target.get("x_range")
        forced_y_range = forced_target.get("y_range")
        if forced_color is not None and forced_color not in colors:
            raise ValueError(f"forced target color={forced_color}가 colors={colors}에 없습니다.")
        if forced_object_type is not None and forced_object_type not in object_types:
            raise ValueError(f"forced object_type={forced_object_type}가 object_types={object_types}에 없습니다.")
        if forced_x_range is not None and forced_x_range[0] >= forced_x_range[1]:
            raise ValueError(f"잘못된 forced target x_range입니다: {forced_x_range}")
        if forced_y_range is not None and forced_y_range[0] >= forced_y_range[1]:
            raise ValueError(f"잘못된 forced target y_range입니다: {forced_y_range}")

        specs = {}
        placed_xy = []
        # Place a forced target first so its larger clearance is guaranteed
        # during sampling instead of rejecting an otherwise complete layout.
        placement_order = list(colors)
        rng.shuffle(placement_order)
        if forced_color is not None:
            placement_order.remove(forced_color)
            placement_order.insert(0, forced_color)

        for placement_rank, color in enumerate(placement_order):
            if color not in cls.CYLINDER_BODY_BY_COLOR:
                raise ValueError(f"지원하지 않는 색상입니다: {color}")

            for _ in range(max_tries):
                sample_x_range = (
                    forced_x_range
                    if color == forced_color and forced_x_range is not None
                    else x_range
                )
                sample_y_range = (
                    forced_y_range
                    if color == forced_color and forced_y_range is not None
                    else y_range
                )
                x = float(rng.uniform(sample_x_range[0], sample_x_range[1]))
                y = float(rng.uniform(sample_y_range[0], sample_y_range[1]))
                xy = np.array([x, y], dtype=np.float64)

                required_distances = [
                    (
                        target_clearance
                        if forced_color is not None
                        and (color == forced_color or other_color == forced_color)
                        else min_distance
                    )
                    for other_color, _ in placed_xy
                ]
                if all(
                    np.linalg.norm(xy - other_xy) >= required_distance
                    for (_, other_xy), required_distance in zip(
                        placed_xy,
                        required_distances,
                    )
                ):
                    specs[color] = {
                        "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                        "object_type": (
                            str(forced_object_type)
                            if color == forced_color and forced_object_type is not None
                            else str(rng.choice(object_types))
                        ),
                        "x": x,
                        "y": y,
                        "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                        "placement_rank": placement_rank,
                    }
                    placed_xy.append((color, xy))
                    break
            else:
                raise RuntimeError(
                    "색상 object들을 겹치지 않게 배치하지 못했습니다. "
                    f"x_range={x_range}, y_range={y_range}, min_distance={min_distance}를 확인하세요."
                )

        # Return in canonical color order for stable metadata.
        return {color: specs[color] for color in colors}

    @staticmethod
    def specs_to_meta(object_specs):
        return {
            color: {
                "body_name": str(spec["body_name"]),
                "object_type": str(spec.get("object_type", "cylinder")),
                "xy": [float(spec["x"]), float(spec["y"])],
                "yaw": float(spec["yaw"]),
                "placement_rank": int(spec.get("placement_rank", -1)),
            }
            for color, spec in object_specs.items()
        }

    @staticmethod
    def validate_target_clearance(object_specs, target_color, min_clearance):
        """Keep distractors outside the swept volume around the grasp target."""
        if target_color not in object_specs:
            raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")
        if min_clearance <= 0:
            raise ValueError("min_clearance는 0보다 커야 합니다.")

        target_spec = object_specs[target_color]
        target_xy = np.asarray(
            [target_spec["x"], target_spec["y"]],
            dtype=np.float64,
        )
        distances = {}
        for color, spec in object_specs.items():
            if color == target_color:
                continue
            other_xy = np.asarray([spec["x"], spec["y"]], dtype=np.float64)
            distance = float(np.linalg.norm(other_xy - target_xy))
            distances[color] = distance
            if distance < float(min_clearance):
                raise ValueError(
                    "target clearance violation: "
                    f"target='{target_color}' distractor='{color}' "
                    f"distance={distance:.4f}m required={float(min_clearance):.4f}m"
                )
        return distances

    def reset_colored_objects(self, object_specs, target_color):
        """
        Place every colored object in the scene. The target color controls
        which body is used for object_pose logging and grasp trajectory target.
        """
        if target_color not in object_specs:
            raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")
        self.validate_object_specs(object_specs)

        self.active_object_body_name = object_specs[target_color]["body_name"]
        self.active_object_type = object_specs[target_color].get("object_type", "cylinder")

        # Update every model geom first. Calling mj_setConst between individual
        # object placements resets MjData and silently erases poses already set
        # earlier in this loop.
        for color, spec in object_specs.items():
            body_name = spec["body_name"]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"body not found for color '{color}': {body_name}")

            object_type = spec.get("object_type", "cylinder")
            self.set_object_geom_type(body_name, object_type)

        # Recompute model constants once, before writing any episode poses.
        try:
            mujoco.mj_setConst(self.model, self.data)
        except Exception:
            pass

        for spec in object_specs.values():
            body_name = spec["body_name"]
            object_type = spec.get("object_type", "cylinder")
            z = float(self.OBJECT_GEOM_CONFIGS[object_type]["z"])

            self.reset_object_pose(
                body_name,
                x=spec["x"],
                y=spec["y"],
                z=z,
                yaw=spec["yaw"],
            )

    def get_non_target_object_positions(self, object_specs, target_color):
        return {
            color: np.asarray(
                self.get_object_pose(spec["body_name"])[:3],
                dtype=np.float64,
            )
            for color, spec in object_specs.items()
            if color != target_color
        }

    def validate_non_target_objects_undisturbed(
        self,
        object_specs,
        target_color,
        initial_positions,
        max_displacement=0.003,
    ):
        """Reject demonstrations that touch or move any distractor object."""
        non_target_bodies = {
            spec["body_name"]
            for color, spec in object_specs.items()
            if color != target_color
        }

        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            body1_id = int(self.model.geom_bodyid[geom1_id])
            body2_id = int(self.model.geom_bodyid[geom2_id])
            body1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id
            ) or f"body_{body1_id}"
            body2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id
            ) or f"body_{body2_id}"
            if (
                (body1 in non_target_bodies or body2 in non_target_bodies)
                and body1 != "world"
                and body2 != "world"
            ):
                raise RuntimeError(
                    "non-target object contact detected: "
                    f"{body1}<->{body2}"
                )

        current_positions = self.get_non_target_object_positions(
            object_specs,
            target_color,
        )
        for color, initial_position in initial_positions.items():
            # Distractor disturbance is a table-plane event. Small residual
            # vertical settling after reset must not be mistaken for a collision.
            displacement = float(
                np.linalg.norm(current_positions[color][:2] - initial_position[:2])
            )
            if displacement > float(max_displacement) + 1e-12:
                raise RuntimeError(
                    "non-target object moved: "
                    f"color='{color}' xy_displacement={displacement:.6f}m "
                    f"limit={float(max_displacement):.6f}m"
                )

    def reset_episode(self, object_specs, target_color="red"):
        home = np.radians([0.0, -10.0, -140.0, 60.0])

        # Clear all MuJoCo runtime state before restoring the episode-specific
        # robot and object poses. Resetting qpos/qvel alone leaves solver
        # warm-start values, applied forces, contacts, and simulation time from
        # the previous attempt, which can accumulate into unstable QACC values.
        mujoco.mj_resetData(self.model, self.data)

        # Geometry changes call mj_setConst, which also reinitializes MjData.
        # Therefore place/configure all objects before restoring robot state.
        self.reset_colored_objects(object_specs=object_specs, target_color=target_color)

        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION

        self.data.qvel[:] = 0.0

        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_close_requested = False
        self.gripper_contact_hold_target = None
        self.grasp_stabilizer_active = False
        self.grasp_stabilizer_offset = None
        self.grasp_stabilizer_min_z = None
        self.grasp_contact_confirmed = False
        self.grasp_contact_source = None
        self.gripper_mode = self.GRIP_MODE_FREE

        mujoco.mj_forward(self.model, self.data)

        # Short stabilization after reset.
        self.step_n(20)

    def get_gripper_touch_state(self):
        """
        Return whether the left/right gripper touch sensors are in contact.
        If the XML does not expose these sensors, this returns False for both sides.
        """
        try:
            touch_l = float(self.data.sensor("sensor_L").data[0])
            touch_r = float(self.data.sensor("sensor_R").data[0])
        except Exception:
            touch_l = 0.0
            touch_r = 0.0

        return touch_l, touch_r

    def is_grasp_success(self, touch_threshold=0.1, require_closed=True):
        """
        Grasp-only success criterion.
        The episode is considered successful when both gripper touch sensors detect contact.
        Optionally also require the gripper to have moved away from its fully-open position.
        """
        touch_l, touch_r = self.get_gripper_touch_state()
        both_touched = (touch_l > touch_threshold) and (touch_r > touch_threshold)

        if not require_closed:
            return bool(both_touched)

        # Make sure this is not just an accidental touch while the gripper is still fully open.
        gripper_is_closing_or_closed = float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        return bool(both_touched and gripper_is_closing_or_closed)

    def is_body_touching_robot(self, body_name, ignored_geom_names=("floor",)):
        """
        Return True when the requested object body is in contact with a non-floor,
        non-cylinder body. This makes success target-specific when all four
        colored cylinders are present: touching the wrong color does not count.
        """
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target_body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        cylinder_body_ids = set()
        for cylinder_body_name in self.CYLINDER_BODY_BY_COLOR.values():
            cylinder_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cylinder_body_name)
            if cylinder_body_id != -1:
                cylinder_body_ids.add(cylinder_body_id)

        ignored_geom_names = set(ignored_geom_names or [])

        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            if target_body_id not in (body1, body2):
                continue

            other_geom = geom2 if body1 == target_body_id else geom1
            other_body = body2 if body1 == target_body_id else body1

            other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_geom_name in ignored_geom_names:
                continue

                # Do not count target-object contact with another colored object
            # as a grasp. We only want contacts against the robot/gripper.
            if other_body in cylinder_body_ids:
                continue

            return True

        return False

    def get_target_finger_contact_state(self, target_body_name):
        """Return whether the target contacts the left and right finger bodies."""
        left_contact = False
        right_contact = False
        for contact in self.get_target_contact_details(target_body_name):
            other_bodies = {contact["body1"], contact["body2"]} - {target_body_name}
            left_contact = left_contact or ("Gripper_L_Finger" in other_bodies)
            right_contact = right_contact or ("Gripper_R_Finger" in other_bodies)
        return bool(left_contact), bool(right_contact)

    def is_target_grasp_success(
        self,
        target_body_name,
        initial_target_z,
        grasp_contact_confirmed,
        grasp_contact_retained,
        min_lift_height=0.015,
        require_closed=True,
    ):
        """
        Require a physical lift, not only pad sensors or incidental contact.

        Grasp is confirmed by bilateral finger contact and must still have at
        least one target-finger contact at the close diagnostic waypoint.
        Pick-up is confirmed afterward by the target rising min_lift_height.
        """
        final_target_z = float(self.get_object_pose(target_body_name)[2])
        lifted = final_target_z >= (float(initial_target_z) + float(min_lift_height))
        gripper_closed = (
            (not require_closed)
            or float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        )
        return bool(
            grasp_contact_confirmed
            and grasp_contact_retained
            and gripper_closed
            and lifted
        )

    def get_target_contact_details(self, target_body_name):
        """Return JSON-safe contact details for contacts involving the target body."""
        target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, target_body_name
        )
        if target_body_id == -1:
            raise ValueError(f"body not found: {target_body_name}")

        contacts = []
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            body1_id = int(self.model.geom_bodyid[geom1_id])
            body2_id = int(self.model.geom_bodyid[geom2_id])
            if target_body_id not in (body1_id, body2_id):
                continue

            contacts.append({
                "contact_index": contact_index,
                "geom1": mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id
                ) or f"geom_{geom1_id}",
                "body1": mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id
                ) or f"body_{body1_id}",
                "geom2": mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id
                ) or f"geom_{geom2_id}",
                "body2": mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id
                ) or f"body_{body2_id}",
                "distance": float(contact.dist),
            })
        return contacts

    def get_grasp_diagnostic(
        self,
        target_body_name,
        stage,
        waypoint_index,
        touch_threshold=0.1,
        initial_target_z=None,
        min_lift_height=0.015,
    ):
        touch_l, touch_r = self.get_gripper_touch_state()
        object_pose = self.get_object_pose(target_body_name)
        target_contact = self.is_body_touching_robot(target_body_name)
        left_finger_contact, right_finger_contact = self.get_target_finger_contact_state(
            target_body_name
        )
        gripper_is_closing_or_closed = float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        lift_delta = (
            None
            if initial_target_z is None
            else float(object_pose[2]) - float(initial_target_z)
        )
        return {
            "stage": str(stage),
            "waypoint_index": int(waypoint_index),
            "simulation_time": float(self.data.time),
            "touch_left": float(touch_l),
            "touch_right": float(touch_r),
            "touch_left_pass": bool(touch_l > touch_threshold),
            "touch_right_pass": bool(touch_r > touch_threshold),
            "gripper_qpos": float(self.data.qpos[4]),
            "gripper_closed_pass": bool(gripper_is_closing_or_closed),
            "target_z": float(object_pose[2]),
            "target_robot_contact_pass": bool(target_contact),
            "left_finger_contact_pass": bool(left_finger_contact),
            "right_finger_contact_pass": bool(right_finger_contact),
            "grasp_stabilizer_active": bool(
                getattr(self, "grasp_stabilizer_active", False)
            ),
            "grasp_contact_confirmed": bool(
                getattr(self, "grasp_contact_confirmed", False)
            ),
            "grasp_contact_source": getattr(self, "grasp_contact_source", None),
            "lift_delta": lift_delta,
            "lift_pass": bool(
                lift_delta is not None and lift_delta >= float(min_lift_height)
            ),
            "contacts": self.get_target_contact_details(target_body_name),
        }

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            # Older MuJoCo releases (including the user's Python 3.8 runtime)
            # do not expose Renderer.close(). Drop the reference there and let
            # the renderer/context destructor perform cleanup.
            renderer_close = getattr(self.renderer, "close", None)
            if callable(renderer_close):
                renderer_close()
            self.renderer = None

    # ---------- task plans ----------

    def make_grasp_plan(self, object_x, object_y, object_type=None):
        object_type = str(object_type or getattr(self, "active_object_type", "cylinder"))
        if object_type not in self.SUPPORTED_OBJECT_TYPES:
            raise ValueError(f"지원하지 않는 object_type입니다: {object_type}")

        z_above = 0.10
        z_grasp = self.GRASP_HEIGHT_BY_OBJECT_TYPE[object_type]
        z_lift = 0.08

        return [
            [object_x, object_y, z_above, 0],   # Move above object with gripper open.
            [object_x, object_y, z_grasp, 0],   # Move down to grasp height.
            [object_x, object_y, z_grasp, 1],   # Close gripper around the object.
            [object_x, object_y, z_lift, 1],    # Lift while keeping the gripper closed.
        ]

    def make_push_plan(self, object_x, object_y, goal_xy, object_type=None):
        object_type = str(object_type or getattr(self, "active_object_type", "cylinder"))
        if object_type not in self.SUPPORTED_OBJECT_TYPES:
            raise ValueError(f"지원하지 않는 object_type입니다: {object_type}")

        goal_x, goal_y = (float(goal_xy[0]), float(goal_xy[1]))
        direction = np.asarray([goal_x - object_x, goal_y - object_y], dtype=np.float64)
        distance = float(np.linalg.norm(direction))
        if distance <= 0:
            raise ValueError("push goal은 object 위치와 달라야 합니다.")
        direction /= distance
        start_xy = np.asarray([object_x, object_y], dtype=np.float64) - direction * 0.025
        # A closed gripper commanded at 3 cm places the finger faces near the
        # object centroid. Reusing grasp height drives the fingers into the
        # floor and fails to produce a horizontal push.
        push_z = 0.030

        return [
            [float(start_xy[0]), float(start_xy[1]), 0.08, 0],
            [float(start_xy[0]), float(start_xy[1]), 0.08, 1],
            [float(start_xy[0]), float(start_xy[1]), push_z, 1],
            [goal_x, goal_y, push_z, 1],
            [goal_x, goal_y, 0.08, 0],
        ]

    def make_pick_and_place_plan(self, object_x, object_y, goal_xy, object_type=None):
        object_type = str(object_type or getattr(self, "active_object_type", "cylinder"))
        goal_x, goal_y = (float(goal_xy[0]), float(goal_xy[1]))
        z_grasp = self.GRASP_HEIGHT_BY_OBJECT_TYPE[object_type]
        z_lift = 0.08
        z_place = 0.025

        return [
            [object_x, object_y, 0.10, 0],
            [object_x, object_y, z_grasp, 0],
            [object_x, object_y, z_grasp, 1],
            [object_x, object_y, z_lift, 1],
            [goal_x, goal_y, z_lift, 1],
            [goal_x, goal_y, z_place, 1],
            [goal_x, goal_y, z_place, 0],
            [goal_x, goal_y, z_lift, 0],
        ]

    def make_task_plan(self, task_type, object_x, object_y, goal_xy, object_type=None):
        if task_type == "push":
            return self.make_push_plan(
                object_x,
                object_y,
                goal_xy,
                object_type=object_type,
            )
        if task_type == "pick_and_place":
            return self.make_pick_and_place_plan(
                object_x,
                object_y,
                goal_xy,
                object_type=object_type,
            )
        raise ValueError(f"지원하지 않는 task_type입니다: {task_type}")

    def validate_task_plan_ik(
        self,
        task_type,
        object_x,
        object_y,
        goal_xy,
        object_type=None,
    ):
        plan = self.make_task_plan(
            task_type,
            object_x,
            object_y,
            goal_xy,
            object_type=object_type,
        )
        checked_positions = set()
        previous_position = None
        for waypoint_index, action in enumerate(plan):
            position = tuple(float(value) for value in action[:3])
            positions_to_check = [position]
            if previous_position is not None:
                positions_to_check = self.interpolate_cartesian_segment(
                    previous_position,
                    position,
                    max_step=0.002,
                )
            for sample_index, sample in enumerate(positions_to_check):
                sampled_position = tuple(float(value) for value in sample)
                if sampled_position in checked_positions:
                    continue
                checked_positions.add(sampled_position)
                angles = self._calc_inv_kinematics(
                    *(value * 100.0 for value in sampled_position)
                )
                if angles is None:
                    raise ValueError(
                        f"{task_type} IK precheck failed: waypoint={waypoint_index} "
                        f"segment_sample={sample_index} "
                        f"xyz_m=({sampled_position[0]:.4f}, "
                        f"{sampled_position[1]:.4f}, {sampled_position[2]:.4f})"
                    )
            previous_position = position
        return plan

    @staticmethod
    def interpolate_cartesian_segment(start_xyz, target_xyz, max_step):
        """Return target-inclusive Cartesian points no farther than max_step apart."""
        start = np.asarray(start_xyz, dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64)
        if start.shape != (3,) or target.shape != (3,):
            raise ValueError("start_xyz와 target_xyz는 길이 3이어야 합니다.")
        if max_step <= 0:
            raise ValueError("max_step은 0보다 커야 합니다.")

        distance = float(np.linalg.norm(target - start))
        segment_count = max(1, int(math.ceil(distance / float(max_step))))
        return [
            (start + (target - start) * (index / segment_count)).tolist()
            for index in range(1, segment_count + 1)
        ]

    @staticmethod
    def next_cartesian_command(observed_xyz, target_xyz, max_step):
        """Advance from the observed pose, preventing command/robot drift."""
        observed = np.asarray(observed_xyz, dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64)
        if observed.shape != (3,) or target.shape != (3,):
            raise ValueError("observed_xyz와 target_xyz는 길이 3이어야 합니다.")
        if max_step <= 0:
            raise ValueError("max_step은 0보다 커야 합니다.")

        delta = target - observed
        distance = float(np.linalg.norm(delta))
        if distance <= float(max_step):
            return target.tolist()
        return (observed + delta * (float(max_step) / distance)).tolist()

    @staticmethod
    def is_pre_close_aligned(observed_xyz, target_xyz, tolerance):
        """Only allow closing after the open gripper has reached grasp height."""
        observed = np.asarray(observed_xyz, dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64)
        if observed.shape != (3,) or target.shape != (3,):
            raise ValueError("observed_xyz와 target_xyz는 길이 3이어야 합니다.")
        if tolerance <= 0:
            raise ValueError("tolerance는 0보다 커야 합니다.")
        return bool(np.linalg.norm(observed - target) <= float(tolerance))

    @staticmethod
    def validate_recorded_transition(
        previous_observation,
        next_observation,
        max_ee_delta=0.005,
        max_joint_delta=0.35,
    ):
        """Reject frames whose observed motion exceeds the inference-time contract."""
        previous_ee = np.asarray(previous_observation["ee_pose"][:3], dtype=np.float64)
        next_ee = np.asarray(next_observation["ee_pose"][:3], dtype=np.float64)
        ee_delta = next_ee - previous_ee
        ee_delta_norm = float(np.linalg.norm(ee_delta))

        previous_joint = np.asarray(previous_observation["joint_angles"], dtype=np.float64)
        next_joint = np.asarray(next_observation["joint_angles"], dtype=np.float64)
        joint_delta = next_joint - previous_joint
        max_abs_joint_delta = float(np.max(np.abs(joint_delta)))

        if not np.all(np.isfinite(ee_delta)) or not np.all(np.isfinite(joint_delta)):
            raise RuntimeError("recorded transition contains non-finite motion")
        if ee_delta_norm > float(max_ee_delta) + 1e-12:
            raise RuntimeError(
                "recorded EE transition exceeds limit: "
                f"norm={ee_delta_norm:.6f}m limit={float(max_ee_delta):.6f}m "
                f"delta={ee_delta.tolist()}"
            )
        if max_abs_joint_delta > float(max_joint_delta) + 1e-12:
            raise RuntimeError(
                "recorded joint transition exceeds limit: "
                f"max_abs={max_abs_joint_delta:.6f}rad "
                f"limit={float(max_joint_delta):.6f}rad"
            )

    def validate_grasp_plan_ik(self, object_x, object_y, object_type=None):
        """
        Validate every distinct Cartesian waypoint before resetting or stepping.

        Returns the plan when all waypoints are reachable. Otherwise raises a
        ValueError containing the exact failed waypoint for sampling diagnostics.
        """
        plan = self.make_grasp_plan(object_x, object_y, object_type=object_type)
        checked_positions = set()

        for waypoint_index, action in enumerate(plan):
            x_m, y_m, z_m = (float(action[0]), float(action[1]), float(action[2]))
            position = (x_m, y_m, z_m)
            if position in checked_positions:
                continue
            checked_positions.add(position)

            angles = self._calc_inv_kinematics(x_m * 100.0, y_m * 100.0, z_m * 100.0)
            if angles is None:
                raise ValueError(
                    "grasp IK precheck failed: "
                    f"waypoint={waypoint_index} xyz_m=({x_m:.4f}, {y_m:.4f}, {z_m:.4f})"
                )

        return plan


def run_episode_and_record(
    rc: SyncSimRaccoonDataset,
    logger: DatasetLogger,
    episode_id: int,
    instruction: str,
    object_specs: dict,
    target_color: str = "red",
    target_object_type: str = "cylinder",
    speed: int = 70,
    settle_seconds_per_action: float = 2.0,
    initial_settle_seconds: float = 0.5,
    hz: int = 10,
    touch_threshold: float = 0.1,
    diagnostic_logging: bool = True,
    min_lift_height: float = 0.015,
    max_cartesian_step: float = 0.004,
    max_ee_delta: float = 0.005,
    max_joint_delta: float = 0.35,
    target_clearance: float = 0.070,
    max_non_target_displacement: float = 0.003,
    physical_hold_validation_seconds: float = 0.15,
    pre_close_alignment_tolerance: float = 0.012,
    pre_close_alignment_timeout_seconds: float = 8.0,
):
    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

    target_spec = object_specs[target_color]
    target_body_name = target_spec["body_name"]
    target_object_type = str(target_spec.get("object_type", target_object_type))
    target_x = float(target_spec["x"])
    target_y = float(target_spec["y"])
    target_yaw = float(target_spec["yaw"])

    rc.validate_target_clearance(
        object_specs,
        target_color,
        min_clearance=target_clearance,
    )

    # Reject unreachable samples before mutating simulation state or creating an
    # episode directory. Collection normally prechecks this while sampling, but
    # keep the guard here for direct callers as well.
    plan = rc.validate_grasp_plan_ik(
        target_x,
        target_y,
        object_type=target_object_type,
    )

    rc.reset_episode(object_specs=object_specs, target_color=target_color)
    rc.lockh()

    # Let newly reset free-joint objects fall/settle before capturing frame_000000.
    # Without this, the first saved image can show objects slightly floating while
    # later frames look normal after one physics step.
    if initial_settle_seconds > 0:
        rc.settle_steps(seconds=initial_settle_seconds)
    initial_target_z = float(rc.get_object_pose(target_body_name)[2])
    initial_non_target_positions = rc.get_non_target_object_positions(
        object_specs,
        target_color,
    )

    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        task_type="grasp",
        goal_xy=[target_x, target_y],
        box_init_xy=[target_x, target_y],
        box_init_yaw=target_yaw,
        target_color=target_color,
        target_object_type=target_object_type,
        target_body_name=target_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
        collection_config={
            "hz": int(hz),
            "speed": int(speed),
            "settle_seconds_per_action": float(settle_seconds_per_action),
            "max_cartesian_step": float(max_cartesian_step),
            "max_ee_delta": float(max_ee_delta),
            "max_joint_delta": float(max_joint_delta),
            "target_clearance": float(target_clearance),
            "max_non_target_displacement": float(max_non_target_displacement),
            "physical_hold_validation_seconds": float(
                physical_hold_validation_seconds
            ),
            "pre_close_alignment_tolerance": float(
                pre_close_alignment_tolerance
            ),
            "pre_close_alignment_timeout_seconds": float(
                pre_close_alignment_timeout_seconds
            ),
            "grasp_stabilizer_enabled": bool(rc.enable_grasp_stabilizer),
            "grasp_mode": (
                "assisted_qpos_stabilizer"
                if rc.enable_grasp_stabilizer
                else "mujoco_physics_only"
            ),
        },
    )

    try:
        # The prompt decides which color/object pair to grasp. All objects are
        # visible, but the trajectory is aimed only at the prompted target.
        # Initial observation.
        obs = rc.get_observation()
        dt = 1.0 / hz
        step_counter = 0

        stage_names = ("above_open", "grasp_open", "grasp_close", "lift_closed")
        hold_frames = max(1, int(round(settle_seconds_per_action * hz)))
        previous_gripper_command = 0.0
        for waypoint_index, action in enumerate(plan):
            target_xyz = np.asarray(action[:3], dtype=np.float64)
            gripper_command = float(action[3])
            is_close_transition = (
                previous_gripper_command < 0.5 and gripper_command >= 0.5
            )
            motion_gripper_command = (
                previous_gripper_command
                if is_close_transition
                else gripper_command
            )
            interpolated_targets = rc.interpolate_cartesian_segment(
                start_xyz=obs["ee_pose"][:3],
                target_xyz=target_xyz,
                max_step=max_cartesian_step,
            )

            for commanded_xyz in interpolated_targets:
                # Never descend toward the object while closing. For the
                # open->closed transition, finish all Cartesian motion with
                # the gripper open and close only after the alignment gate.
                frame_action = [*commanded_xyz, motion_gripper_command]
                rc.execute_action(frame_action, speed=speed)
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=frame_action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )

                rc.settle_steps(seconds=dt)
                next_obs = rc.get_observation()
                rc.validate_recorded_transition(
                    previous_observation=obs,
                    next_observation=next_obs,
                    max_ee_delta=max_ee_delta,
                    max_joint_delta=max_joint_delta,
                )
                rc.validate_non_target_objects_undisturbed(
                    object_specs=object_specs,
                    target_color=target_color,
                    initial_positions=initial_non_target_positions,
                    max_displacement=max_non_target_displacement,
                )
                obs = next_obs
                step_counter += 1

            if is_close_transition:
                max_alignment_frames = max(
                    1,
                    int(round(pre_close_alignment_timeout_seconds * hz)),
                )
                for _ in range(max_alignment_frames):
                    if rc.is_pre_close_aligned(
                        obs["ee_pose"][:3],
                        target_xyz,
                        pre_close_alignment_tolerance,
                    ):
                        break

                    frame_action = [*target_xyz.tolist(), 0.0]
                    rc.execute_action(frame_action, speed=speed)
                    logger.log_step(
                        step_idx=step_counter,
                        image_rgb=obs["image"],
                        joint_angles=obs["joint_angles"],
                        gripper_state=obs["gripper_state"],
                        object_pose=obs["object_pose"],
                        ee_pose=obs["ee_pose"],
                        action=frame_action,
                        is_first=(step_counter == 0),
                        is_last=False,
                    )
                    rc.settle_steps(seconds=dt)
                    next_obs = rc.get_observation()
                    rc.validate_recorded_transition(
                        previous_observation=obs,
                        next_observation=next_obs,
                        max_ee_delta=max_ee_delta,
                        max_joint_delta=max_joint_delta,
                    )
                    rc.validate_non_target_objects_undisturbed(
                        object_specs=object_specs,
                        target_color=target_color,
                        initial_positions=initial_non_target_positions,
                        max_displacement=max_non_target_displacement,
                    )
                    obs = next_obs
                    step_counter += 1
                else:
                    alignment_error = np.asarray(
                        obs["ee_pose"][:3],
                        dtype=np.float64,
                    ) - target_xyz
                    raise RuntimeError(
                        "gripper remained open because pre-close alignment "
                        "did not converge: "
                        f"error={alignment_error.tolist()} "
                        f"norm={np.linalg.norm(alignment_error):.6f}m "
                        f"limit={pre_close_alignment_tolerance:.6f}m"
                    )

            for _ in range(hold_frames):
                frame_action = [*target_xyz.tolist(), gripper_command]
                rc.execute_action(frame_action, speed=speed)
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=frame_action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )
                rc.settle_steps(seconds=dt)
                next_obs = rc.get_observation()
                rc.validate_recorded_transition(
                    previous_observation=obs,
                    next_observation=next_obs,
                    max_ee_delta=max_ee_delta,
                    max_joint_delta=max_joint_delta,
                )
                rc.validate_non_target_objects_undisturbed(
                    object_specs=object_specs,
                    target_color=target_color,
                    initial_positions=initial_non_target_positions,
                    max_displacement=max_non_target_displacement,
                )
                obs = next_obs
                step_counter += 1

            previous_gripper_command = gripper_command

            if waypoint_index == 3 and rc.enable_grasp_stabilizer:
                # Assisted motion may only bring the object to the lift pose. It
                # must then remain physically held without qpos stabilization.
                rc.grasp_stabilizer_active = False
                rc.grasp_stabilizer_offset = None
                rc.grasp_stabilizer_min_z = None
                pre_validation_obs = obs
                rc.settle_steps(seconds=physical_hold_validation_seconds)
                obs = rc.get_observation()
                rc.validate_recorded_transition(
                    previous_observation=pre_validation_obs,
                    next_observation=obs,
                    max_ee_delta=max_ee_delta,
                    max_joint_delta=max_joint_delta,
                )
                rc.validate_non_target_objects_undisturbed(
                    object_specs=object_specs,
                    target_color=target_color,
                    initial_positions=initial_non_target_positions,
                    max_displacement=max_non_target_displacement,
                )

            diagnostic = rc.get_grasp_diagnostic(
                target_body_name=target_body_name,
                stage=stage_names[waypoint_index],
                waypoint_index=waypoint_index,
                touch_threshold=touch_threshold,
                initial_target_z=initial_target_z,
                min_lift_height=min_lift_height,
            )
            logger.log_grasp_diagnostic(diagnostic)
            if diagnostic_logging:
                commanded_xyz = np.asarray(target_xyz, dtype=np.float64)
                actual_ee_xyz = np.asarray(obs["ee_pose"][:3], dtype=np.float64)
                ee_position_error = actual_ee_xyz - commanded_xyz
                ee_position_error_norm = float(np.linalg.norm(ee_position_error))
                contact_pairs = [
                    f"{contact['body1']}/{contact['geom1']}<->{contact['body2']}/{contact['geom2']}"
                    for contact in diagnostic["contacts"]
                ]
                print(
                    f"[Grasp diagnostic] episode_id={episode_id:06d} | "
                    f"stage={diagnostic['stage']} | "
                    f"commanded_xyz={commanded_xyz.round(6).tolist()} | "
                    f"actual_ee_xyz={actual_ee_xyz.round(6).tolist()} | "
                    f"ee_error={ee_position_error.round(6).tolist()} | "
                    f"ee_error_norm={ee_position_error_norm:.6f}m | "
                    f"touch=({diagnostic['touch_left']:.4f}, {diagnostic['touch_right']:.4f}) | "
                    f"gripper_qpos={diagnostic['gripper_qpos']:.4f} | "
                    f"target_z={diagnostic['target_z']:.4f} | "
                    f"lift_delta={diagnostic['lift_delta']:.4f} | "
                    f"finger_contact=({diagnostic['left_finger_contact_pass']}, "
                    f"{diagnostic['right_finger_contact_pass']}) | "
                    f"stabilizer={diagnostic['grasp_stabilizer_active']} | "
                    f"contact_source={diagnostic['grasp_contact_source']} | "
                    f"target_robot_contact={diagnostic['target_robot_contact_pass']} | "
                    f"contacts={contact_pairs or ['none']}"
                )

        # Record terminal observation.
        logger.log_step(
            step_idx=step_counter,
            image_rgb=obs["image"],
            joint_angles=obs["joint_angles"],
            gripper_state=obs["gripper_state"],
            object_pose=obs["object_pose"],
            ee_pose=obs["ee_pose"],
            action=plan[-1],
            is_first=False,
            is_last=True,
        )

        grasp_diagnostic = logger.meta["grasp_diagnostics"][2]
        grasp_contact_confirmed = bool(
            grasp_diagnostic["grasp_contact_confirmed"]
            and grasp_diagnostic["grasp_contact_source"] == "target_bilateral_finger"
        )
        grasp_contact_retained_at_close = bool(
            grasp_diagnostic["left_finger_contact_pass"]
            or grasp_diagnostic["right_finger_contact_pass"]
        )
        final_diagnostic = logger.meta["grasp_diagnostics"][-1]
        physical_hold_bilateral_contact = bool(
            final_diagnostic["left_finger_contact_pass"]
            and final_diagnostic["right_finger_contact_pass"]
        )
        success = rc.is_target_grasp_success(
            target_body_name=target_body_name,
            initial_target_z=initial_target_z,
            grasp_contact_confirmed=grasp_contact_confirmed,
            grasp_contact_retained=(
                grasp_contact_retained_at_close
                and physical_hold_bilateral_contact
            ),
            min_lift_height=min_lift_height,
        )
        final_diagnostic["grasp_contact_confirmed"] = grasp_contact_confirmed
        final_diagnostic["grasp_contact_retained_at_close"] = (
            grasp_contact_retained_at_close
        )
        final_diagnostic["physical_hold_bilateral_contact"] = (
            physical_hold_bilateral_contact
        )
        final_diagnostic["grasp_contact_retained"] = bool(
            grasp_contact_retained_at_close
            and physical_hold_bilateral_contact
        )
        final_diagnostic["grasp_stabilizer_enabled"] = bool(
            rc.enable_grasp_stabilizer
        )
        final_diagnostic["grasp_mode"] = (
            "assisted_qpos_stabilizer"
            if rc.enable_grasp_stabilizer
            else "mujoco_physics_only"
        )
        final_diagnostic["success"] = bool(success)
        final_diagnostic["failure_checks"] = [
            check_name
            for check_name, passed in (
                ("grasp_bilateral_finger_contact", grasp_contact_confirmed),
                (
                    "grasp_contact_retained_at_close",
                    grasp_contact_retained_at_close,
                ),
                (
                    "physical_hold_bilateral_contact",
                    physical_hold_bilateral_contact,
                ),
                ("gripper_closed", final_diagnostic["gripper_closed_pass"]),
                ("pick_up_lift", final_diagnostic["lift_pass"]),
            )
            if not passed
        ]
        if diagnostic_logging:
            print(
                f"[Grasp result] episode_id={episode_id:06d} | success={success} | "
                f"failed_checks={final_diagnostic['failure_checks'] or ['none']}"
            )
        logger.finalize_episode(success=success)
        return success

    except Exception as e:
        logger.abort_episode()
        raise e


def _sample_task_goal(task_type, target_spec, object_specs):
    target_xy = np.asarray(
        [float(target_spec["x"]), float(target_spec["y"])],
        dtype=np.float64,
    )
    if task_type == "push":
        # The fixed horizontal gripper orientation produces a clean forward
        # push near the centerline. The collector places push targets directly
        # in this corridor instead of relying on repeated full-range rejection.
        if abs(float(target_xy[0])) > 0.012:
            raise ValueError("push target is outside the reliable center corridor")
        goal_xy = target_xy + np.asarray([0.0, 0.04], dtype=np.float64)
        if goal_xy[1] > 0.225:
            raise ValueError("push target is too close to the far workspace boundary")
    elif task_type == "pick_and_place":
        x_offset = -0.08 if target_xy[0] >= 0.0 else 0.08
        goal_xy = target_xy + np.asarray([x_offset, 0.0], dtype=np.float64)
        if goal_xy[0] < -0.12 or goal_xy[0] > 0.12:
            raise ValueError("pick-and-place goal is outside the workspace")
    else:
        raise ValueError(f"지원하지 않는 task_type입니다: {task_type}")

    for spec in object_specs.values():
        other_xy = np.asarray([float(spec["x"]), float(spec["y"])])
        if spec is target_spec:
            continue
        if float(np.linalg.norm(goal_xy - other_xy)) < 0.045:
            raise ValueError("task goal is too close to a distractor")
    return goal_xy


def run_multitask_episode_and_record(
    rc,
    logger,
    episode_id,
    instruction,
    task_type,
    object_specs,
    target_color,
    target_object_type,
    goal_xy,
    speed=10,
    settle_seconds_per_action=1.0,
    initial_settle_seconds=0.5,
    hz=100,
    max_cartesian_step=0.002,
    max_ee_delta=0.005,
    max_joint_delta=0.35,
    target_clearance=0.070,
    max_non_target_displacement=0.003,
    pre_close_alignment_tolerance=0.012,
    pre_close_alignment_timeout_seconds=8.0,
    diagnostic_logging=True,
):
    target_spec = object_specs[target_color]
    target_body_name = target_spec["body_name"]
    target_x = float(target_spec["x"])
    target_y = float(target_spec["y"])
    target_yaw = float(target_spec["yaw"])
    goal_xy = np.asarray(goal_xy, dtype=np.float64)

    rc.validate_target_clearance(object_specs, target_color, target_clearance)
    plan = rc.validate_task_plan_ik(
        task_type,
        target_x,
        target_y,
        goal_xy,
        object_type=target_object_type,
    )

    rc.reset_episode(object_specs=object_specs, target_color=target_color)
    rc.lockh()
    if initial_settle_seconds > 0:
        rc.settle_steps(initial_settle_seconds)

    initial_target_pose = rc.get_object_pose(target_body_name).copy()
    initial_non_target_positions = rc.get_non_target_object_positions(
        object_specs,
        target_color,
    )
    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        task_type=task_type,
        goal_xy=goal_xy,
        box_init_xy=[target_x, target_y],
        box_init_yaw=target_yaw,
        target_color=target_color,
        target_object_type=target_object_type,
        target_body_name=target_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
        collection_config={
            "hz": int(hz),
            "speed": int(speed),
            "settle_seconds_per_action": float(settle_seconds_per_action),
            "max_cartesian_step": float(max_cartesian_step),
            "max_ee_delta": float(max_ee_delta),
            "max_joint_delta": float(max_joint_delta),
            "target_clearance": float(target_clearance),
            "max_non_target_displacement": float(max_non_target_displacement),
            "pre_close_alignment_tolerance": float(pre_close_alignment_tolerance),
            "pre_close_alignment_timeout_seconds": float(
                pre_close_alignment_timeout_seconds
            ),
            "grasp_stabilizer_enabled": False,
            "grasp_mode": "mujoco_physics_only",
        },
    )
    logger.meta["task_diagnostics"] = []

    try:
        obs = rc.get_observation()
        dt = 1.0 / hz
        step_counter = 0
        hold_frames = max(1, int(round(settle_seconds_per_action * hz)))
        previous_gripper_command = 0.0
        grasp_contact_confirmed = False

        for waypoint_index, action in enumerate(plan):
            target_xyz = np.asarray(action[:3], dtype=np.float64)
            gripper_command = float(action[3])
            is_close_transition = (
                previous_gripper_command < 0.5 and gripper_command >= 0.5
            )
            motion_gripper_command = (
                previous_gripper_command if is_close_transition else gripper_command
            )

            commanded_points = rc.interpolate_cartesian_segment(
                obs["ee_pose"][:3],
                target_xyz,
                max_cartesian_step,
            )
            for commanded_xyz in commanded_points:
                frame_action = [*commanded_xyz, motion_gripper_command]
                rc.execute_action(frame_action, speed=speed)
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=frame_action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )
                rc.settle_steps(dt)
                next_obs = rc.get_observation()
                rc.validate_recorded_transition(
                    obs,
                    next_obs,
                    max_ee_delta=max_ee_delta,
                    max_joint_delta=max_joint_delta,
                )
                rc.validate_non_target_objects_undisturbed(
                    object_specs,
                    target_color,
                    initial_non_target_positions,
                    max_non_target_displacement,
                )
                obs = next_obs
                step_counter += 1

            if is_close_transition:
                max_alignment_frames = max(
                    1,
                    int(round(pre_close_alignment_timeout_seconds * hz)),
                )
                for _ in range(max_alignment_frames):
                    if rc.is_pre_close_aligned(
                        obs["ee_pose"][:3],
                        target_xyz,
                        pre_close_alignment_tolerance,
                    ):
                        break
                    frame_action = [*target_xyz.tolist(), 0.0]
                    rc.execute_action(frame_action, speed=speed)
                    logger.log_step(
                        step_idx=step_counter,
                        image_rgb=obs["image"],
                        joint_angles=obs["joint_angles"],
                        gripper_state=obs["gripper_state"],
                        object_pose=obs["object_pose"],
                        ee_pose=obs["ee_pose"],
                        action=frame_action,
                        is_first=False,
                        is_last=False,
                    )
                    rc.settle_steps(dt)
                    next_obs = rc.get_observation()
                    rc.validate_recorded_transition(
                        obs,
                        next_obs,
                        max_ee_delta=max_ee_delta,
                        max_joint_delta=max_joint_delta,
                    )
                    obs = next_obs
                    step_counter += 1
                else:
                    raise RuntimeError("pre-close alignment did not converge")
            else:
                max_alignment_frames = max(
                    1,
                    int(round(pre_close_alignment_timeout_seconds * hz)),
                )
                for _ in range(max_alignment_frames):
                    if rc.is_pre_close_aligned(
                        obs["ee_pose"][:3],
                        target_xyz,
                        pre_close_alignment_tolerance,
                    ):
                        break
                    frame_action = [*target_xyz.tolist(), gripper_command]
                    rc.execute_action(frame_action, speed=speed)
                    logger.log_step(
                        step_idx=step_counter,
                        image_rgb=obs["image"],
                        joint_angles=obs["joint_angles"],
                        gripper_state=obs["gripper_state"],
                        object_pose=obs["object_pose"],
                        ee_pose=obs["ee_pose"],
                        action=frame_action,
                        is_first=(step_counter == 0),
                        is_last=False,
                    )
                    rc.settle_steps(dt)
                    next_obs = rc.get_observation()
                    rc.validate_recorded_transition(
                        obs,
                        next_obs,
                        max_ee_delta=max_ee_delta,
                        max_joint_delta=max_joint_delta,
                    )
                    rc.validate_non_target_objects_undisturbed(
                        object_specs,
                        target_color,
                        initial_non_target_positions,
                        max_non_target_displacement,
                    )
                    grasp_contact_confirmed = bool(
                        grasp_contact_confirmed or rc.grasp_contact_confirmed
                    )
                    obs = next_obs
                    step_counter += 1
                else:
                    alignment_error = np.asarray(
                        obs["ee_pose"][:3],
                        dtype=np.float64,
                    ) - target_xyz
                    # During the contact segment of a push, the object itself
                    # resists the commanded endpoint. Reaching the exact pose
                    # is not required; the timeout already provides a bounded
                    # sustained push command.
                    if not (task_type == "push" and waypoint_index == 3):
                        raise RuntimeError(
                            f"{task_type} waypoint alignment did not converge: "
                            f"waypoint={waypoint_index} "
                            f"norm={np.linalg.norm(alignment_error):.6f}m"
                        )

            for _ in range(hold_frames):
                frame_action = [*target_xyz.tolist(), gripper_command]
                rc.execute_action(frame_action, speed=speed)
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=frame_action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )
                rc.settle_steps(dt)
                next_obs = rc.get_observation()
                rc.validate_recorded_transition(
                    obs,
                    next_obs,
                    max_ee_delta=max_ee_delta,
                    max_joint_delta=max_joint_delta,
                )
                rc.validate_non_target_objects_undisturbed(
                    object_specs,
                    target_color,
                    initial_non_target_positions,
                    max_non_target_displacement,
                )
                grasp_contact_confirmed = bool(
                    grasp_contact_confirmed or rc.grasp_contact_confirmed
                )
                obs = next_obs
                step_counter += 1

            target_pose = rc.get_object_pose(target_body_name)
            diagnostic = {
                "waypoint_index": int(waypoint_index),
                "commanded_xyz": target_xyz.tolist(),
                "actual_ee_xyz": [float(v) for v in obs["ee_pose"][:3]],
                "target_pose": [float(v) for v in target_pose],
                "gripper_command": gripper_command,
                "gripper_qpos": float(obs["gripper_state"]),
                "grasp_contact_confirmed": bool(grasp_contact_confirmed),
            }
            logger.meta["task_diagnostics"].append(diagnostic)
            if diagnostic_logging:
                print(
                    f"[Task diagnostic] episode_id={episode_id:06d} | "
                    f"task_type='{task_type}' | waypoint={waypoint_index} | "
                    f"target_xy={target_pose[:2].round(4).tolist()} | "
                    f"goal_xy={goal_xy.round(4).tolist()} | "
                    f"gripper_qpos={obs['gripper_state']:.4f} | "
                    f"contact_confirmed={grasp_contact_confirmed}"
                )
            previous_gripper_command = gripper_command

        logger.log_step(
            step_idx=step_counter,
            image_rgb=obs["image"],
            joint_angles=obs["joint_angles"],
            gripper_state=obs["gripper_state"],
            object_pose=obs["object_pose"],
            ee_pose=obs["ee_pose"],
            action=plan[-1],
            is_first=False,
            is_last=True,
        )

        final_pose = rc.get_object_pose(target_body_name)
        final_xy = np.asarray(final_pose[:2], dtype=np.float64)
        goal_error = float(np.linalg.norm(final_xy - goal_xy))
        displacement = float(
            np.linalg.norm(final_xy - np.asarray(initial_target_pose[:2]))
        )
        if task_type == "push":
            success = bool(
                displacement >= 0.025
                and goal_error <= 0.035
                and float(final_pose[2]) <= float(initial_target_pose[2]) + 0.012
            )
        else:
            gripper_reopened = float(obs["gripper_state"]) > rc.GRIP_OPEN - 0.03
            success = bool(
                grasp_contact_confirmed
                and goal_error <= 0.035
                and float(final_pose[2]) <= float(initial_target_pose[2]) + 0.015
                and gripper_reopened
            )

        logger.meta["task_result"] = {
            "goal_xy": goal_xy.tolist(),
            "final_target_xy": final_xy.tolist(),
            "goal_error": goal_error,
            "target_displacement": displacement,
            "grasp_contact_confirmed": bool(grasp_contact_confirmed),
            "success": success,
        }
        print(
            f"[Task result] episode_id={episode_id:06d} | task_type='{task_type}' | "
            f"success={success} | displacement={displacement:.4f}m | "
            f"goal_error={goal_error:.4f}m"
        )
        logger.finalize_episode(success=success)
        return success
    except Exception:
        logger.abort_episode()
        raise


def _balanced_target_counts(num_episodes, target_units):
    """
    Return per-target episode counts. If num_episodes is divisible by the
    number of target units, the split is exactly equal. Otherwise the remainder
    is distributed one-by-one to the first target units.
    """
    base = num_episodes // len(target_units)
    remainder = num_episodes % len(target_units)
    return {
        unit: base + (1 if idx < remainder else 0)
        for idx, unit in enumerate(target_units)
    }


def _sample_remaining_target(rng, target_counts, success_counts):
    remaining_targets = []
    remaining_weights = []

    for target_unit, target_count in target_counts.items():
        remaining = target_count - success_counts[target_unit]
        if remaining > 0:
            remaining_targets.append(target_unit)
            remaining_weights.append(remaining)

    if not remaining_targets:
        return None

    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_weights /= remaining_weights.sum()
    idx = int(rng.choice(len(remaining_targets), p=remaining_weights))
    return remaining_targets[idx]


def _load_existing_multitask_progress(dataset_root, target_units):
    """Recover successful unit counts and the next safe episode ID."""
    dataset_root = Path(dataset_root)
    target_units = set(target_units)
    success_counts = {target_unit: 0 for target_unit in target_units}
    max_episode_id = 0
    loaded_successes = 0
    skipped_entries = 0

    if not dataset_root.exists():
        return success_counts, 1, loaded_successes, skipped_entries

    for episode_dir in sorted(dataset_root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        try:
            episode_id = int(episode_dir.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            skipped_entries += 1
            continue
        max_episode_id = max(max_episode_id, episode_id)

        meta_path = episode_dir / "meta.json"
        if not meta_path.is_file():
            skipped_entries += 1
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped_entries += 1
            continue
        if not bool(meta.get("success")):
            continue

        target_unit = (
            str(meta.get("task_type", "")),
            str(meta.get("target_color", "")),
            str(meta.get("target_object_type", "")),
        )
        if target_unit not in target_units:
            skipped_entries += 1
            continue
        success_counts[target_unit] += 1
        loaded_successes += 1

    return success_counts, max_episode_id + 1, loaded_successes, skipped_entries


def _normalize_instruction_templates(instruction_templates=None, instruction_template=None):
    if instruction_templates is None:
        if instruction_template is None:
            instruction_templates = SyncSimRaccoonDataset.DEFAULT_INSTRUCTION_TEMPLATES
        else:
            instruction_templates = (instruction_template,)

    if isinstance(instruction_templates, str):
        instruction_templates = (instruction_templates,)

    instruction_templates = tuple(str(t) for t in instruction_templates if str(t).strip())
    if not instruction_templates:
        raise ValueError("instruction_templates는 비어 있을 수 없습니다.")
    return instruction_templates


def collect_dataset(
    xml_path="Raccoon_colored_cylinder.xml",
    dataset_root="raccoon_multitask_colored_objects",
    num_episodes=720,
    task_types=("push", "pick_and_place"),
    colors=("red", "blue", "green", "yellow"),
    object_types=("cylinder", "cube", "sphere"),
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    speed=10,
    settle_seconds_per_action=1.5,
    initial_settle_seconds=0.5,
    hz=125,
    touch_threshold=0.1,
    seed=None,
    max_attempts=None,
    object_x_range=(-0.12, 0.12),
    object_y_range=(0.14, 0.22),
    min_object_distance=0.045,
    target_clearance=0.070,
    max_ik_resamples=100,
    diagnostic_logging=True,
    min_lift_height=0.015,
    max_cartesian_step=0.002,
    max_ee_delta=0.005,
    max_joint_delta=0.35,
    max_non_target_displacement=0.003,
    physical_hold_validation_seconds=0.15,
    enable_grasp_stabilizer=False,
    pre_close_alignment_tolerance=0.012,
    pre_close_alignment_timeout_seconds=8.0,
    resume=True,
):
    """
    Collect balanced push and pick-and-place demonstrations.

    The default 720 episodes produce 30 successful examples for each of:
      2 tasks x 4 colors x 3 object types.
    """
    task_types = tuple(task_types)
    colors = tuple(colors)
    object_types = tuple(object_types)
    unknown_tasks = [
        task for task in task_types
        if task not in SyncSimRaccoonDataset.MULTITASK_INSTRUCTION_TEMPLATES
    ]
    if unknown_tasks:
        raise ValueError(f"지원하지 않는 task_types입니다: {unknown_tasks}")
    valid_colors = set(SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR.keys())
    unknown_colors = [color for color in colors if color not in valid_colors]
    if unknown_colors:
        raise ValueError(f"지원하지 않는 색상입니다: {unknown_colors}. 지원 색상: {sorted(valid_colors)}")
    unknown_object_types = [obj for obj in object_types if obj not in SyncSimRaccoonDataset.SUPPORTED_OBJECT_TYPES]
    if unknown_object_types:
        raise ValueError(
            f"지원하지 않는 object_types입니다: {unknown_object_types}. "
            f"지원 object_types: {list(SyncSimRaccoonDataset.SUPPORTED_OBJECT_TYPES)}"
        )

    if len(colors) == 0:
        raise ValueError("colors는 비어 있을 수 없습니다.")
    if len(object_types) == 0:
        raise ValueError("object_types는 비어 있을 수 없습니다.")
    if max_ik_resamples < 1:
        raise ValueError("max_ik_resamples는 1 이상이어야 합니다.")
    if (
        max_cartesian_step <= 0
        or max_ee_delta <= 0
        or max_joint_delta <= 0
        or target_clearance <= 0
        or max_non_target_displacement <= 0
        or physical_hold_validation_seconds <= 0
        or pre_close_alignment_tolerance <= 0
        or pre_close_alignment_timeout_seconds <= 0
    ):
        raise ValueError("motion limit 값은 모두 0보다 커야 합니다.")

    target_units = tuple(
        (task_type, color, object_type)
        for task_type in task_types
        for color in colors
        for object_type in object_types
    )
    target_counts = _balanced_target_counts(num_episodes, target_units)
    rng = np.random.default_rng(seed)

    if resume:
        (
            success_counts,
            next_episode_id,
            loaded_successes,
            skipped_existing_entries,
        ) = _load_existing_multitask_progress(dataset_root, target_units)
    else:
        success_counts = {target_unit: 0 for target_unit in target_units}
        next_episode_id = 1
        loaded_successes = 0
        skipped_existing_entries = 0

    if max_attempts is None:
        # Prevent infinite loops if grasp repeatedly fails.
        remaining_episode_count = max(0, num_episodes - sum(success_counts.values()))
        max_attempts = max(
            remaining_episode_count * 20,
            remaining_episode_count + 100,
        )

    rc = SyncSimRaccoonDataset(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
        enable_grasp_stabilizer=enable_grasp_stabilizer,
    )
    logger = DatasetLogger(root_dir=dataset_root, keep_failed=keep_failed)

    attempt_count = 0
    sample_rejection_count = 0

    print(f"Target task/color/object counts: {target_counts}")
    if resume:
        print(
            f"Resume: loaded_successes={loaded_successes}, "
            f"next_episode_id={next_episode_id:06d}, "
            f"skipped_existing_entries={skipped_existing_entries}, "
            f"dataset_root='{Path(dataset_root)}'"
        )
    print(
        "Instruction templates: "
        f"{SyncSimRaccoonDataset.MULTITASK_INSTRUCTION_TEMPLATES}"
    )
    print(
        f"Sampling config: x_range={object_x_range}, y_range={object_y_range}, "
        f"min_distance={min_object_distance}, max_ik_resamples={max_ik_resamples}, "
        f"hz={hz}, speed={speed}, max_cartesian_step={max_cartesian_step}, "
        f"max_ee_delta={max_ee_delta}, max_joint_delta={max_joint_delta}, "
        f"target_clearance={target_clearance}, "
        f"max_non_target_displacement={max_non_target_displacement}, "
        f"physical_hold_validation_seconds={physical_hold_validation_seconds}, "
        f"pre_close_alignment_tolerance={pre_close_alignment_tolerance}, "
        f"pre_close_alignment_timeout_seconds={pre_close_alignment_timeout_seconds}, "
        f"grasp_stabilizer={enable_grasp_stabilizer}, "
        f"generator_version={DATASET_GENERATOR_VERSION}"
    )

    try:
        while sum(success_counts.values()) < num_episodes and attempt_count < max_attempts:
            attempt_count += 1

            target_unit = _sample_remaining_target(rng, target_counts, success_counts)
            if target_unit is None:
                break
            task_type, target_color, target_object_type = target_unit

            template = str(
                rng.choice(
                    SyncSimRaccoonDataset.MULTITASK_INSTRUCTION_TEMPLATES[
                        task_type
                    ]
                )
            )
            instruction = template.format(color=target_color, object=target_object_type, object_type=target_object_type)
            object_specs = None
            goal_xy = None
            for ik_sample_index in range(1, max_ik_resamples + 1):
                forced_target = {
                    "color": target_color,
                    "object_type": target_object_type,
                }
                if task_type == "push":
                    forced_target.update(
                        {
                            "x_range": (-0.010, 0.010),
                            # Leave 4 cm of forward push distance while keeping
                            # the goal inside the reliable workspace.
                            "y_range": (
                                max(float(object_y_range[0]), 0.145),
                                min(float(object_y_range[1]), 0.180),
                            ),
                        }
                    )
                try:
                    object_specs = SyncSimRaccoonDataset.sample_object_specs(
                        rng=rng,
                        colors=colors,
                        object_types=object_types,
                        forced_target=forced_target,
                        x_range=object_x_range,
                        y_range=object_y_range,
                        min_distance=min_object_distance,
                        target_clearance=target_clearance,
                    )
                except RuntimeError as exc:
                    sample_rejection_count += 1
                    print(
                        f"[Sample reject {sample_rejection_count:04d}] color='{target_color}' | "
                        f"object='{target_object_type}' | sample={ik_sample_index}/{max_ik_resamples} | "
                        f"placement failed: {exc}"
                    )
                    continue
                target_spec = object_specs[target_color]
                try:
                    goal_xy = _sample_task_goal(
                        task_type,
                        target_spec,
                        object_specs,
                    )
                    rc.validate_target_clearance(
                        object_specs,
                        target_color,
                        min_clearance=target_clearance,
                    )
                    rc.validate_task_plan_ik(
                        task_type,
                        target_spec["x"],
                        target_spec["y"],
                        goal_xy,
                        object_type=target_object_type,
                    )
                    break
                except ValueError as exc:
                    sample_rejection_count += 1
                    print(
                        f"[Sample reject {sample_rejection_count:04d}] color='{target_color}' | "
                        f"object='{target_object_type}' | sample={ik_sample_index}/{max_ik_resamples} | {exc}"
                    )
            else:
                print(
                    "[Sample skip] 도달 가능한 multitask target을 이번 attempt에서 "
                    "샘플링하지 못해 다음 attempt로 진행합니다. "
                    f"task='{task_type}', color='{target_color}', "
                    f"object='{target_object_type}', "
                    f"x_range={object_x_range}, y_range={object_y_range}, "
                    f"max_ik_resamples={max_ik_resamples}"
                )
                continue

            # Never derive IDs from the number of successes: resumed datasets
            # may contain gaps or preserved failed episodes.
            episode_id = next_episode_id

            try:
                success = run_multitask_episode_and_record(
                    rc=rc,
                    logger=logger,
                    episode_id=episode_id,
                    instruction=instruction,
                    task_type=task_type,
                    object_specs=object_specs,
                    target_color=target_color,
                    target_object_type=target_object_type,
                    goal_xy=goal_xy,
                    speed=speed,
                    settle_seconds_per_action=settle_seconds_per_action,
                    initial_settle_seconds=initial_settle_seconds,
                    hz=hz,
                    diagnostic_logging=diagnostic_logging,
                    max_cartesian_step=max_cartesian_step,
                    max_ee_delta=max_ee_delta,
                    max_joint_delta=max_joint_delta,
                    target_clearance=target_clearance,
                    max_non_target_displacement=max_non_target_displacement,
                    pre_close_alignment_tolerance=pre_close_alignment_tolerance,
                    pre_close_alignment_timeout_seconds=pre_close_alignment_timeout_seconds,
                )

                if success:
                    success_counts[target_unit] += 1
                    next_episode_id += 1
                elif keep_failed:
                    next_episode_id += 1

                print(
                    f"[Attempt {attempt_count:04d}] episode_id={episode_id:06d} | "
                    f"task_type='{task_type}' | color='{target_color}' | object='{target_object_type}' | "
                    f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f}) | "
                    f"goal_xy=({goal_xy[0]:.3f}, {goal_xy[1]:.3f}) | "
                    f"instruction='{instruction}' | success={success} | "
                    f"success_counts={success_counts}"
                )
            except Exception as e:
                print(
                    f"[Attempt {attempt_count:04d}] task_type='{task_type}' | "
                    f"color='{target_color}' | "
                    f"object='{target_object_type}' | exception: {e}"
                )

    finally:
        rc.close()

    total_success = sum(success_counts.values())
    print(
        f"완료: success episodes = {total_success}/{num_episodes}, "
        f"attempts = {attempt_count}, sample_rejections = {sample_rejection_count}"
    )
    print(f"target pair별 성공 episode 수: {success_counts}")

    if total_success < num_episodes:
        print(
            "주의: max_attempts에 도달해서 목표 episode 수를 모두 채우지 못했습니다. "
            "max_attempts를 늘리거나 grasp 성공 조건/동작 파라미터를 확인하세요."
        )


if __name__ == "__main__":
    collect_dataset(
        # Resolve paths from this file so local execution does not depend on
        # the terminal/notebook working directory.
        xml_path=SCRIPT_DIR / "Raccoon_colored_cylinder.xml",
        dataset_root=SCRIPT_DIR / "raccoon_multitask_colored_objects",
        # 30 successful episodes for each task/color/object combination:
        # 2 tasks x 4 colors x 3 object types x 30 = 720.
        num_episodes=720,
        task_types=("push", "pick_and_place"),
        colors=("red", "blue", "green", "yellow"),
        object_types=("cylinder", "cube", "sphere"),
        keep_failed=False,
        use_viewer=False,
        camera_name="front_view",
        initial_settle_seconds=0.5,
        object_x_range=(-0.12, 0.12),
        object_y_range=(0.14, 0.22),
        min_object_distance=0.045,
        target_clearance=0.070,
        speed=10,
        hz=100,
        settle_seconds_per_action=1.5,
        max_cartesian_step=0.002,
        max_ee_delta=0.005,
        max_joint_delta=0.35,
        max_non_target_displacement=0.003,
        physical_hold_validation_seconds=0.15,
        # Physics-only grasp: do not move the object by directly editing qpos.
        enable_grasp_stabilizer=False,
        pre_close_alignment_tolerance=0.012,
        pre_close_alignment_timeout_seconds=8.0,
        # Continue from existing successful meta.json files without overwriting.
        resume=True,
    )
