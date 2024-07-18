from inspect import Attribute
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch

# 오리엔테이션 에러 계산함수
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[0:3] * torch.sign(q_r[3])

# 역기구학 제어 함수
def control_ik(dpose, damping, j_eef, num_envs):
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2) #: j_eef 텐서의 1번째와 2번째 차원을 서로 바꾸어 전치
    lmbda = torch.eye(6, device=device) * (damping ** 2) # 6*6 단위행렬 생성 및 댐핑계수 제곱하여 정규화행렬 구성
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 5) #댐핑된 최소 자승 문제를 풀고 야코비안의 전치행렬과 역행렬을 곱해 ee 자세변화량 dpose를 계산 
    return u

# dpose: (num_envs, 6) 형태의 텐서로, 원하는 엔드 이펙터 자세 변화량을 나타냅니다 (위치 및 방향 오차)
# damping: 정규화를 위한 댐핑 계수로, 스칼라 값
# j_eef: (num_envs, 6, 7) 형태의 텐서로, 각 환경에 대한 엔드 이펙터의 야코비안 행렬을 나타냅니다
# num_envs: IK 제어가 계산되는 환경 또는 인스턴스의 수



class ScrewFSM:

    def __init__(self, sim_dt, nut_height, bolt_height, screw_speed, screw_limit_angle, device, env_idx):
        self._sim_dt = sim_dt
        self._nut_height = nut_height
        self._bolt_height = bolt_height
        self._screw_speed = screw_speed
        self._screw_limit_angle = screw_limit_angle
        self.device = device
        self.env_idx = env_idx

        # states:
        self._state = "go_above_nut"

        # control / position constants:
        self._above_offset = torch.tensor([0, 0, 0.08 + self._bolt_height], dtype=torch.float32, device=self.device)
        self._grip_offset = torch.tensor([0, 0, 0.12 + self._nut_height], dtype=torch.float32, device=self.device)
        self._lift_offset = torch.tensor([0, 0, 0.15 + self._bolt_height], dtype=torch.float32, device=self.device)
        self._above_bolt_offset = torch.tensor([0, 0, self._bolt_height], dtype=torch.float32, device=self.device) + self._grip_offset
        self._on_bolt_offset = torch.tensor([0, 0, 0.8 * self._bolt_height], dtype=torch.float32, device=self.device) + self._grip_offset
        self._hand_down_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        grab_angle = torch.tensor([np.pi / 6.0], dtype=torch.float32, device=self.device)
        grab_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        grab_quat = quat_from_angle_axis(grab_angle, grab_axis).squeeze()
        self._nut_grab_q = quat_mul(grab_quat, self._hand_down_quat)
        self._screw_angle = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        self._screw_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        self._dpose = torch.zeros(6, dtype=torch.float32, device=self.device)
        self._gripper_separation = 0.0

    # 목표 위치와 로봇의 현재 자세 사이의 거리를 계산
    def get_dp_from_target(self, target_pos, target_quat, hand_pose) -> float:
        self._dpose[:3] = target_pos - hand_pose[:3]
        self._dpose[3:] = orientation_error(target_quat, hand_pose[3:])
        return torch.norm(self._dpose, p=2)

    # returns
    def update(self, nut_pose, bolt_pose, hand_pose, current_gripper_sep):
        newState = self._state
        if self._state == "go_above_nut":
            self._gripper_separation = 0.08
            target_pos = nut_pose[:3] + self._above_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "prep_grip"
        elif self._state == "prep_grip":
            pass
            self._gripper_separation = 0.08
            target_pos = nut_pose[:3] + self._grip_offset
            targetQ = quat_mul(nut_pose[3:], self._nut_grab_q)
            error = self.get_dp_from_target(target_pos, targetQ, hand_pose)
            if error < 2e-3:
                newState = "grip"
        elif self._state == "grip":
            self._gripper_separation = 0.0
            target_pos = nut_pose[:3] + self._grip_offset
            targetQ = quat_mul(nut_pose[3:], self._nut_grab_q)
            error = self.get_dp_from_target(target_pos, targetQ, hand_pose)
            gripped = (current_gripper_sep < 0.035)
            if error < 1e-2 and gripped:
                newState = "lift"
        elif self._state == "lift":
            self._gripper_separation = 0.0
            target_pos = nut_pose[:3]
            target_pos[2] = bolt_pose[2] + 0.004
            target_pos = target_pos + self._lift_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "go_above_nut"
        
        if newState != self._state:
            self._state = newState
            print(f"Env {self.env_idx} going to state {newState}")
        

    @property
    def d_pose(self):
        return self._dpose

    @property
    def gripper_separation(self):
        return self._gripper_separation

###############################################################################

# set random seed
np.random.seed(15)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments : 사용자 명령 인자들을 해석 및 처리하는 과정

# Add custom arguments
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) Nut-Bolt Screwing",
    custom_parameters=[{"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},],
)

# Force GPU:
if not args.use_gpu or args.use_gpu_pipeline:
    print("Forcing GPU sim - CPU sim not supported by SDF")
    args.use_gpu = True
    args.use_gpu_pipeline = True

# set torch device
device = args.sim_device

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 32
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.15

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "../../assets"

###########################################################################################

# create table asset
table_dims = gymapi.Vec3(0.3, 0.5, 0.2)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create bolt asset
bolt_file = "urdf/nut_bolt/bolt_m4_tight_SI_5x.urdf"
bolt_options = gymapi.AssetOptions()
bolt_options.flip_visual_attachments = False  # default = False
bolt_options.fix_base_link = True
bolt_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
bolt_options.density = 800.0  # 7850.0
bolt_options.armature = 0.0  # default = 0.0
bolt_options.linear_damping = 0.0  # default = 0.0
bolt_options.max_linear_velocity = 1000.0  # default = 1000.0
bolt_options.angular_damping = 0.0  # default = 0.5
bolt_options.max_angular_velocity = 1000.0  # default = 64.0
bolt_options.disable_gravity = False  # default = False
bolt_options.enable_gyroscopic_forces = True  # default = True
bolt_asset = gym.load_asset(sim, asset_root, bolt_file, bolt_options)

# create nut asset
nut_file = "urdf/nut_bolt/nut_m4_tight_SI_5x.urdf"
nut_options = gymapi.AssetOptions()
nut_options.flip_visual_attachments = False  # default = False
nut_options.fix_base_link = False
nut_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
nut_options.density = 800  # 7850.0  # default = 1000
nut_options.armature = 0.0  # default = 0.0
nut_options.linear_damping = 0.0  # default = 0.0
nut_options.max_linear_velocity = 1000.0  # default = 1000.0
nut_options.angular_damping = 0.0  # default = 0.5
nut_options.max_angular_velocity = 1000.0  # default = 64.0
nut_options.disable_gravity = False  # default = False
nut_options.enable_gyroscopic_forces = True  # default = True
nut_asset = gym.load_asset(sim, asset_root, nut_file, nut_options)

# load franka asset
franka_asset_file = "urdf/soomac_description/urdf/soomac.urdf"
asset_options = gymapi.AssetOptions()
asset_options.override_com = True
asset_options.override_inertia = True
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = False
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset) # 로봇의 관절 속성을 가져옴
#franka_lower_limits = franka_dof_props["lower"] # 각 관절의 하한 제한 값
#franka_upper_limits = franka_dof_props["upper"] # 각 관절의 상한 제한 값
franka_upper_limits = np.array([2.1, 2.1, 2.1, 2.1, 2.1, 0.0281, 0.0281])
franka_lower_limits = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, 0.0, 0.0])
#상하한 값이 무한대로 설정되어있어 임의로 제한해뒀음

franka_ranges = franka_upper_limits - franka_lower_limits # 상하한 제한 값들의 차이를 계산 (관절의 가동 범위를 나타내는 벡터)
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
print(franka_upper_limits, franka_lower_limits,franka_ranges,franka_mids)

# use position drive for all dofs
franka_dof_props["driveMode"][:5].fill(gymapi.DOF_MODE_POS) #위치 제어 모드 (처음 7개 관절 = 로봇팔)
franka_dof_props["stiffness"][:5].fill(400.0) # 강성 설정 (움직일때의 강도)
franka_dof_props["damping"][:5].fill(40.0) # 감쇠 설정 (목표위치로 이동할 때의 진동을 줄여 부드럽게 함)
# grippers
franka_dof_props["driveMode"][5:].fill(gymapi.DOF_MODE_POS) #위치 제어 모드 (7개 이후 관절 = 그리퍼)
franka_dof_props["stiffness"][5:].fill(800.0)
franka_dof_props["damping"][5:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset) # 관절 개수 (7+2)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:5] = franka_mids[:5] # 로봇팔(7번 관절까지)은 중간 위치를 초기위치로 설정
# grippers open
default_dof_pos[5:] = franka_upper_limits[5:] # 그리퍼(7번 이후)는 상한 제한 값을 초기위치로 설정하여 항상 열어두도록

# default_dof_pos = 관절의 목표 위치값
# default_dof_state = 관절의 현재 상태값

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# pos를 state에 할당하여 각 관절의 목표 위치를 설정함으로써 제어 시스템에서 로봇을 원하는 위치로 이동시킬 수 있다


# Numpy 배열을 PyTorch Tensor로 변환하는 과정
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
# 로봇팔 링크 인덱스를 가져와, EE로 사용할 강체의 인덱스를 추출
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["link5_1"]

# configure env grid = 환경 그리드 설정
num_envs = args.num_envs # 생성할 환경의 개수를 결정
num_per_row = int(math.sqrt(num_envs)) # 환경 개수의 제곱근을 계산하여 한 행에 배치될 환경의 개수 결정
spacing = 1.0 # 환경 간의 간격 설정
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0) # 그리드의 왼쪽 하단 모서리 설정
env_upper = gymapi.Vec3(spacing, spacing, spacing) # 그리드의 오른쪽 상단 모서리 설정
print("Creating %d environments" % num_envs)

# 객체들의 위치 설정
franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.3, 0.0, 0.5 * table_dims.z)
bolt_pose = gymapi.Transform()
nut_pose = gymapi.Transform()

###############################################################################

# fsm parameters:
fsm_device = 'cpu'

envs = []
nut_idxs = []
bolt_idxs = []
hand_idxs = []
fsms = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table -> create_actor(환경지정, 테이블 모델 및 에셋, 위치와 자세지정 gymapi.Transform(), 이름, 환경 번호, 액터의 초기화 옵션)
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add bolt
    bolt_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
    bolt_pose.p.y = table_pose.p.y + np.random.uniform(-0.1, 0.0)
    bolt_pose.p.z = table_dims.z
    #주어진 축과 각도에 따른 회전을 나타내는 쿼터니언을 생성하는 함수 (z축 중심으로 랜덤으로 회전)
    bolt_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi)) 
    bolt_handle = gym.create_actor(env, bolt_asset, bolt_pose, "bolt", i, 0)
    bolt_props = gym.get_actor_rigid_shape_properties(env, bolt_handle)
    #bolt_props[0].filter = imesh
    bolt_props[0].friction = 0.0  # default = ?
    bolt_props[0].rolling_friction = 0.0  # default = 0.0
    bolt_props[0].torsion_friction = 0.0  # default = 0.0
    bolt_props[0].restitution = 0.0  # default = ?
    bolt_props[0].compliance = 0.0  # default = 0.0
    bolt_props[0].thickness = 0.0  # default = 0.0
    gym.set_actor_rigid_shape_properties(env, bolt_handle, bolt_props)

    # get global index of box in rigid body state tensor
    bolt_idx = gym.get_actor_rigid_body_index(env, bolt_handle, 0, gymapi.DOMAIN_SIM)
    # gymapi.DOMAIN_SIM = 글로벌 도메인. 즉, 시뮬레이션 전체에서 유일한 인덱스를 반환하도록 지정하는 것
    bolt_idxs.append(bolt_idx)

    # add nut
    nut_pose.p.x = bolt_pose.p.x + np.random.uniform(-0.04, 0.04)
    nut_pose.p.y = bolt_pose.p.y + 0.2 + np.random.uniform(-0.04, 0.04)
    nut_pose.p.z = table_dims.z + 0.02
    nut_handle = gym.create_actor(env, nut_asset, nut_pose, "nut", i, 0)
    nut_props = gym.get_actor_rigid_shape_properties(env, nut_handle)
    #nut_props[0].filter = i
    nut_props[0].friction = 0.2  # default = ?
    nut_props[0].rolling_friction = 0.0  # default = 0.0
    nut_props[0].torsion_friction = 0.0  # default = 0.0
    nut_props[0].restitution = 0.0  # default = ?
    nut_props[0].compliance = 0.0  # default = 0.0
    nut_props[0].thickness = 0.0  # default = 0.0
    gym.set_actor_rigid_shape_properties(env, nut_handle, nut_props)

    # get global index of box in rigid body state tensor
    nut_idx = gym.get_actor_rigid_body_index(env, nut_handle, 0, gymapi.DOMAIN_SIM)
    nut_idxs.append(nut_idx)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 0)

    # set dof properties = 로봇의 자유도 속성 설정 (각 관절의 구동모드, 강성, 감쇠 등)
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states = 로봇의 초기 자유도 상태 설정
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets = 로봇의 초기 위치 목표 설정
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "link5_1", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # create env's fsm - run them on CPU
    fsms.append(ScrewFSM(sim_params.dt, 0.016, 0.1, 30.0 / 180.0 * np.pi, 60.0/180.0 * np.pi, fsm_device, i))

##########################################################################3

# point camera at middle env
cam_pos = gymapi.Vec3(1, 0, 0.6)
cam_target = gymapi.Vec3(-1, 0, 0.5)
middle_env = envs[0]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

###################################################################################

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka") # 로봇의 자코비안 텐서 획득 (원시 야코비안)
    # num envs : 환경의 수
    # 10 : 로봇의 링크 또는 세그먼트 수
    # 6 : 자코비안 행렬의 자유도, 일반적으로 이동 및 회전 속도를 포함
    # 9 : 로봇의 관절 또는 자유도
jacobian = gymtorch.wrap_tensor(_jacobian) # 원시 야코비안을 PyTorch 텐서로 변환


# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :5]
# 자코비안 텐서에서 EE에 해당하는 항목 추출. 
# [모든 환경, 로봇 손에 해당하는 인덱스, 모든 자유도, 첫 7개 열]


# get rigid body state tensor : 강체 상태에 대한 텐서 가져와서 PyTorch 텐서로 변환
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor : 시뮬 환경에서 관절 상태에 대한 정보 획득하고 파이토치 텐서로 변환하여 관절의 위치 데이터 추출
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 7, 1)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)

# dp and gripper sep tensors:
d_pose = torch.zeros((num_envs, 6), dtype=torch.float32, device=fsm_device)
grip_sep = torch.zeros((num_envs, 1), dtype=torch.float32, device=fsm_device)

##################################################################################3

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    rb_states_fsm = rb_states.to(fsm_device)
    nut_poses = rb_states_fsm[nut_idxs, :7]
    bolt_poses = rb_states_fsm[bolt_idxs, :7]
    hand_poses = rb_states_fsm[hand_idxs, :7]
    dof_pos_fsm = dof_pos.to(fsm_device)
    cur_grip_sep_fsm = dof_pos_fsm[:, 5] + dof_pos_fsm[:, 6]
    for env_idx in range(num_envs):
        fsms[env_idx].update(nut_poses[env_idx, :], bolt_poses[env_idx, :], hand_poses[env_idx, :], cur_grip_sep_fsm[env_idx])
        d_pose[env_idx, :] = fsms[env_idx].d_pose
        grip_sep[env_idx] = fsms[env_idx].gripper_separation

    pos_action[:, :5] = dof_pos.squeeze(-1)[:, :5] + control_ik(d_pose.unsqueeze(-1).to(device), damping, j_eef, num_envs)
    # gripper actions depend on distance between hand and box

    grip_acts = torch.cat((0.5 * grip_sep, 0.5 * grip_sep), 1).to(device)
    pos_action[:, 5:7] = grip_acts

    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)