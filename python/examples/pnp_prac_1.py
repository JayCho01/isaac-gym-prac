from isaacgym import gymapi
from isaacgym import gymutil
import random
import math

gym = gymapi.acquire_gym() # isaac gym 초기화 및 gym 객체 생성

########################################################################
# 환경 설정 #

sim_params = gymapi.SimParams()

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.use_gpu_pipeline = False

sim_params.physx.num_position_iterations = 5
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.rest_offset = 0.001
sim_params.physx.contact_offset = 0.02
sim_params.physx.use_gpu = True
# gymapi.SimParams : 시뮬레이션 환경의 다양한 설정을 정의 클래스
# 시뮬레이션의 시간 스텝, 서브 스텝, 물리 엔진 특성 등을 관리
# 즉 시뮬레이션의 물리적 동작 방식 조정
########################################################################
# 시뮬 설정 #

# gym 객체와 sim_params 설정 바탕으로 시뮬레이션 환경 생성
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params) 

# 바닥 객체 설정
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# sim 환경에 plane_params 바닥 추가
gym.add_ground(sim, plane_params)
# urdf 주소 및 파일 정의
asset_root = "../../assets"
asset_file = "urdf/soomac_description/urdf/soomac.urdf"

# asset_root = "../../assets"
# asset_file = "urdf/ur10_sy/urdf/ur10_gripper2.urdf"
# urdf 객체 추가 시 옵션 정의
asset_options = gymapi.AssetOptions() # 옵션 담는 객체 생성
asset_options.fix_base_link = True # 베이스 링크 fix 여부 설정
asset_options.flip_visual_attachments = False # 뒤집기(?) 설정
asset_options.override_com = True
asset_options.override_inertia = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
# sim 환경에 asset_root, asset_file의 urdf 객체를 asset_option으로 추가

# 객체 개수, row, 객체 간 거리 설정
num_envs = 1
envs_per_row = 1
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
########################################################################
# 객체 정보 출력(only 출력용, 기능에 영향 x) #
num_ur5e_bodies = gym.get_asset_rigid_body_count(asset)
num_ur5e_dofs = gym.get_asset_dof_count(asset)
body_info = gym.get_asset_rigid_body_dict(asset)

print('rigid body : ', num_ur5e_bodies)
print('DOFs : ', num_ur5e_dofs)
print('body dict : ', body_info)
########################################################################
# 실제 시뮬레이션 진행 #
# envs = []
# actor_handles = []
# for i in range(num_envs):
#     env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
#     envs.append(env)

#     height = 2.0

#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0.0, height, 0.0)

#     actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
#     actor_handles.append(actor_handle)


env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
height = 2.0
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, height, 0.0)

actor_handle = gym.create_actor(env, asset, pose, "MyActor", 1, 1)
cam_props = gymapi.CameraProperties()

# 카메라 위치 설정
cam_pose = gymapi.Transform()
cam_pose.p = gymapi.Vec3(0.0, height - 0.1, 0.5)  # 로봇에 더 가까운 위치
cam_pose.r = gymapi.Quat.from_euler_zyx(0, math.radians(45.0), 0)  # 카메라 각도 조정





viewer = gym.create_viewer(sim, cam_props)

    
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

########################################################################
# 종료 #
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)



