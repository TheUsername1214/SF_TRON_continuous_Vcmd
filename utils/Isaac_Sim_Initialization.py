def setup_world_physics(World,device,backend,dt,sub_step):
    """配置物理世界参数"""
    batch = 2
    world = World(device=device, backend=backend)
    world.set_simulation_dt(physics_dt=dt / sub_step, rendering_dt=dt / sub_step)
    world.get_physics_context().set_gpu_max_rigid_patch_count(int(60010/batch))
    world.get_physics_context().set_gpu_max_rigid_contact_count(int(111111/batch))
    world.get_physics_context().set_gpu_found_lost_pairs_capacity(int(26161311/batch))
    world.get_physics_context().set_gpu_found_lost_aggregate_pairs_capacity(int(113982000/batch))
    world.get_physics_context().set_gpu_total_aggregate_pairs_capacity(int(8002000/batch))
    world.get_physics_context().enable_gpu_dynamics(True)
    return world

