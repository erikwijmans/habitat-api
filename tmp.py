import habitat
import habitat_baselines

habitat_baselines.config.get_config()

cfg = habitat.get_config(
    "configs/tasks/pointnav.yaml",
    [
        "habitat.task.measure.spl.success_distance=0.5",
        "habitat/task/sensor=compass_sensor",
        "habitat/task/sensor=gps_sensor",
        "habitat.task.sensor.gps_sensor.dimensionality=2",
    ],
)

print(cfg.pretty())


env = habitat.Env(cfg.habitat)

print(env.observation_space)
print(env.action_space)
