import habitat

cfg = habitat.get_config(
    "configs/tasks/pointnav.yaml",
    ["habitat.task.measure.spl.success_distance=0.5"],
)

print(cfg.pretty())


env = habitat.Env(cfg.habitat)

print(env.observation_space)
print(env.action_space)
