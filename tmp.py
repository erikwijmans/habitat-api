import hydra

import habitat

hydra.experimental.initialize()
cfg = habitat.get_config(
    "configs/tasks/pointnav.yaml",
    ["habitat.task.measure.spl.success_distance=0.5"],
)


env = habitat.Env(cfg.habitat)
