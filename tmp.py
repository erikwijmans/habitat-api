import hydra

import habitat

hydra.experimental.initialize()
cfg = habitat.get_config("configs/tasks/pointnav.yaml")
print(cfg.pretty())


env = habitat.Env(cfg.habitat)
