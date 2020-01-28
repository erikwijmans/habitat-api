Habitat-API Configuration System
================================


The Habitat-API configuration system leverages `Hydra <https://hydra.cc/>`_ to provide compositional configuration.
Hydra allows the configuration to be composed at runtime based on provided "defaults".  The
core functionality hydra provides that enables us to do this is config values that specify *files* to load.
In a config file, these will be in the ``defaults`` list and specified by key value pairs where the key is a folder
and the value is the name of a file (minus the ``.yaml`` suffix) in that folder.


Habitat + Hydra
---------------


Hydra configs will commonly have a list under the key ``defaults`` at the top of the file, i.e.

::

    defaults:
        - habitat/task/sensor: pointgoal_sensor
        - habitat/task/sensor: proximity_sensor


These config values point to *files* to compose into the config, not values to put into the config!
Habitat uses these to pull in the default configuration for sensors/measure/actions/etc.
This enables the config
to only include the configuration for sensors, measure, actions, etc. that you are actually using -
what you see in the config is what you get!


The rest of the configuration can be used to override values in the default specifications and set values for your experiments!


::

    defaults:
        - habitat/task/sensor: pointgoal_sensor
        - habitat/task/sensor: proximity_sensor

    my_exp:
        my_value: 5

    habitat:
        task:
            sensor:
                pointgoal_sensor:
                    max_detection_radius: 3.0





Frequently Asked Questions
--------------------------


* How do I remove a sensor/measure/action/etc from the config at runtime?

    Since the config is what you see is what you get, this is actually quite simple!  Just delete
    the node for that (e.g.) sensor.

    ::

        with omegaconf.read_write(cfg):
            del cfg.habitat.task.sensor["pointgoal_sensor"]


* How do I add a sensor/measure/action/etc to the config at runtime?


    Just compose it in.  Note that you will want to query hydra directly to avoid pulling in the habitat base config again.


    ::

        with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
            cfg = omegaconf.OmegaConfg.merge(
                cfg, hydra.experimental.compse(overrides=["habitat/task/sensor=pointgoal_sensor"])
            )



    You can also easily override the default values for that sensor with this method

    ::

        with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
            cfg = omegaconf.OmegaConfg.merge(
                cfg,
                hydra.experimental.compse(
                    overrides=[
                        "habitat/task/sensor=pointgoal_sensor",
                        "habitat.task.sensor.pointgoal_sensor.dimensionality=3",
                    ],
                ),
            )
