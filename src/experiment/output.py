import json
import logging
import os
import shutil
from dataclasses import dataclass


RUN_SUBDIRECTORIES = ("circuit", "topology", "logs", "metrics", "graphs", "quantum")


class RunDirectoryConflict(RuntimeError):
    pass


class RunDirectory:

    def __init__(self, root):

        self.root = os.path.abspath(root)
        self.written = []

    def path(self, *parts):

        target = os.path.abspath(os.path.join(self.root, *parts))

        if os.path.commonpath([target, self.root]) != self.root:
            raise ValueError(f"{target} is outside the simulation directory {self.root}")

        os.makedirs(os.path.dirname(target), exist_ok=True)
        self.written.append(target)

        return target

    def prepare(self):

        for name in RUN_SUBDIRECTORIES:
            os.makedirs(os.path.join(self.root, name), exist_ok=True)

    def write_json(self, data, *parts):

        with open(self.path(*parts), "w") as handle:
            json.dump(data, handle, indent=2, default=_json_default)

    def write_text(self, text, *parts):

        with open(self.path(*parts), "w") as handle:
            handle.write(text)

    def child(self, name):
        return RunDirectory(os.path.join(self.root, name))


def _json_default(value):

    if isinstance(value, tuple):
        return list(value)

    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, set):
        return sorted(value)

    return str(value)


def prepare_seed_directory(results_directory, master_seed, controls, overwrite):

    root = os.path.join(results_directory, f"simulation_{master_seed}")
    manifest_path = os.path.join(root, "manifest.json")

    if os.path.exists(root):

        existing_controls = None

        if os.path.exists(manifest_path):
            with open(manifest_path) as handle:
                existing_controls = json.load(handle).get("experiment", {}).get("controls")

        comparable = json.loads(json.dumps(controls, default=_json_default))

        if existing_controls is not None and existing_controls != comparable and not overwrite:
            raise RunDirectoryConflict(
                f"{root} already holds a different experiment for seed {master_seed}. "
                "Use --overwrite to replace it, or choose another seed."
            )

        shutil.rmtree(root)

    directory = RunDirectory(root)
    directory.prepare()

    return directory


@dataclass
class RunLoggers:
    simulation: logging.Logger
    rerouting: logging.Logger
    repartition: logging.Logger
    handlers: list

    def close(self):

        for logger, handler in self.handlers:
            logger.removeHandler(handler)
            handler.close()


def open_run_loggers(directory, namespace):

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    simulation = logging.getLogger(namespace)
    rerouting = logging.getLogger(f"{namespace}.rerouting")
    repartition = logging.getLogger(f"{namespace}.repartition")

    handlers = []

    for logger, filename in (
        (simulation, "simulation.log"),
        (rerouting, "rerouting.log"),
        (repartition, "repartition.log"),
    ):
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(directory.path("logs", filename), mode="w")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handlers.append((logger, handler))

    simulation.propagate = False
    rerouting.propagate = True
    repartition.propagate = True

    return RunLoggers(simulation, rerouting, repartition, handlers)

