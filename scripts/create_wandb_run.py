import time
import wandb
import sys
import subprocess
import os


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main(entity: str, project: str, starting_value: float, tags: list[str]):
    run = wandb.init(entity=entity, project=project)

    for i in range(10):
        run.log({"loss": starting_value - 0.17 * i})

    run.tags = run.tags + tuple(tags) + (get_git_revision_hash(),)
    time.sleep(1)
    run.finish()


if __name__ == "__main__":
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    main(entity, project, float(sys.argv[1]), sys.argv[2:])
