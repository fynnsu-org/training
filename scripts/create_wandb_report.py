import wandb
import os
import wandb_workspaces.reports.v2 as wr
import subprocess


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_last_tagged_run(
    entity: str, project: str, tag: str, finished_only: bool = True
):
    filters = {"tags": {"$eq": tag}}
    if finished_only:
        filters["state"] = {"$eq": "finished"}

    try:
        return next(
            wandb.Api().runs(
                f"{entity}/{project}",
                filters=filters,
                order="-created_at",
                per_page=1,
            )
        )
    except StopIteration:
        return None


def compare_runs(entity: str, project: str, run_ids: list[str], run_names: list[str]):

    report = wr.Report(
        entity=entity,
        project=project,
        title=f"Compare {'-'.join(run_ids)}",
        description=f"This report compares runs: \n{'\n'.join(f'{name} ({id})' for name, id in zip(run_names, run_ids))}",
    )
    pg = wr.PanelGrid(
        runsets=[
            wr.Runset(entity, project, "Run Comparison", filters=f"name in {run_ids}")
        ],
        panels=[
            wr.RunComparer(diff_only=True, layout={"x": 0, "y": 21, "w": 24, "h": 16}),
            wr.LinePlot(
                "GPU Power Usage (%)",
                "step",
                [
                    "system/gpu.0.powerPercent",
                    "system/gpu.1.powerPercent",
                    "system/gpu.2.powerPercent",
                    "system/gpu.3.powerPercent",
                ],
                range_y=(0, 100),
                layout={"x": 8, "y": 15, "w": 8, "h": 6},
            ),
            wr.LinePlot(
                "GPU Memory Allocated (%)",
                "step",
                [
                    "system/gpu.0.memoryAllocated",
                    "system/gpu.1.memoryAllocated",
                    "system/gpu.2.memoryAllocated",
                    "system/gpu.3.memoryAllocated",
                ],
                range_y=(0, 100),
                layout={"x": 0, "y": 15, "w": 8, "h": 6},
            ),
            wr.LinePlot(
                "Overall Throughput",
                "step",
                ["overall_throughput"],
                layout={"x": 0, "y": 9, "w": 8, "h": 6},
            ),
            wr.LinePlot(
                "Number of Loss Counted Tokens",
                "step",
                ["num_loss_counted_tokens"],
                layout={"x": 16, "y": 9, "w": 8, "h": 6},
            ),
            wr.LinePlot(
                "Samples Seen",
                "step",
                ["samples_seen"],
                layout={"x": 8, "y": 9, "w": 8, "h": 6},
            ),
            wr.LinePlot(
                "Learning Rate",
                "step",
                ["lr"],
                layout={"x": 12, "y": 0, "w": 12, "h": 9},
            ),
            wr.LinePlot(
                "System Disk Usage (GB)",
                "step",
                ["system/disk./.usageGB"],
                layout={"x": 16, "y": 15, "w": 8, "h": 6},
            ),
            wr.LinePlot(
                "Total Loss",
                "step",
                ["total_loss"],
                layout={"x": 0, "y": 0, "w": 12, "h": 9},
            ),
        ],
    )
    report.blocks = report.blocks[:1] + [pg] + report.blocks[1:]
    report.save()

    if os.getenv("CI"):
        # is set to `true` in GitHub Actions https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"REPORT_URL={report.url}", file=f)
    return report.url


if __name__ == "__main__":
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    baseline_tag = os.getenv("BASELINE_TAG") or "main"

    git_hash = get_git_revision_hash()
    new_run = get_last_tagged_run(entity, project, tag=git_hash, finished_only=False)
    baseline_run = get_last_tagged_run(entity, project, tag=baseline_tag)

    report_url = compare_runs(
        entity, project, [new_run.id, baseline_run.id], ["PR", "Base"]
    )
    print("Report URL:", report_url)
