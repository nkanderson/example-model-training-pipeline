"""
Multi-GPU launcher for Optuna optimization.

This script launches 4 worker processes pinned to GPU devices 0-3,
all sharing the same Optuna study/storage so trials are coordinated.

It accepts most of the same CLI options as optimize.py, except
--no-hw-acceleration (GPU usage is implicit here).

Examples:
    python optimize_multi_gpu.py --neuron-type leaky --n-trials 120
    python optimize_multi_gpu.py --neuron-type fractional --n-trials 200 --study-name flif-mgpu-v1
    python optimize_multi_gpu.py --neuron-type bitshift --n-trials 80 --get-importance
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


GPU_WORKERS = [0, 1, 2, 3]


def split_trials(total_trials: int, workers: int) -> list[int]:
    """Split total trials as evenly as possible across workers."""
    base = total_trials // workers
    remainder = total_trials % workers
    parts = [base] * workers
    for index in range(remainder):
        parts[index] += 1
    return parts


def build_worker_command(
    args, study_name: str, storage: str, worker_trials: int
) -> list[str]:
    """Build one optimize.py command for a worker."""
    command = [
        sys.executable,
        "optimize.py",
        "--neuron-type",
        args.neuron_type,
        "--study-name",
        study_name,
        "--storage",
        storage,
        "--n-trials",
        str(worker_trials),
    ]

    if args.search_space:
        command.extend(["--search-space", args.search_space])
    if args.num_episodes is not None:
        command.extend(["--num-episodes", str(args.num_episodes)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.pruner:
        command.extend(["--pruner", args.pruner])

    return command


def run_final_importance(args, study_name: str, storage: str) -> int:
    """Run a final single-process importance/export pass after workers complete."""
    command = [
        sys.executable,
        "optimize.py",
        "--neuron-type",
        args.neuron_type,
        "--study-name",
        study_name,
        "--storage",
        storage,
        "--n-trials",
        "0",
        "--get-importance",
    ]

    if args.search_space:
        command.extend(["--search-space", args.search_space])
    if args.num_episodes is not None:
        command.extend(["--num-episodes", str(args.num_episodes)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.pruner:
        command.extend(["--pruner", args.pruner])

    print("\nRunning final post-optimization importance pass...")
    print(" ".join(command))
    return subprocess.call(command)


def main():
    parser = argparse.ArgumentParser(
        description="Launch optimize.py across GPUs 0-3 using one worker per GPU"
    )

    parser.add_argument(
        "--neuron-type",
        type=str,
        required=True,
        choices=["leaky", "fractional", "bitshift"],
        help="Neuron type to optimize",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        help="Path to search space YAML config",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Total number of trials across all 4 GPU workers",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name shared by all workers",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI shared by all workers",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override num_episodes per trial",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. Worker index is added for per-worker seeds.",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "hyperband", "none"],
        help="Optuna pruner strategy",
    )
    parser.add_argument(
        "--get-importance",
        action="store_true",
        help="Run one final --get-importance pass after all workers finish",
    )

    args = parser.parse_args()

    if args.n_trials < 0:
        raise ValueError("--n-trials must be >= 0")

    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.study_name = f"{args.neuron_type}-mgpu-{timestamp}"

    if args.storage is None:
        studies_dir = Path("optuna_studies")
        studies_dir.mkdir(exist_ok=True)
        args.storage = f"sqlite:///optuna_studies/{args.neuron_type}.db"

    log_dir = Path("optuna_studies") / "logs" / args.study_name
    log_dir.mkdir(parents=True, exist_ok=True)

    trial_splits = split_trials(args.n_trials, len(GPU_WORKERS))

    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Total trials: {args.n_trials}")
    print(f"GPUs: {GPU_WORKERS}")
    print(f"Trial split: {trial_splits}")

    processes = []

    for worker_index, gpu_id in enumerate(GPU_WORKERS):
        worker_trials = trial_splits[worker_index]

        # Skip idle workers when total trials < number of workers
        if worker_trials == 0:
            continue

        # Worker-local seed offset for reproducibility without collision
        worker_seed = None
        if args.seed is not None:
            worker_seed = args.seed + worker_index

        worker_args = argparse.Namespace(**vars(args))
        worker_args.seed = worker_seed

        command = build_worker_command(
            worker_args,
            study_name=args.study_name,
            storage=args.storage,
            worker_trials=worker_trials,
        )

        log_file = log_dir / f"worker_gpu{gpu_id}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(
            f"\nLaunching worker {worker_index} on GPU {gpu_id} ({worker_trials} trials)"
        )
        print(" ".join(command))
        print(f"Log: {log_file}")

        handle = open(log_file, "w")
        proc = subprocess.Popen(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append((proc, handle, gpu_id, worker_trials))

    exit_codes = []
    for proc, handle, gpu_id, worker_trials in processes:
        code = proc.wait()
        handle.close()
        exit_codes.append(code)
        status = "OK" if code == 0 else "FAILED"
        print(f"Worker GPU {gpu_id} ({worker_trials} trials): {status} (exit={code})")

    if any(code != 0 for code in exit_codes):
        print("\nOne or more workers failed. See logs for details:")
        print(log_dir)
        sys.exit(1)

    if args.get_importance:
        rc = run_final_importance(
            args, study_name=args.study_name, storage=args.storage
        )
        if rc != 0:
            print("Final importance pass failed.")
            sys.exit(rc)

    print("\nAll workers completed successfully.")
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
