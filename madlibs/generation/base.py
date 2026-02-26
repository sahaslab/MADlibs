import os
import abc
import multiprocessing
import dataclasses
import uuid
import tempfile
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import Any


@dataclasses.dataclass(frozen=True)
class PromptTask:
    idx: int
    prompt: str


_WORKER_GENERATOR: "MusicGenerator | None" = None


class MusicGenerator(abc.ABC):
    def __init__(
        self,
        num_workers: int = 1,
        device: str = "cuda",
        log_gpu_usage: bool = True
    ):
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.log_gpu_usage = log_gpu_usage

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but 'cuda' was specified as device.")

    @abc.abstractmethod
    def _get_init_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_audio(self, prompt: str, save_path: str) -> str:
        raise NotImplementedError

    def _estimate_worker_vram_bytes(self, probe_prompt: str | None = None) -> int | None:
        del probe_prompt
        return None

    def generate_one(self, prompt: str, save_path: str) -> str:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "prompt.txt"), "w", encoding="utf-8") as file_obj:
            file_obj.write(prompt)
        return self._generate_audio(prompt=prompt, save_path=save_path)

    def generate(self, prompts: list[str], output_dir: str) -> str:
        run_path = self._create_run_path(output_dir=output_dir)
        tasks = [PromptTask(idx=idx, prompt=prompt) for idx, prompt in enumerate(prompts)]
        parallel_workers = self._resolve_parallel_workers(task_count=len(tasks))
        self._maybe_log_gpu_usage(worker_count=parallel_workers)
        if parallel_workers <= 1:
            self._generate_serial(tasks=tasks, run_path=run_path)
            return run_path

        self._generate_parallel(tasks=tasks, run_path=run_path, num_workers=parallel_workers)
        return run_path

    def _generate_serial(self, tasks: list[PromptTask], run_path: str) -> None:
        for task in tqdm(tasks, desc="Generating audio"):
            self.generate_one(
                prompt=task.prompt,
                save_path=self._create_prompt_path(run_path=run_path, idx=task.idx)
            )

    def _generate_parallel(self, tasks: list[PromptTask], run_path: str, num_workers: int) -> None:
        mp_context = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_context,
            initializer=_init_prompt_worker,
            initargs=(self.__class__, self._get_worker_init_kwargs())
        ) as executor:
            futures = [
                executor.submit(
                    _run_prompt_task,
                    run_path=run_path,
                    task=task
                ) for task in tasks
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating audio"):
                future.result()

    def _resolve_parallel_workers(self, task_count: int) -> int:
        return max(1, min(self.num_workers, task_count))

    def estimate_max_parallel_workers(
        self,
        gpu_utilization_target: float = 0.9,
        reserve_gb: float = 1.0,
        per_worker_overhead_gb: float = 0.5,
        probe_prompt: str | None = None
    ) -> int:
        if self.device.type != "cuda":
            return 1
        if not (0.0 < gpu_utilization_target <= 1.0):
            raise ValueError("gpu_utilization_target must be in the range (0, 1].")

        total_bytes: int
        free_bytes: int
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)

        per_worker_bytes = self._estimate_worker_vram_bytes(probe_prompt=probe_prompt)
        if per_worker_bytes is None or per_worker_bytes <= 0:
            raise ValueError(
                "Unable to estimate per-worker VRAM usage. "
                "Override _estimate_worker_vram_bytes in this generator."
            )

        reserve_bytes = int(reserve_gb * 1024**3)
        per_worker_overhead_bytes = int(per_worker_overhead_gb * 1024**3)
        currently_used_bytes = total_bytes - free_bytes
        budget_bytes = int(total_bytes * gpu_utilization_target) - reserve_bytes
        available_for_workers = budget_bytes - currently_used_bytes
        if available_for_workers <= 0:
            return 1

        per_worker_total = per_worker_bytes + per_worker_overhead_bytes
        if per_worker_total <= 0:
            return 1

        return max(1, available_for_workers // per_worker_total)

    def estimate_gpu_usage_for_workers(
        self,
        worker_count: int,
        per_worker_overhead_gb: float = 0.5,
        probe_prompt: str | None = None
    ) -> dict[str, float]:
        if self.device.type != "cuda":
            raise ValueError("GPU usage estimation requires a CUDA device.")
        if worker_count <= 0:
            raise ValueError("worker_count must be a positive integer.")

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        currently_used_bytes = total_bytes - free_bytes
        per_worker_bytes = self._estimate_worker_vram_bytes(probe_prompt=probe_prompt)
        if per_worker_bytes is None or per_worker_bytes <= 0:
            raise ValueError(
                "Unable to estimate per-worker VRAM usage. "
                "Override _estimate_worker_vram_bytes in this generator."
            )
        per_worker_total = per_worker_bytes + int(per_worker_overhead_gb * 1024**3)
        projected_total_bytes = currently_used_bytes + (worker_count * per_worker_total)

        return {
            "total_gb": total_bytes / 1024**3,
            "currently_used_gb": currently_used_bytes / 1024**3,
            "per_worker_gb": per_worker_total / 1024**3,
            "projected_total_gb": projected_total_bytes / 1024**3,
            "projected_utilization": projected_total_bytes / total_bytes
        }

    def _estimate_peak_generation_vram_bytes(self, prompt: str) -> int | None:
        if self.device.type != "cuda":
            return None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline = torch.cuda.memory_allocated(self.device)

        with tempfile.TemporaryDirectory(prefix="gen_probe_") as tmp_dir:
            self.generate_one(prompt=prompt, save_path=tmp_dir)
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        return max(0, peak - baseline)

    def _maybe_log_gpu_usage(self, worker_count: int) -> None:
        if not self.log_gpu_usage:
            return
        if self.device.type != "cuda":
            return

        try:
            usage = self.estimate_gpu_usage_for_workers(worker_count=worker_count)
            max_workers = self.estimate_max_parallel_workers()
            tqdm.write(
                "GPU usage estimate: "
                f"workers={worker_count}, "
                f"per_worker={usage['per_worker_gb']:.2f} GB, "
                f"projected={usage['projected_total_gb']:.2f}/{usage['total_gb']:.2f} GB "
                f"({usage['projected_utilization'] * 100:.1f}%), "
                f"max_recommended={max_workers}"
            )
        except Exception as error:
            tqdm.write(f"GPU usage estimate is unavailable: {error}")

    def _base_init_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "device": str(self.device),
            "log_gpu_usage": self.log_gpu_usage
        }

    def _get_worker_init_kwargs(self) -> dict[str, Any]:
        worker_kwargs = self._get_init_kwargs().copy()
        worker_kwargs["num_workers"] = 1
        return worker_kwargs

    def _create_run_path(self, output_dir: str) -> str:
        run_path = os.path.join(output_dir, str(uuid.uuid1()))
        os.makedirs(run_path, exist_ok=True)
        return run_path

    def _create_prompt_path(self, run_path: str, idx: int) -> str:
        prompt_path = os.path.join(run_path, f"prompt_{idx}")
        os.makedirs(prompt_path, exist_ok=True)
        return prompt_path


def _init_prompt_worker(
    generator_class: type[MusicGenerator],
    generator_kwargs: dict[str, Any]
) -> None:
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = generator_class(**generator_kwargs)


def _run_prompt_task(
    run_path: str,
    task: PromptTask
) -> str:
    if _WORKER_GENERATOR is None:
        raise RuntimeError("Parallel worker generator is not initialized.")
    save_path = os.path.join(run_path, f"prompt_{task.idx}")
    return _WORKER_GENERATOR.generate_one(prompt=task.prompt, save_path=save_path)
