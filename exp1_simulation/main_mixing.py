import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    from exact import MAX_EXACT_N
    from graphs import parse_float_grid, parse_int_grid, parse_interval_grid
    import mixing_core as ms
else:
    from .exact import MAX_EXACT_N
    from .graphs import parse_float_grid, parse_int_grid, parse_interval_grid
    from . import mixing_core as ms


SAVE_ROOT = Path(__file__).resolve().parent / "save" / "mixing"
DEFAULT_SMALL_N_GRID = "4,6,8,10,12,14,16,18,20,22,24"
DEFAULT_U_LAMBDA_GRID = "1,100"
DEFAULT_OMEGA_INTERVALS = "0,0.5;0.5,1.0"
DEFAULT_GRAPH_P_GRID = "0.2,0.8,1.0"
DEFAULT_GRAPH_FAMILIES = ("random_dag", "random_digraph")


def _save_pickle(path: Path, payload: Dict[str, object]) -> None:
    ms._save_pickle(path, payload)


def _ordered_results(results_by_index: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    return [results_by_index[idx] for idx in sorted(results_by_index.keys())]


def _parse_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean flag (True/False), got {value!r}")


def _graph_cases(graph_p_grid: Sequence[float]) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for family in DEFAULT_GRAPH_FAMILIES:
        for p in graph_p_grid:
            cases.append({"family": str(family), "p": float(p)})
    return cases


def _unique_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    seen = set()
    out: List[Tuple[float, float]] = []
    for lo, hi in intervals:
        key = (float(lo), float(hi))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _add_omega0_special_case(
    *,
    include_special: bool,
    family: str,
    p: float,
    u_lambda: float,
) -> bool:
    if not bool(include_special):
        return False
    if float(u_lambda) != 1.0:
        return False

    p_is_one = abs(float(p) - 1.0) <= 1e-12
    fam = str(family)
    if fam == "random_digraph":
        return p_is_one
    if fam == "random_dag":
        return not p_is_one
    return False


def _effective_graph_reps(graph_reps: int, p: float) -> int:
    p_is_one = abs(float(p) - 1.0) <= 1e-12
    if p_is_one:
        return 1
    return int(graph_reps)


def _build_small_tasks_new(
    *,
    n_grid: Sequence[int],
    graph_p_grid: Sequence[float],
    u_lambda_grid: Sequence[float],
    omega_intervals: Sequence[Tuple[float, float]],
    include_p1_u1_omega0_case: bool,
    num_chains: int,
    graph_reps: int,
    init_reps: int,
    epsilon: float,
    epsilon0: float,
    lazy_prob: float,
    seed_base: int,
    persist_k: int,
    greedy_init_samples: int,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    cases = _graph_cases(graph_p_grid)
    case_index = 0

    for n in n_grid:
        horizon = ms._exact_horizon(int(n))
        checkpoints = ms._build_checkpoints_with_exact_horizon(int(horizon))
        for case in cases:
            family = str(case["family"])
            p = float(case["p"])
            reps_run = _effective_graph_reps(int(graph_reps), float(p))

            for graph_rep in range(reps_run):
                graph_seed = ms._seed_from(seed_base, "small_new", n, family, p, graph_rep, "graph")
                for u_lambda in u_lambda_grid:
                    lambda_seed = ms._seed_from(
                        seed_base, "small_new", n, family, p, graph_rep, float(u_lambda), "lambda"
                    )

                    omega_list = [(float(lo), float(hi)) for lo, hi in omega_intervals]
                    if _add_omega0_special_case(
                        include_special=bool(include_p1_u1_omega0_case),
                        family=family,
                        p=float(p),
                        u_lambda=float(u_lambda),
                    ):
                        omega_list.append((0.0, 0.0))
                    omega_list = _unique_intervals(omega_list)

                    for omega_interval in omega_list:
                        omega_seed = ms._seed_from(
                            seed_base,
                            "small_new",
                            n,
                            family,
                            p,
                            graph_rep,
                            float(u_lambda),
                            float(omega_interval[0]),
                            float(omega_interval[1]),
                            "omega",
                        )
                        for init_rep in range(int(init_reps)):
                            case_index += 1
                            tasks.append(
                                {
                                    "case_index": int(case_index),
                                    "mode": "small_new",
                                    "n": int(n),
                                    "graph_family": str(family),
                                    "graph_p": float(p),
                                    "graph_rep": int(graph_rep),
                                    "graph_reps_run_for_family": int(reps_run),
                                    "init_rep": int(init_rep),
                                    "init_reps_run_for_case": int(init_reps),
                                    "u_lambda": float(u_lambda),
                                    "omega_interval": (
                                        float(omega_interval[0]),
                                        float(omega_interval[1]),
                                    ),
                                    "graph_seed": int(graph_seed),
                                    "lambda_seed": int(lambda_seed),
                                    "omega_seed": int(omega_seed),
                                    "horizon": int(horizon),
                                    "checkpoints": [int(t) for t in checkpoints],
                                    "num_chains": int(num_chains),
                                    "epsilon": float(epsilon),
                                    "epsilon0": float(epsilon0),
                                    "lazy_prob": float(lazy_prob),
                                    "persist_k": int(persist_k),
                                    "greedy_init_samples": int(greedy_init_samples),
                                    "shared_init_across_chains": True,
                                }
                            )
    return tasks


def _format_task_brief(task: Dict[str, object], total_cases: int) -> str:
    return (
        f"[small_new] case {int(task['case_index'])}/{int(total_cases)} "
        f"n={int(task['n'])} graph={task['graph_family']} p={task['graph_p']} "
        f"grep={int(task['graph_rep']) + 1}/{int(task['graph_reps_run_for_family'])} "
        f"irep={int(task.get('init_rep', 0)) + 1}/{int(task.get('init_reps_run_for_case', 1))} "
        f"U_lambda={float(task['u_lambda'])} omega={task['omega_interval']}"
    )


def _format_case_done(
    task: Dict[str, object],
    total_cases: int,
    done: int,
    result: Dict[str, object],
) -> str:
    return (
        f"{_format_task_brief(task, total_cases)} done "
        f"({done}/{total_cases} completed, "
        f"elapsed={float(result['elapsed_seconds']):.2f}s; "
        f"{ms._summarize_result_for_log(result)})"
    )


def _run_condition_from_task_new(task: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    result = ms._run_condition(
        n=int(task["n"]),
        graph_family=str(task["graph_family"]),
        graph_p=task["graph_p"],
        graph_rep=int(task["graph_rep"]),
        u_lambda=float(task["u_lambda"]),
        omega_interval=task["omega_interval"],
        graph_seed=int(task["graph_seed"]),
        lambda_seed=int(task["lambda_seed"]),
        omega_seed=int(task["omega_seed"]),
        graph_reps_run_for_family=int(task["graph_reps_run_for_family"]),
        target_graph_reps_for_family=int(task["graph_reps_run_for_family"]),
        horizon=int(task["horizon"]),
        checkpoints=task["checkpoints"],
        num_chains=int(task["num_chains"]),
        epsilon=float(task["epsilon"]),
        epsilon0=float(task["epsilon0"]),
        lazy_prob=float(task["lazy_prob"]),
        persist_k=int(task["persist_k"]),
        greedy_init_samples=int(task["greedy_init_samples"]),
        init_rep=int(task.get("init_rep", 0)),
        shared_init_across_chains=bool(task.get("shared_init_across_chains", True)),
    )
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    result.pop("target_graph_reps_for_family", None)
    return {"case_index": int(task["case_index"]), "result": result}


def _run_small_new(
    *,
    n_grid: Sequence[int],
    graph_p_grid: Sequence[float],
    u_lambda_grid: Sequence[float],
    omega_intervals: Sequence[Tuple[float, float]],
    include_p1_u1_omega0_case: bool,
    num_chains: int,
    graph_reps: int,
    init_reps: int,
    epsilon: float,
    epsilon0: float,
    lazy_prob: float,
    save_dir: Path,
    seed_base: int,
    num_workers: int,
    persist_k: int,
    greedy_init_samples: int,
) -> List[Dict[str, object]]:
    tasks = _build_small_tasks_new(
        n_grid=n_grid,
        graph_p_grid=graph_p_grid,
        u_lambda_grid=u_lambda_grid,
        omega_intervals=omega_intervals,
        include_p1_u1_omega0_case=bool(include_p1_u1_omega0_case),
        num_chains=int(num_chains),
        graph_reps=int(graph_reps),
        init_reps=int(init_reps),
        epsilon=float(epsilon),
        epsilon0=float(epsilon0),
        lazy_prob=float(lazy_prob),
        seed_base=int(seed_base),
        persist_k=int(persist_k),
        greedy_init_samples=int(greedy_init_samples),
    )
    total_cases = len(tasks)
    results_by_index: Dict[int, Dict[str, object]] = {}
    done = 0
    t_start = time.perf_counter()
    partial_path = save_dir / "mixing_partial.pkl"

    if int(num_workers) <= 1:
        for task in tasks:
            print(_format_task_brief(task, total_cases), flush=True)
            payload = _run_condition_from_task_new(task)
            done += 1
            results_by_index[int(payload["case_index"])] = payload["result"]
            print(_format_case_done(task, total_cases, done, payload["result"]), flush=True)
            if done % 5 == 0:
                results = _ordered_results(results_by_index)
                _save_pickle(
                    partial_path,
                    {"mode": "small_new", "num_results": len(results), "results": results},
                )
                print(f"  saved partial: {partial_path}", flush=True)
    else:
        max_workers = min(int(num_workers), max(1, total_cases))
        print(
            f"[small_new] running case-level multiprocessing with num_workers={max_workers}",
            flush=True,
        )
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            future_to_task = {executor.submit(_run_condition_from_task_new, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                payload = future.result()
                done += 1
                results_by_index[int(payload["case_index"])] = payload["result"]
                print(_format_case_done(task, total_cases, done, payload["result"]), flush=True)
                if done % 5 == 0:
                    results = _ordered_results(results_by_index)
                    _save_pickle(
                        partial_path,
                        {"mode": "small_new", "num_results": len(results), "results": results},
                    )
                    print(f"  saved partial: {partial_path}", flush=True)

    results = _ordered_results(results_by_index)
    print(
        f"[small_new] done in {time.perf_counter() - t_start:.2f}s with {len(results)} results",
        flush=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1 small-mode exact-D mixing experiment (new grid)")
    parser.add_argument("--save_dir", type=str, default=str(SAVE_ROOT))
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--small_n_grid", type=str, default=DEFAULT_SMALL_N_GRID)
    parser.add_argument("--graph_p_grid", type=str, default=DEFAULT_GRAPH_P_GRID)
    parser.add_argument("--u_lambda_grid", type=str, default=DEFAULT_U_LAMBDA_GRID)
    parser.add_argument("--omega_intervals", type=str, default=DEFAULT_OMEGA_INTERVALS)
    parser.add_argument(
        "--include_p1_u1_omega0_case",
        type=_parse_bool_flag,
        default=True,
        help=(
            "Whether to include special omega=(0,0) cases: "
            "random_digraph with p=1 and random_dag with p!=1, both at U_lambda=1."
        ),
    )
    parser.add_argument("--num_chains", type=int, default=10000)
    parser.add_argument("--graph_reps", type=int, default=1)
    parser.add_argument("--init_reps", type=int, default=1)
    parser.add_argument("--greedy_init_samples", type=int, default=1000)
    parser.add_argument("--persist_k", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--epsilon0", type=float, default=0.02)
    parser.add_argument("--lazy_prob", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if int(args.graph_reps) <= 0:
        raise ValueError(f"graph_reps must be positive, got {args.graph_reps}")
    if int(args.init_reps) <= 0:
        raise ValueError(f"init_reps must be positive, got {args.init_reps}")
    if int(args.greedy_init_samples) < 0:
        raise ValueError(f"greedy_init_samples must be non-negative, got {args.greedy_init_samples}")
    if int(args.persist_k) <= 0:
        raise ValueError(f"persist_k must be positive, got {args.persist_k}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    small_n_grid = parse_int_grid(args.small_n_grid)
    if any(int(n) > int(MAX_EXACT_N) for n in small_n_grid):
        raise ValueError(
            f"main_mixing.py only supports n <= {MAX_EXACT_N} because exact subset-DP "
            f"is configured for that range, got n_grid={small_n_grid}"
        )

    graph_p_grid = parse_float_grid(args.graph_p_grid)
    u_lambda_grid = parse_float_grid(args.u_lambda_grid)
    omega_intervals = parse_interval_grid(args.omega_intervals)
    graph_cases = _graph_cases(graph_p_grid)

    config = {
        "mode": "small_new",
        "seed_base": int(args.seed_base),
        "small_n_grid": [int(n) for n in small_n_grid],
        "graph_families": list(DEFAULT_GRAPH_FAMILIES),
        "graph_p_grid": [float(x) for x in graph_p_grid],
        "u_lambda_grid": [float(x) for x in u_lambda_grid],
        "omega_intervals": [(float(lo), float(hi)) for lo, hi in omega_intervals],
        "include_p1_u1_omega0_case": bool(args.include_p1_u1_omega0_case),
        "num_chains": int(args.num_chains),
        "graph_reps": int(args.graph_reps),
        "init_reps": int(args.init_reps),
        "greedy_init_samples": int(args.greedy_init_samples),
        "persist_k": int(args.persist_k),
        "horizon_rule": "T(n) = ceil(n^3 log n)",
        "checkpoint_rule": "t_k = 2^k up to the largest power of 2 <= T(n), plus T(n) itself if needed",
        "persist_rule": "persist_k consecutive checkpoints with D_t <= epsilon - epsilon0",
        "epsilon": float(args.epsilon),
        "epsilon0": float(args.epsilon0),
        "lazy_prob": float(args.lazy_prob),
        "num_workers": int(args.num_workers),
        "max_exact_n": int(MAX_EXACT_N),
        "graph_cases": graph_cases,
        "replication_structure": (
            "graph_reps draws distinct graphs; init_reps draws distinct shared initializations "
            "per graph/parameter setup; within each init_rep all num_chains chains share the "
            "same initialization while using different MCMC seeds"
        ),
        "graph_rep_rule": "for p=1.0, effective graph_reps is forced to 1 because the graph is deterministic",
        "special_case": (
            "include omega=(0,0) when U_lambda=1 in: "
            "random_digraph with p=1, and random_dag with p!=1"
        ),
        "timestamp_unix": float(time.time()),
    }

    print("Mixing-new experiment configuration")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    results = _run_small_new(
        n_grid=small_n_grid,
        graph_p_grid=graph_p_grid,
        u_lambda_grid=u_lambda_grid,
        omega_intervals=omega_intervals,
        include_p1_u1_omega0_case=bool(args.include_p1_u1_omega0_case),
        num_chains=int(args.num_chains),
        graph_reps=int(args.graph_reps),
        init_reps=int(args.init_reps),
        epsilon=float(args.epsilon),
        epsilon0=float(args.epsilon0),
        lazy_prob=float(args.lazy_prob),
        save_dir=save_dir,
        seed_base=int(ms._seed_from(args.seed_base, "small_new")),
        num_workers=int(args.num_workers),
        persist_k=int(args.persist_k),
        greedy_init_samples=int(args.greedy_init_samples),
    )

    out_path = save_dir / "mixing.pkl"
    _save_pickle(out_path, {"config": config, "small": results})
    print(f"Saved mixing results to {out_path}")


if __name__ == "__main__":
    main()
