"""
Бенчмарк grid + Shor на парах convex/nonconvex из HousdorfPolygonGen/build/Debug.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scipy.spatial import cKDTree

from geometry import Point, Polygon
from hausdorff_grid_search import find_optimal_translation_grid
from polygon_hausdorff_fast import default_workers, to_array
from polygon_rasterization import rasterize_polygon
from q0_init import initQ0
from shor_optimize import shor_optimize


def load_polygon(path: Path) -> Polygon:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    n = int(lines[0].strip())
    points: Polygon = []
    for line in lines[1 : n + 1]:
        parts = line.split()
        if len(parts) < 2:
            continue
        points.append(Point(float(parts[0]), float(parts[1])))
    if len(points) != n:
        raise ValueError(f"{path}: ожидалось {n} вершин, прочитано {len(points)}")
    return points


def discover_cases(root: Path) -> list[tuple[str, Path, Path]]:
    cases: list[tuple[str, Path, Path]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        convex_files = list(folder.glob("*_polygon_convex.txt"))
        nonconvex_files = list(folder.glob("*_polygon_nonconvex.txt"))
        if len(convex_files) != 1 or len(nonconvex_files) != 1:
            continue
        cases.append((folder.name, convex_files[0], nonconvex_files[0]))
    return cases


@dataclass
class BenchRow:
    case_id: str
    raster_steps: int
    grid_steps: int
    n_convex: int
    n_nonconvex: int
    n_raster_a: int
    n_raster_b: int
    t_raster_s: float
    t_trees_s: float
    t_grid_s: float
    t_shor_s: float
    t_total_s: float
    grid_dist: float
    shor_dist: float
    shift_x: float
    shift_y: float
    workers: int


def run_case(
    case_id: str,
    convex_path: Path,
    nonconvex_path: Path,
    *,
    raster_steps: int,
    grid_steps: int,
    workers: int | None,
) -> BenchRow:
    A = load_polygon(convex_path)
    B = load_polygon(nonconvex_path)

    t0 = time.perf_counter()
    A_r = rasterize_polygon(A, raster_steps, workers=workers)
    B_r = rasterize_polygon(B, raster_steps, workers=workers)
    t1 = time.perf_counter()

    Q0 = initQ0(A, B)
    A_tree = cKDTree(to_array(A_r))
    B_tree = cKDTree(to_array(B_r))
    t2 = time.perf_counter()

    best_x, best_val, _ = find_optimal_translation_grid(
        A_r, B_r, A_tree, B_tree, Q0, grid_steps, workers=workers
    )
    t3 = time.perf_counter()

    refine_x, refine_dist = shor_optimize(A_r, B_r, A_tree, B_tree, best_x)
    t4 = time.perf_counter()

    w = workers if workers is not None else default_workers()
    return BenchRow(
        case_id=case_id,
        raster_steps=raster_steps,
        grid_steps=grid_steps,
        n_convex=len(A),
        n_nonconvex=len(B),
        n_raster_a=len(A_r),
        n_raster_b=len(B_r),
        t_raster_s=t1 - t0,
        t_trees_s=t2 - t1,
        t_grid_s=t3 - t2,
        t_shor_s=t4 - t3,
        t_total_s=t4 - t0,
        grid_dist=best_val,
        shor_dist=refine_dist,
        shift_x=refine_x.x,
        shift_y=refine_x.y,
        workers=w,
    )


def print_table(rows: list[BenchRow], raster_steps: int, grid_steps: int) -> None:
    if not rows:
        print("Нет кейсов для отображения.")
        return

    w = rows[0].workers
    print(
        f"\nПараметры: raster={raster_steps}, grid={grid_steps}, workers={w}, "
        f"кейсов={len(rows)}\n"
    )
    header = (
        f"{'case':<42} {'|V|':>7} {'rast':>7} "
        f"{'raster':>8} {'grid':>8} {'shor':>8} {'total':>8} "
        f"{'H_grid':>10} {'H_shor':>10} {'dx':>9} {'dy':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.case_id:<42} {r.n_convex:>3}/{r.n_nonconvex:<3} "
            f"{r.n_raster_a:>3}/{r.n_raster_b:<3} "
            f"{r.t_raster_s:8.3f} {r.t_grid_s:8.3f} {r.t_shor_s:8.3f} {r.t_total_s:8.3f} "
            f"{r.grid_dist:10.4f} {r.shor_dist:10.4f} "
            f"{r.shift_x:9.2f} {r.shift_y:9.2f}"
        )

    n = len(rows)
    print("-" * len(header))
    print(
        f"{'ИТОГО (сумма)':<42} {'':>7} {'':>7} "
        f"{sum(r.t_raster_s for r in rows):8.3f} "
        f"{sum(r.t_grid_s for r in rows):8.3f} "
        f"{sum(r.t_shor_s for r in rows):8.3f} "
        f"{sum(r.t_total_s for r in rows):8.3f}"
    )
    print(
        f"{'СРЕДНЕЕ':<42} {'':>7} {'':>7} "
        f"{sum(r.t_raster_s for r in rows) / n:8.3f} "
        f"{sum(r.t_grid_s for r in rows) / n:8.3f} "
        f"{sum(r.t_shor_s for r in rows) / n:8.3f} "
        f"{sum(r.t_total_s for r in rows) / n:8.3f}"
    )


CSV_HEADER = [
    "case_id",
    "raster_steps",
    "grid_steps",
    "hausdorff_optimal",
    "hausdorff_grid",
    "t_total_s",
    "t_raster_s",
    "t_trees_s",
    "t_grid_s",
    "t_shor_s",
    "n_convex",
    "n_nonconvex",
    "n_raster_a",
    "n_raster_b",
    "shift_x",
    "shift_y",
    "workers",
]


def row_to_csv(r: BenchRow) -> list:
    return [
        r.case_id,
        r.raster_steps,
        r.grid_steps,
        f"{r.shor_dist:.8f}",
        f"{r.grid_dist:.8f}",
        f"{r.t_total_s:.6f}",
        f"{r.t_raster_s:.6f}",
        f"{r.t_trees_s:.6f}",
        f"{r.t_grid_s:.6f}",
        f"{r.t_shor_s:.6f}",
        r.n_convex,
        r.n_nonconvex,
        r.n_raster_a,
        r.n_raster_b,
        f"{r.shift_x:.8f}",
        f"{r.shift_y:.8f}",
        r.workers,
    ]


def write_csv(path: Path, rows: list[BenchRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in rows:
            writer.writerow(row_to_csv(r))


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def run_sweep(
    cases: list[tuple[str, Path, Path]],
    raster_values: list[int],
    grid_values: list[int],
    workers: int | None,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    total_runs = len(cases) * len(raster_values) * len(grid_values)
    run_idx = 0
    for raster_steps in raster_values:
        for grid_steps in grid_values:
            for case_id, convex_path, nonconvex_path in cases:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] raster={raster_steps} grid={grid_steps} {case_id} ...",
                    end=" ",
                    flush=True,
                )
                row = run_case(
                    case_id,
                    convex_path,
                    nonconvex_path,
                    raster_steps=raster_steps,
                    grid_steps=grid_steps,
                    workers=workers,
                )
                rows.append(row)
                print(f"H={row.shor_dist:.4f} t={row.t_total_s:.3f}s")
    return rows


def main() -> int:
    default_root = (
        Path(__file__).resolve().parents[2]
        / "HousdorfPolygonGen"
        / "build"
        / "Debug"
    )
    parser = argparse.ArgumentParser(description="Бенчмарк grid+Shor на файлах полигонов")
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Каталог Debug с подпапками кейсов",
    )
    parser.add_argument("--raster", type=int, default=50, help="Плотность растеризации")
    parser.add_argument("--grid", type=int, default=20, help="Шаги сетки (N+1)^2 оценок")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Потоки (по умолчанию — авто)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Путь для сохранения CSV (по умолчанию benchmark_results.csv в --root)",
    )
    parser.add_argument(
        "--compare-workers",
        action="store_true",
        help="Дополнительно прогнать сетку с workers=1 для сравнения",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Прогон всех комбинаций --raster-list × --grid-list",
    )
    parser.add_argument(
        "--raster-list",
        type=str,
        default="50,80,100",
        help="Список плотностей растеризации через запятую (для --sweep)",
    )
    parser.add_argument(
        "--grid-list",
        type=str,
        default="20,50",
        help="Список плотностей сетки через запятую (для --sweep)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Каталог не найден: {root}", file=sys.stderr)
        return 1

    cases = discover_cases(root)
    if not cases:
        print(f"В {root} не найдено пар convex/nonconvex", file=sys.stderr)
        return 1

    print(f"Корень: {root}")
    print(f"Найдено кейсов: {len(cases)}")

    if args.sweep:
        raster_values = parse_int_list(args.raster_list)
        grid_values = parse_int_list(args.grid_list)
        print(f"Sweep: raster={raster_values} grid={grid_values}")
        rows = run_sweep(cases, raster_values, grid_values, args.workers)
        csv_path = args.csv or (root / "benchmark_sweep_raster_grid.csv")
        write_csv(csv_path, rows)
        print(f"\nСводная таблица ({len(rows)} строк): {csv_path}")
        return 0

    rows: list[BenchRow] = []
    for case_id, convex_path, nonconvex_path in cases:
        row = run_case(
            case_id,
            convex_path,
            nonconvex_path,
            raster_steps=args.raster,
            grid_steps=args.grid,
            workers=args.workers,
        )
        rows.append(row)
        print(f"  OK {case_id}  total={row.t_total_s:.3f}s  H={row.shor_dist:.4f}")

    print_table(rows, args.raster, args.grid)

    csv_path = args.csv or (root / "benchmark_grid_shor_results.csv")
    write_csv(csv_path, rows)
    print(f"\nCSV: {csv_path}")

    if args.compare_workers and (args.workers is None or args.workers != 1):
        print("\n--- Сравнение сетки: workers=1 vs параллельно ---")
        seq_grid = 0.0
        par_grid = 0.0
        for case_id, convex_path, nonconvex_path in cases:
            A = load_polygon(convex_path)
            B = load_polygon(nonconvex_path)
            A_r = rasterize_polygon(A, args.raster, workers=1)
            B_r = rasterize_polygon(B, args.raster, workers=1)
            Q0 = initQ0(A, B)
            A_tree = cKDTree(to_array(A_r))
            B_tree = cKDTree(to_array(B_r))

            t0 = time.perf_counter()
            find_optimal_translation_grid(
                A_r, B_r, A_tree, B_tree, Q0, args.grid, workers=1
            )
            t1 = time.perf_counter()
            find_optimal_translation_grid(
                A_r, B_r, A_tree, B_tree, Q0, args.grid, workers=args.workers
            )
            t2 = time.perf_counter()
            seq_grid += t1 - t0
            par_grid += t2 - t1
            print(f"  {case_id}: seq={t1 - t0:.3f}s par={t2 - t1:.3f}s")

        print(f"Сумма grid seq={seq_grid:.3f}s  par={par_grid:.3f}s  ускорение={seq_grid / par_grid:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
