# KAD-CLAP ----------------------------------------------------

import os
import logging
from pathlib import Path

try:
    from kadtk.emb_loader import cache_embedding_files, EmbeddingLoader  # type: ignore
    from kadtk.kad import KernelAudioDistance  # type: ignore
    from kadtk.model_loader import CLAPModel  # type: ignore
except ImportError:  # Fallback for running from source checkout
    import sys
    repo_root = Path(__file__).resolve().parents[1] / "kadtk"
    if repo_root.exists():
        sys.path.append(str(repo_root))
    from kadtk.emb_loader import cache_embedding_files, EmbeddingLoader  # type: ignore
    from kadtk.kad import KernelAudioDistance  # type: ignore
    from kadtk.model_loader import CLAPModel  # type: ignore


def score_kad_clap(
    path_og,
    path_gen,
    out_csv=None,
    *,
    model_name="clap-2023",
    device="cuda",
    bandwidth=None,
    workers=8,
    force_emb_encode=False,
    force_stats_calc=False,
):
    """Compute KAD score between two class folders using the kadtk API.

    Parameters
    ----------
    path_og : str | Path
        Directory with original/reference audio files.
    path_gen : str | Path
        Directory with generated/eval audio files.
    out_csv : str | Path | None
        Optional CSV filepath to append results. If ``None``, nothing is written.
    model_name : str
        Must be ``"clap-2023"`` (other models not wired here).
    device : str
        Torch device for scoring (e.g., ``"cuda"`` or ``"cpu"``).
    bandwidth : float | None
        Optional fixed bandwidth for the Gaussian kernel. ``None`` uses the adaptive heuristic.
    workers : int
        Number of parallel workers for embedding extraction.
    force_emb_encode : bool
        If True, recompute embeddings even if cached.
    force_stats_calc : bool
        If True, recalc kernel stats even if cached.
    """

    if model_name != "clap-2023":
        raise ValueError("This helper only wires up the CLAP 2023 model.")

    path_og = Path(path_og).expanduser().resolve()
    path_gen = Path(path_gen).expanduser().resolve()
    out_csv_path = Path(out_csv).expanduser().resolve() if out_csv else None

    if not path_og.is_dir():
        raise FileNotFoundError(f"Folder not found: {path_og}")
    if not path_gen.is_dir():
        raise FileNotFoundError(f"Folder not found: {path_gen}")

    # Load model and cache embeddings for both sets
    model = CLAPModel("2023")
    model.load_model()

    # Simple logger to satisfy kadtk expectations
    logger = logging.getLogger("kadtk-wrapper")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    def cache_dir_serial(directory: Path):
        files = list(directory.glob('*.*'))
        loader = EmbeddingLoader(model, audio_load_worker=max(1, workers))
        if force_emb_encode:
            # Remove cached embeddings folder if requested
            emb_path = directory / "embeddings" / model.name
            if emb_path.exists():
                import shutil
                shutil.rmtree(emb_path)
        for f in files:
            loader.cache_embedding_file(f)

    for d in (path_og, path_gen):
        if workers <= 1:
            cache_dir_serial(d)
        else:
            cache_embedding_files(d, model, workers=workers, force_emb_encode=force_emb_encode)

    # Compute KAD directly via the library
    metric = KernelAudioDistance(
        model,
        device=device,
        bandwidth=bandwidth,
        audio_load_worker=workers,
        logger=logger,
        force_stats_calc=force_stats_calc,
    )
    score = float(metric.score(path_og, path_gen))

    # Optionally append to CSV
    if out_csv_path:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        header_needed = not out_csv_path.exists()
        with out_csv_path.open("a", encoding="utf-8") as f:
            if header_needed:
                f.write("model,baseline,eval,score\n")
            f.write(f"{model_name},{path_og},{path_gen},{score}\n")

    return score
