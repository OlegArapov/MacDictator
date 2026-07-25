"""Issue #3: выбор модели транскрибации large-v3 / turbo."""
import app


def test_turbo_in_models():
    assert app.MLX_MODELS.get("turbo") == "mlx-community/whisper-large-v3-turbo"
    assert app.MLX_MODELS.get("large-v3") == "mlx-community/whisper-large-v3-mlx"


def test_every_model_has_size():
    for name, repo in app.MLX_MODELS.items():
        assert repo in app.MLX_MODEL_MB, f"нет размера для {name} ({repo})"


def test_migration_does_not_clobber_turbo():
    # старая миграция мапила "turbo" → "large-v3" (модели ещё не было);
    # теперь turbo — валидный выбор и сбрасываться не должен
    assert "turbo" not in app._SETTINGS_MIGRATION["model"]


def test_downloaded_check_uses_per_model_size(tmp_path, monkeypatch):
    # снапшот ~1.5 ГБ: для turbo этого достаточно, для large-v3 (порог
    # 90% от 3 ГБ) — нет; глобальный размер под large-v3 ломал бы turbo
    snap = tmp_path / "snap"
    snap.mkdir()
    with open(snap / "weights.npz", "wb") as fh:
        fh.truncate(1500 * 1024 * 1024)  # sparse, места не ест
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **kw: str(snap))
    assert app._mlx_model_downloaded(app.MLX_MODELS["turbo"])
    assert not app._mlx_model_downloaded(app.MLX_MODELS["large-v3"])
