from eurostoxx_iv_rv_backtest.scripts.generate_figures import publish_docs_figures


def test_publish_docs_figures_copies_existing_selected_figures(tmp_path) -> None:
    figures_dir = tmp_path / "outputs" / "figures"
    docs_figures_dir = tmp_path / "docs" / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "equity_curve.png").write_bytes(b"fake-png")
    (figures_dir / "iv_vs_rv.png").write_bytes(b"fake-png")

    copied = publish_docs_figures(figures_dir=figures_dir, docs_figures_dir=docs_figures_dir)

    assert docs_figures_dir.joinpath("equity_curve.png").read_bytes() == b"fake-png"
    assert docs_figures_dir.joinpath("iv_vs_rv.png").read_bytes() == b"fake-png"
    assert {path.name for path in copied} == {"equity_curve.png", "iv_vs_rv.png"}
