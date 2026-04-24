from eurostoxx_iv_rv_backtest.data.market_data import (
    build_raw_dataset,
    clean_sx5e_data,
    clean_vstoxx_data,
    fetch_sx5e_data,
    fetch_vstoxx_data,
    main,
    merge_spot_and_iv,
)


__all__ = [
    "build_raw_dataset",
    "clean_sx5e_data",
    "clean_vstoxx_data",
    "fetch_sx5e_data",
    "fetch_vstoxx_data",
    "main",
    "merge_spot_and_iv",
]


if __name__ == "__main__":
    main()
