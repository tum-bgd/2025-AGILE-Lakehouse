# Point Cloud CLI

## Requirements

- Python >=3.10
- Rust 1.84
- PDAL (optional)
- PotreeConverter 2.0 (optional)

## Data

Extract of about 200 million points from AHN4 near Maastricht (single file).

```sh
python scripts/ahn.py --dataset AHN4 --filenames C_69AZ1
```

Extract of about 2B points form AHN3 around Delft (4 files).

```sh
python scripts/ahn.py --dataset AHN3 --filenames C_37EN1 C_37EN2 C_37EZ1 C_37EZ2
```

## Experiments

### Storage Footprint

To evaluate the storage footprint, convert a laz file with various configurations:

```sh
# grid rounded coordinates, uncompressed
cargo run --release -- convert -i ./data/AHN4/C_69AZ1.LAZ -o ./data/AHN4/C_69AZ1_i32.parquet --raw
# resolved coordinates, uncompressed
cargo run --release -- convert -i ./data/AHN4/C_69AZ1.LAZ -o ./data/AHN4/C_69AZ1.parquet
# resolved coordinates, uncompressed, with importance
cargo run --release -- convert -i ./data/AHN4/C_69AZ1.LAZ -o ./data/AHN4/C_69AZ1_importance.parquet --importance
# grid rounded coordinates, zstd compression
cargo run --release -- convert -i ./data/AHN4/C_69AZ1.LAZ -o ./data/AHN4/C_69AZ1_zstd_i32.parquet --compression 'zstd(3)' --raw
# resolved coordinates, zstd compression
cargo run --release -- convert -i ./data/AHN4/C_69AZ1.LAZ -o ./data/AHN4/C_69AZ1_zstd.parquet --compression 'zstd(3)'
# resolved coordinates, zstd compression, with importance
cargo run --release -- convert -i ./data/AHN4/C_69AZ1.LAZ -o ./data/AHN4/C_69AZ1_zstd_importance.parquet --compression 'zstd(3)' --importance
```

Calculate storage amplification with `converted` / `laz` file sizes.

For comparison with LAS you may use `pdal translate ./data/AHN4/C_69AZ1.LAZ ./data/AHN4/C_69AZ1.LAS`.

For comparison with Potree you may use `PotreeConverter(.exe) ./data/AHN4/C_69AZ1.LAZ -o ./data/AHN4/C_69AZ1_potree`.

### Data Loading

Partitioning and query execution:

```sh
cargo run -p pc-cli --example partition --release
```

### Visualization Workload

Vizualization workload:

```sh
cargo run -p pc-cli --example viz --release
```

## Evaluation

Appart from the outputs in the console, an evaluation script is provided: [../scripts/evaluation.py](../scripts/evaluation.py)
