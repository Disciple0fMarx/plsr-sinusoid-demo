# PLSR Sinusoid Demo (Rust)

Educational demonstration that **PLSR discovers the 3 hidden characteristics** (amplitude, frequency, phase) from pure sinusoids and predicts future values **perfectly**.

## Run

```bash
cargo run --release --features plot
```

### Output

```text
Comp          MSE       NMSE       R²_Y      Corr
      1     0.000123   3.42e-01     0.658    0.999
      2     0.000008   2.21e-02     0.978    0.999  0.998
      3     0.000000   1.23e-12     1.000    0.999  0.998  0.997
```

Generates `plsr_demo.png`.

