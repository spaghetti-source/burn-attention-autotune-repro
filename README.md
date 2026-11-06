# Burn CUDA attention autotune reproduction

Standalone FP32 attention reproduction using Burn `0.23.0-pre.1` and the included
`Cargo.lock`. Requires a CUDA-capable NVIDIA GPU and a compatible Rust toolchain.

Arguments: batch size, number of heads, query length, key length, head dimension,
value dimension.

```sh
# Fails on NVIDIA L4 / EC2 g6.xlarge:
cargo run --locked --release -- 1 3 513 16 16 10

# Passes on the same GPU; only the value dimension changes:
cargo run --locked --release -- 1 3 513 16 16 16
```
