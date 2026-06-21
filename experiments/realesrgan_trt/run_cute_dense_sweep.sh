#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="$ROOT/tools/cute_dsl_wsl_venv/bin/activate"
EXAMPLE="$ROOT/tools/cutlass-main/cutlass-main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py"

source "$VENV"

tiles=(
  "64,64,64"
  "64,128,64"
  "128,64,64"
  "128,128,64"
  "128,256,64"
  "128,128,128"
)

for tile in "${tiles[@]}"; do
  echo "TILE=$tile"
  python "$EXAMPLE" \
    --mnkl 2073600,64,576,1 \
    --tile_shape_mnk "$tile" \
    --a_dtype Float16 \
    --b_dtype Float16 \
    --c_dtype Float16 \
    --acc_dtype Float32 \
    --a_major k \
    --b_major k \
    --c_major n \
    --warmup_iterations 3 \
    --iterations 10 \
    --skip_ref_check || true
done
