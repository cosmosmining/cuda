#!/usr/bin/env bash
# 18-647 HW2: build, test, measure and plot.  Writes ../runs/*.txt,
# ../plots.xlsx and ../plots.pdf.  Run it inside tmux/screen on the target
# machine; the full run takes roughly 2-2.5 hours on 16 cores (REPS=2: ~1.5-2 h).
#
# Environment knobs (all optional):
#   CORES=16        number of physical cores to scale to (default: all physical cores)
#   THREADS=1-16    explicit thread-count list (default 1-$CORES)
#   REPS=3          repetitions per configuration (best one is reported)
#   LOG2N=32        problem size 2^LOG2N bytes (default 2^32 = 4 GiB per buffer)
#   SKIP_TEST=1     skip the correctness suite
set -euo pipefail
cd "$(dirname "$0")"
RUNS=../runs
mkdir -p "$RUNS"

make

physical_cores() {
    local c=0
    if command -v lscpu >/dev/null; then
        c=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    elif command -v sysctl >/dev/null; then              # macOS
        c=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 0)
    fi
    if [ "${c:-0}" -lt 1 ]; then c=$(getconf _NPROCESSORS_ONLN); fi
    echo "$c"
}
mem_gib() {
    if [ -r /proc/meminfo ]; then
        awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo
    else                                                  # macOS
        echo $(( $(sysctl -n hw.memsize) / 1073741824 ))
    fi
}
CORES=${CORES:-$(physical_cores)}
THREADS=${THREADS:-1-$CORES}
REPS=${REPS:-3}
LOG2N=${LOG2N:-32}
export OMP_PROC_BIND=close OMP_PLACES=cores   # one thread per physical core

mem_gib=$(mem_gib)
need_gib=$(( (2 << (LOG2N - 30 > 0 ? LOG2N - 30 : 0)) ))   # in + out
echo "machine: $(getconf _NPROCESSORS_ONLN) logical CPUs, $CORES physical cores used, ${mem_gib} GiB RAM"
if [ "$mem_gib" -lt $(( need_gib + 1 )) ]; then
    echo "not enough memory for LOG2N=$LOG2N (needs ~${need_gib} GiB); try LOG2N=30" >&2
    exit 1
fi

{
    echo "date: $(date)"
    echo "host: $(hostname)"
    echo "cores used: $CORES   threads: $THREADS   reps: $REPS   N = 2^$LOG2N"
    echo "OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
    uname -a
    ${CC:-gcc} --version | head -1
    echo
    lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    echo
    free -g 2>/dev/null || echo "memory: $(mem_gib) GiB"
} > "$RUNS/machine.txt"

# 1. correctness (all six implementations, 1..7 threads)
if [ -z "${SKIP_TEST:-}" ]; then
    ./reorder test | tee "$RUNS/test.txt"
fi

# 2. main measurement: the parallel implementation (iterative) on 1..p cores,
#    every square interpretation of the 2^LOG2N-byte cube
./reorder bench -a iterative -t "$THREADS" -r "$REPS" -l "$LOG2N" | tee "$RUNS/bench_iterative.txt"

# 3. Part 3: the blocked implementation on 1..p cores
./reorder bench -a blocked -t "$THREADS" -r "$REPS" -l "$LOG2N" | tee "$RUNS/bench_blocked.txt"

# 4. every implementation at p cores (stages needs a third buffer)
ALGS=index,recursive,recursive_nd
if [ "$mem_gib" -ge $(( need_gib * 3 / 2 + 2 )) ]; then ALGS=$ALGS,stages; fi
./reorder bench -a "$ALGS" -t "$CORES" -r 1 -l "$LOG2N" | tee "$RUNS/bench_all_algorithms.txt"

# 5. runtime as a function of problem size (p cores)
SIZES=16,20,24,28
[ "$LOG2N" -gt 28 ] || SIZES=16,20,24
./reorder bench -a iterative,blocked -t "$CORES" -r "$REPS" -l "$SIZES" | tee "$RUNS/size_sweep.txt"

# 6. Part 3 exploration: the five tile kernels (1 and p cores)
for k in 0 1 2 3 4; do
    ${CC:-gcc} -O3 -march=native -std=c11 -fopenmp -DBLOCK_KERNEL=$k -o reorder_k$k main.c reorder.c
done
for k in 0 1 2 3 4; do
    echo "### BLOCK_KERNEL=$k"
    ./reorder_k$k bench -a blocked -t "1,$CORES" -r 2 -l "$LOG2N"
done | tee "$RUNS/part3_kernel_ladder.txt"

# 7. plots
python3 plots.py
echo "done: see ../runs/, ../plots.xlsx, ../plots.pdf"
