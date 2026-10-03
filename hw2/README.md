# 18-647 HW2: start here

```
hw2/
├── src/          C11 + OpenMP code, Makefile, run_all.sh, plots.py, README.md (design + answers)
├── runs/         program logs (.txt), one per experiment
├── ai/           AI-use disclosure: prompts, session summary (add the exported transcript)
├── plots.xlsx    Excel charts: bandwidth vs cores, one line per rank (+ extra sheets)
├── plots.pdf     the same plots as a PDF
├── canvas_text.txt   paste into the Canvas text box (fill in the hours)
└── package.sh    ./package.sh Lastname  ->  Lastname-hw2.zip in the required layout
```

**The `runs/` and plots in this folder come from the 4-core cloud VM used
during development.** The handout requires one of the listed machines (an
ECE cluster machine, an AWS instance, or PSC Bridges). Rerun `src/run_all.sh`
there; it overwrites `runs/`, `plots.xlsx` and `plots.pdf`.

---

## Option A: run on AWS (recommended)

### 1. Launch the instance (EC2 console -> Launch instance)

| setting | value |
|---|---|
| Name | `hw2` |
| AMI | **Ubuntu Server 24.04 LTS, 64-bit (x86)** |
| Instance type | **`c7i.8xlarge`**: 32 vCPU = 16 physical cores, 64 GiB RAM, about $1.4/h (check current pricing) |
| Key pair | *Create new key pair* -> RSA, `.pem` -> it downloads `hw2-key.pem` |
| Network | allow SSH (port 22) from **My IP** |
| Storage | 30 GiB gp3 |
| Advanced details -> CPU options (optional) | *Threads per core = 1* (hides hyper-threads, so "cores" = vCPUs) |

Notes:

* 16 cores gives a 1..16 core sweep (~1-2 h). If you want more cores, use
  `c7i.16xlarge` (32 cores, 128 GiB), but the run takes longer.
* If launch fails with `VcpuLimitExceeded`, either use `c7i.4xlarge`
  (8 cores, 32 GiB) or request more under *Service Quotas -> EC2 ->
  Running On-Demand Standard instances*.
* AWS Academy / Learner Lab accounts may only allow certain sizes. Pick the
  largest c7i/m7i/c6i/m6i you are allowed that has at least 16 GiB RAM.

### 2. Connect and upload

From your laptop (macOS/Linux terminal, or Windows PowerShell):

```bash
chmod 400 hw2-key.pem                                   # macOS/Linux only
scp -i hw2-key.pem hw2.zip ubuntu@<PUBLIC-IP>:~         # public IP: EC2 console -> instance
ssh -i hw2-key.pem ubuntu@<PUBLIC-IP>
```

(If you cannot SSH, the console's **Connect -> EC2 Instance Connect** opens
a browser terminal. Then upload the zip somewhere you can `wget`, or
`git clone` the repository instead.)

### 3. On the instance: install, run, package

```bash
sudo apt-get update
sudo apt-get install -y build-essential unzip tmux python3-openpyxl python3-matplotlib
unzip hw2.zip && cd hw2/src
make && ./reorder test                     # ~1 min, must print ALL TESTS PASSED

tmux new -s hw2                            # survives SSH disconnects
./run_all.sh 2>&1 | tee ../runs/run_all.log
#   detach: Ctrl-b then d      re-attach later: tmux attach -t hw2
```

When it prints `done`:

```bash
cd ~/hw2 && ./package.sh YourLastname      # -> YourLastname-hw2.zip
```

### 4. Download, then **terminate** the instance

```bash
scp -i hw2-key.pem ubuntu@<PUBLIC-IP>:~/hw2/YourLastname-hw2.zip .
```

EC2 console -> instance -> *Instance state -> Terminate* (otherwise it keeps billing).

---

## Option B: ECE number cluster (needs the CMU VPN)

```bash
scp hw2.zip ANDREWID@ece0XX.ece.local.cmu.edu:~
ssh ANDREWID@ece0XX.ece.local.cmu.edu
unzip hw2.zip && cd hw2/src
python3 -m pip install --user -r requirements.txt
tmux new -s hw2        # or: screen -S hw2
./run_all.sh 2>&1 | tee ../runs/run_all.log
```

Check `nproc` and `free -g` first, and pick a machine nobody else is
loading (`top`). Shared machines give noisy numbers.

## Option C: run locally (laptop/desktop)

**Linux, or Windows with WSL2 (Ubuntu):**

```bash
sudo apt-get install -y build-essential python3-openpyxl python3-matplotlib
cd hw2/src && make && ./reorder test
LOG2N=30 ./run_all.sh     # 2^30 = 1 GiB cube if you have < 12 GiB RAM; omit LOG2N for 4 GiB
```

**macOS:** Apple's clang has no OpenMP, so use Homebrew gcc:

```bash
brew install gcc && pip3 install -r requirements.txt
cd hw2/src && CC=gcc-15 make && ./reorder test     # use the gcc-NN that brew installed
CC=gcc-15 LOG2N=30 ./run_all.sh
```

(On Apple Silicon the Part 3 SSE2 kernel is compiled out automatically; the
scalar buffered tile kernel is used instead.)

---

## After the run

1. Look at `plots.pdf`. Copy the "iterative" table from `plots.xlsx` (cores x
   rank, GB/s) into the course's Excel template if they want that file
   specifically.
2. Fill in the hours in `canvas_text.txt`. Update the numbers quoted there
   from your new `runs/` (lines that say "dev VM").
3. Export the AI transcript into `ai/` (see `ai/README.md`), then run
   `./package.sh YourLastname` and submit the zip plus the Canvas text.
