#!/bin/bash
# Run this on the HPC to discover your cluster's configuration.
# Usage: bash discover_hpc.sh

echo "============================================"
echo "HPC Configuration Discovery"
echo "============================================"

echo ""
echo "--- 1. Available partitions ---"
sinfo -s 2>/dev/null || echo "sinfo not available"

echo ""
echo "--- 2. Partition details (GPU nodes) ---"
sinfo -N -l 2>/dev/null | head -30 || echo "N/A"

echo ""
echo "--- 3. Available GPU resources ---"
sinfo -o "%P %G %N %C" 2>/dev/null || echo "N/A"

echo ""
echo "--- 4. Python / Conda / Module availability ---"
echo "which python3: $(which python3 2>/dev/null || echo 'not found')"
echo "which conda:   $(which conda 2>/dev/null || echo 'not found')"
echo "which module:  $(which module 2>/dev/null || echo 'not found')"

echo ""
echo "--- 5. Available modules (Python/CUDA related) ---"
module avail 2>&1 | grep -iE "python|cuda|anaconda|miniconda|gcc" || echo "No module system or no matches"

echo ""
echo "--- 6. Current disk usage ---"
lquota 2>/dev/null || df -h ~ 2>/dev/null | head -5 || echo "N/A"

echo ""
echo "--- 7. Scratch / work directory ---"
echo "HOME: $HOME"
ls -d /scratch/$USER 2>/dev/null && echo "Scratch exists" || echo "No /scratch/$USER"
ls -d /work/$USER 2>/dev/null && echo "Work exists" || echo "No /work/$USER"
ls -d /tmp/$USER 2>/dev/null && echo "Tmp exists" || echo "No /tmp/$USER"

echo ""
echo "============================================"
