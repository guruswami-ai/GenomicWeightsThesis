#!/bin/bash
#
# Experiment Archive Generator
# Creates a complete, self-contained archive of all experiment data
#
# Output: experiments/YYYYMMDD_HHMMSS_{strategy}/
#
set -e

# Configuration
EXPERIMENT_NAME="${1:-experiment}"
STRATEGY="${2:-all}"
SEED="${3:-unknown}"
SOURCE_DIR="$HOME/genomic_evo"
ARCHIVE_BASE="$HOME/genomic_evo/experiments"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="${ARCHIVE_BASE}/${TIMESTAMP}_${STRATEGY}_s${SEED}"

echo "============================================================="
echo "EXPERIMENT ARCHIVE GENERATOR"
echo "============================================================="
echo "Timestamp:   $(date)"
echo "Strategy:    ${STRATEGY}"
echo "Seed:        ${SEED}"
echo "Archive:     ${ARCHIVE_DIR}"
echo "============================================================="

# Create archive directory structure
mkdir -p "${ARCHIVE_DIR}"/{source,config,results,logs,validation,hardware,checkpoints}

echo "1. Archiving source code..."
# Copy all Python source files
cp "${SOURCE_DIR}"/*.py "${ARCHIVE_DIR}/source/" 2>/dev/null || echo "   No .py files"
cp "${SOURCE_DIR}"/*.sh "${ARCHIVE_DIR}/source/" 2>/dev/null || echo "   No .sh files"

# Git info if available
if [ -d "${SOURCE_DIR}/.git" ]; then
    git -C "${SOURCE_DIR}" log -1 --format="%H%n%ai%n%s" > "${ARCHIVE_DIR}/source/git_commit.txt"
    git -C "${SOURCE_DIR}" diff > "${ARCHIVE_DIR}/source/git_diff.txt" 2>/dev/null
fi

echo "2. Archiving configuration..."
# Copy config files
cp "${SOURCE_DIR}"/config_*.json "${ARCHIVE_DIR}/config/" 2>/dev/null || echo "   No config files"

# Create master config
cat > "${ARCHIVE_DIR}/config/experiment_metadata.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "strategy": "${STRATEGY}",
    "seed": "${SEED}",
    "generator_version": "1.0",
    "archive_path": "${ARCHIVE_DIR}",
    "source_dir": "${SOURCE_DIR}"
}
EOF

echo "3. Archiving results..."
# Copy result files
cp "${SOURCE_DIR}"/results_*.json "${ARCHIVE_DIR}/results/" 2>/dev/null || echo "   No result files"
cp "${SOURCE_DIR}"/*.csv "${ARCHIVE_DIR}/results/" 2>/dev/null || echo "   No CSV files"

echo "4. Archiving logs..."
# Copy log files
cp "${SOURCE_DIR}"/*.log "${ARCHIVE_DIR}/logs/" 2>/dev/null || echo "   No log files"

echo "5. Archiving validation reports..."
cp "${SOURCE_DIR}"/validation_*.json "${ARCHIVE_DIR}/validation/" 2>/dev/null || echo "   No validation files"

echo "6. Collecting hardware information..."
# Hardware profile
cat > "${ARCHIVE_DIR}/hardware/system_info.txt" << EOF
SYSTEM INFORMATION
==================
Date: $(date)
Hostname: $(hostname)
OS: $(sw_vers -productName) $(sw_vers -productVersion)

HARDWARE:
$(system_profiler SPHardwareDataType 2>/dev/null | grep -E "Chip|Memory|Cores|Model")

MEMORY STATUS:
$(vm_stat | head -15)

CPU USAGE (snapshot):
$(top -l 1 | head -10)
EOF

# Python environment
cat > "${ARCHIVE_DIR}/hardware/python_env.txt" << EOF
PYTHON ENVIRONMENT
==================
Python: $(python3 --version 2>&1)
Location: $(which python3)

INSTALLED PACKAGES:
$(pip3 freeze 2>/dev/null || echo "pip3 not available")
EOF

echo "7. Creating experiment summary..."
# Generate summary
cat > "${ARCHIVE_DIR}/EXPERIMENT_SUMMARY.md" << EOF
# Experiment Archive

## Metadata
- **Timestamp**: $(date -Iseconds)
- **Hostname**: $(hostname)
- **Strategy**: ${STRATEGY}
- **Seed**: ${SEED}
- **Archive Path**: ${ARCHIVE_DIR}

## Directory Structure
\`\`\`
${ARCHIVE_DIR}/
├── source/           # All Python/shell source code
├── config/           # Run configurations (JSON)
├── results/          # Fitness histories, final results
├── logs/             # Execution logs
├── validation/       # Integrity validation reports
├── hardware/         # System specs and environment
└── checkpoints/      # Evolution checkpoints (if any)
\`\`\`

## Reproducibility
To reproduce this experiment:
1. Copy \`source/\` to target machine
2. Install dependencies: \`pip install -r source/requirements.txt\`
3. Run: \`python main.py --strategy ${STRATEGY} --seed ${SEED} --generations N\`

## Files Included
$(ls -la "${ARCHIVE_DIR}"/*/ 2>/dev/null | head -50)
EOF

echo "8. Creating requirements.txt..."
pip3 freeze > "${ARCHIVE_DIR}/source/requirements.txt" 2>/dev/null || echo "   Could not generate requirements"

echo ""
echo "============================================================="
echo "ARCHIVE COMPLETE"
echo "============================================================="
echo "Location: ${ARCHIVE_DIR}"
echo "Size: $(du -sh "${ARCHIVE_DIR}" | cut -f1)"
echo ""
echo "Contents:"
ls -la "${ARCHIVE_DIR}"
echo "============================================================="
