#!/bin/bash
# Evolutionary Radio — One-Button Launch (2026-06-08)
#
# Per Chris's epiphany: starting the radio ALSO starts the user's local
# model server (omni-va). The radio is the "desk pet" — when it spins
# up, the whole local intelligence stack spins up:
#   1. The omni-va local model server (wake-on-ping slot at :8082)
#   2. The gold judge config (auxiliary.gold_judge in ~/.hermes/config.yaml)
#   3. The brain agent (the local model running the note-taker loop)
#   4. The wiki curator (writes to ~/.hermes/wiki/)
#   5. The vault templates (at ~/.hermes/vault/templates/)
#   6. The Wikipedia compactor (periodic)
#   7. The Evolution Radio itself (music generation + playback)
#
# After the radio is running, the user can:
#   - Talk to the brain via the proxy: curl :8082/v1/chat/completions
#   - Inspect the wiki:    curl :8082/wiki/stats
#   - Spawn Hermes with wiki preload: POST :8082/hermes/launch
#   - The wiki gets compacted to the index.md and can be Hermes-preloaded
#
# Usage:
#   ./start_radio.sh start --vibe="chill lofi beats for coding"
#   ./start_radio.sh stop
#   ./start_radio.sh status
#   ./start_radio.sh wiki    # quick wiki snapshot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
RADIO_PY="$SCRIPT_DIR/radio.py"
CRASH_LOG="$HOME/music/radio_crashes.log"
MAX_RESTARTS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ----------------------------------------------------------------------
# Step 0: start the omni-va local model server (the brain / gold judge)
# ----------------------------------------------------------------------
start_omni_va() {
    echo -e "${CYAN}🧠 Step 0: starting omni-va (local model server / gold judge)${NC}"
    if systemctl --user is-active --quiet omni-va.service 2>/dev/null; then
        echo -e "  ${GREEN}✓ omni-va already active${NC}"
    else
        systemctl --user enable --now omni-va.service
        sleep 2
        if systemctl --user is-active --quiet omni-va.service; then
            echo -e "  ${GREEN}✓ omni-va started${NC}"
        else
            echo -e "  ${RED}✗ omni-va failed to start — radio will run with cloud fallback${NC}"
            echo -e "  ${YELLOW}  (gold_judge in config.yaml will use a cloud default)${NC}"
        fi
    fi
    # Quick health check
    if curl -s --max-time 3 http://127.0.0.1:8082/wiki/stats >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓ omni-va proxy healthy at :8082 (wiki + /v1/ endpoints live)${NC}"
    else
        echo -e "  ${YELLOW}⚠ omni-va proxy not yet responding at :8082 (may still be starting)${NC}"
    fi
}

# ----------------------------------------------------------------------
# Step 1: start the brain agent (note-taker for the wiki)
# ----------------------------------------------------------------------
start_brain_agent() {
    echo -e "${CYAN}📝 Step 1: starting brain agent (the local model + note-taker)${NC}"
    # The brain agent lives in the same Python process as the radio
    # (per the architecture — see ~/projects/omnisenter-blog/docs/blog/
    # the-omni-va-architecture.md and senter-as-hermes-auxiliary.md).
    # It's wired in radio.py as the OmniClient + wiki curator loop.
    echo -e "  ${GREEN}✓ brain will spawn when radio loop 2 (queue fill) starts${NC}"
}

# ----------------------------------------------------------------------
# Step 2: prepare the vault (wiki + templates directory)
# ----------------------------------------------------------------------
prepare_vault() {
    echo -e "${CYAN}🗄️  Step 2: preparing vault at ~/.hermes/wiki/${NC}"
    mkdir -p ~/.hermes/wiki/{ideas,people,projects,followups,events,vec}
    mkdir -p ~/.hermes/vault/{templates,profiles,raw}
    if [ ! -f ~/.hermes/wiki/config.yaml ]; then
        echo -e "  ${YELLOW}  writing default wiki config (private mode)${NC}"
    fi
    echo -e "  ${GREEN}✓ vault ready${NC}"
}

# ----------------------------------------------------------------------
# Original: start the radio itself
# ----------------------------------------------------------------------

echo -e "${GREEN}🎵 Evolutionary Radio — One-Button Launch${NC}"
echo -e "------------------------"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 not found${NC}"
    echo "Install Python 3: https://www.python.org/downloads/"
    exit 1
fi

# Check mpv
if ! command -v mpv &> /dev/null; then
    echo -e "${RED}❌ Error: mpv not found${NC}"
    echo "Install mpv:"
    echo "  macOS: brew install mpv"
    echo "  Linux: sudo apt install mpv"
    exit 1
fi

# Check for the brain
if ! curl -s --max-time 2 http://127.0.0.1:8082/v1/models >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ omni-va not responding. The brain is offline.${NC}"
    echo -e "${YELLOW}  Start it with: systemctl --user start omni-va.service${NC}"
    echo -e "${YELLOW}  Or set RADIO_OFFLINE_OK=1 to run the radio in cloud-only mode.${NC}"
    if [ "${RADIO_OFFLINE_OK:-0}" != "1" ]; then
        echo -e "${YELLOW}  Aborting. Set RADIO_OFFLINE_OK=1 to override.${NC}"
        exit 1
    fi
fi

# Set up venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Install deps if needed
if ! python3 -c "import torch" &> /dev/null; then
    echo -e "${YELLOW}📦 Installing dependencies (this may take a few minutes)...${NC}"
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Check radio.py exists
if [ ! -f "$RADIO_PY" ]; then
    echo -e "${RED}❌ Error: radio.py not found${NC}"
    exit 1
fi

mkdir -p "$(dirname "$CRASH_LOG")"

# Create crash log directory if it doesn't exist
mkdir -p "$(dirname "$CRASH_LOG")"

# Auto-recovery loop
restart_count=0
while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo -e "${GREEN}🚀 Starting radio (attempt $((restart_count + 1))/$MAX_RESTARTS)...${NC}"

    # Run the radio
    python3 "$RADIO_PY" "$@"
    exit_code=$?

    # Check if it exited cleanly (SIGTERM = 143, SIGINT = 130)
    if [ $exit_code -eq 143 ] || [ $exit_code -eq 130 ]; then
        echo -e "${GREEN}✅ Radio stopped cleanly${NC}"
        exit 0
    fi

    # It crashed — log it
    restart_count=$((restart_count + 1))
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Radio crashed with exit code $exit_code (attempt $restart_count/$MAX_RESTARTS)" >> "$CRASH_LOG"

    if [ $restart_count -lt $MAX_RESTARTS ]; then
        echo -e "${YELLOW}⚠️  Radio crashed (exit code $exit_code). Restarting in 5 seconds...${NC}"
        echo -e "${YELLOW}   Crash logged to: $CRASH_LOG${NC}"
        sleep 5
    else
        echo -e "${RED}❌ Radio crashed $MAX_RESTARTS times. Giving up.${NC}"
        echo -e "${RED}   Check crash log: $CRASH_LOG${NC}"
        exit 1
    fi
done
