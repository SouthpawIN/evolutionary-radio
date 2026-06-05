#!/bin/bash
# Evolutionary Radio - Start Script
# Usage: ./start_radio.sh start --vibe="chill beats for coding"
# Options:
#   --vibe="..."        Music vibe (default: "chill lofi beats for coding")
#   --use-cache         Start from cached tracks (faster startup)
#   --pre-generate      Generate 3 tracks before playing

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
NC='\033[0m' # No Color

echo -e "${GREEN}🎵 Evolutionary Radio${NC}"
echo "------------------------"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 not found${NC}"
    echo "Install Python 3: https://www.python.org/downloads/"
    exit 1
fi

# Check if mpv is installed
if ! command -v mpv &> /dev/null; then
    echo -e "${RED}❌ Error: mpv not found${NC}"
    echo "Install mpv:"
    echo "  macOS: brew install mpv"
    echo "  Linux: sudo apt install mpv"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Check if requirements are installed
if ! python3 -c "import torch" &> /dev/null; then
    echo -e "${YELLOW}📦 Installing dependencies (this may take a few minutes)...${NC}"
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Check if radio.py exists
if [ ! -f "$RADIO_PY" ]; then
    echo -e "${RED}❌ Error: radio.py not found${NC}"
    exit 1
fi

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
