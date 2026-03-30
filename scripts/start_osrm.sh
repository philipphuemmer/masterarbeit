#!/usr/bin/env bash
# =============================================================================
# OSRM Server starten (nach einmaligem Setup via setup_osrm.sh)
#
# Ausführung (aus dem Projektordner):
#   ./scripts/start_osrm.sh
#
# Server läuft dann auf http://localhost:5000
# Mit Ctrl+C beenden.
# =============================================================================

set -euo pipefail

DATA_DIR="$(pwd)/data/osrm"
OSRM_FILE="$DATA_DIR/bayern.osrm"

if [ ! -f "$OSRM_FILE" ]; then
    echo "[FEHLER] $OSRM_FILE nicht gefunden."
    echo "         Führe zuerst das Setup aus: ./scripts/setup_osrm.sh"
    exit 1
fi

echo "=== Starte OSRM-Server auf http://localhost:5000 ==="
echo "    Beenden mit Ctrl+C"
echo ""

docker run --rm -t -i \
    -p 5000:5000 \
    -v "$DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-routed --algorithm mld --max-table-size 500 /data/bayern.osrm
