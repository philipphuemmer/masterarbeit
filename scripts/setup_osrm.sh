#!/usr/bin/env bash
# =============================================================================
# OSRM Setup — Einmaliges Preprocessing für Bayern (Würzburg-Datensatz)
#
# Ausführung (aus dem Projektordner):
#   chmod +x scripts/setup_osrm.sh
#   ./scripts/setup_osrm.sh
#
# Danach OSRM-Server starten:
#   ./scripts/start_osrm.sh
# =============================================================================

set -euo pipefail

DATA_DIR="$(pwd)/data/osrm"
PBF_FILE="$DATA_DIR/bayern.osm.pbf"
PBF_URL="https://download.geofabrik.de/europe/germany/bayern-latest.osm.pbf"

echo "=== OSRM Setup für Bayern ==="
echo ""

# 1. Verzeichnis anlegen
mkdir -p "$DATA_DIR"

# 2. Bayern-OSM-Daten herunterladen (wenn noch nicht vorhanden)
if [ -f "$PBF_FILE" ]; then
    echo "[1/4] bayern.osm.pbf bereits vorhanden — überspringe Download."
else
    echo "[1/4] Lade Bayern-OSM-Daten herunter (~800 MB) …"
    wget --progress=bar:force -O "$PBF_FILE" "$PBF_URL"
    echo "      Download abgeschlossen."
fi
echo ""

# 3. OSRM Extract (Straßennetz extrahieren)
echo "[2/4] OSRM Extract (Straßenprofil: Auto) …"
docker run --rm -t \
    -v "$DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-extract -p /opt/car.lua /data/bayern.osm.pbf
echo "      Extract abgeschlossen."
echo ""

# 4. OSRM Partition (Multi-Level Dijkstra Vorbereitung)
echo "[3/4] OSRM Partition …"
docker run --rm -t \
    -v "$DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-partition /data/bayern.osrm
echo "      Partition abgeschlossen."
echo ""

# 5. OSRM Customize
echo "[4/4] OSRM Customize …"
docker run --rm -t \
    -v "$DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-customize /data/bayern.osrm
echo "      Customize abgeschlossen."
echo ""

echo "=== Setup fertig! ==="
echo ""
echo "Starte jetzt den OSRM-Server mit:"
echo "  ./scripts/start_osrm.sh"
echo ""
echo "Dann Matrix berechnen mit:"
echo "  python scripts/build_travel_matrix.py"
