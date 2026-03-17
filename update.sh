#!/bin/bash
set -euo pipefail

DEMO_NAME="ARIES Fire Detection Demo"
DEMO_DIR_NAME="aries-fire-detection-demo"

if [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  USER_HOME="$HOME"
fi

if [ "$EUID" -eq 0 ] && [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  RUN_AS_USER="$SUDO_USER"
else
  RUN_AS_USER="$USER"
fi

run_as_user() {
  if [ "$EUID" -eq 0 ] && [ "${SUDO_USER-}" ] && [ "$RUN_AS_USER" != "root" ]; then
    sudo -H -u "$RUN_AS_USER" "$@"
  else
    "$@"
  fi
}

APP_DIR="$USER_HOME/$DEMO_DIR_NAME"

cd "$APP_DIR"

# Green banner text
printf '\033[1;32m=========== %s ============\033[0m\n' "$DEMO_NAME"

sudo apt install -y linux-headers-$(uname -r) build-essential

# Add Mobilint's official GPG key:
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc
sudo chmod a+r /etc/apt/keyrings/mblt.asc

# Add the repository to apt sources:
printf "%s\n" \
  "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt \
    stable multiverse" |
  sudo tee /etc/apt/sources.list.d/mobilint.list >/dev/null

# Update available packages
sudo apt update

# Install driver & utilities
sudo apt install -y mobilint-aries-driver mobilint-qb-runtime mobilint-cli

CREDENTIAL_TIMEOUT="${GIT_CREDENTIAL_TIMEOUT:-3600}"

echo "Configuring Git credential cache (timeout: ${CREDENTIAL_TIMEOUT}s)..."
if ! git config --global credential.helper "cache --timeout=${CREDENTIAL_TIMEOUT}"; then
  echo "Failed to configure Git credential cache for user $USER."
  exit 1
fi

if [ "$RUN_AS_USER" != "$USER" ]; then
  if ! run_as_user git config --global credential.helper "cache --timeout=${CREDENTIAL_TIMEOUT}"; then
    echo "Failed to configure Git credential cache for user $RUN_AS_USER."
    exit 1
  fi
fi

run_as_user git pull

echo "Preparing build..."

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs not found. Install git-lfs to fetch LFS assets (mxq/mp4)."
else
  echo "Pulling LFS assets (mxq/mp4) from repository..."
  run_as_user git lfs pull
fi

echo "Installing system dependency (libopencv-dev)..."
if ! sudo apt-get install -y libopencv-dev; then
  echo "Failed to install libopencv-dev. Please resolve the apt issue and rerun."
  exit 1
fi

echo "Installing build dependency (cmake)..."
if ! sudo apt-get install -y cmake; then
  echo "Failed to install cmake. Please resolve the apt issue and rerun."
  exit 1
fi

echo "Starting build..."

if [ ! -d build ]; then
  run_as_user mkdir build
  cd build
else
  cd build
fi

if run_as_user cmake ..; then
  echo "CMake completed successfully."
else
  echo "CMake failed."
  exit 1
fi

if run_as_user make -j $(nproc); then
  echo "Build completed successfully."
else
  echo "Build failed."
  exit 1
fi

cd "$APP_DIR"

echo "Updating desktop shortcut..."
# delete old desktop file
if [ -f /usr/share/applications/fire_detection_demo.desktop ]; then
  sudo rm /usr/share/applications/fire_detection_demo.desktop
fi
if [ -f /usr/share/applications/fire-detection-detection.desktop ]; then
  sudo rm /usr/share/applications/fire-detection-demo.desktop
fi
sudo mkdir -p "$USER_HOME/.local/share/applications/" || {
  echo "Failed to create desktop directory at $USER_HOME/.local/share/applications/."
  exit 1
}
sudo cp *.desktop "$USER_HOME/.local/share/applications/"
if [ $? -eq 0 ]; then
  echo "Updating desktop shortcut completed successfully."
else
  echo "Updating desktop shortcut failed."
  exit 1
fi

echo "Updating desktop icon..."
sudo mkdir -p "$USER_HOME/.icons/" || {
  echo "Failed to create icon directory at $USER_HOME/.icons/."
  exit 1
}
sudo cp *.png "$USER_HOME/.icons/"
if [ $? -eq 0 ]; then
  echo "Updating desktop icon completed successfully."
else
  echo "Updating desktop icon failed."
  exit 1
fi

exit 0
