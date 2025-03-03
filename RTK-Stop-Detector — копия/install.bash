#!/usr/bin bash
#
# install.bash
# A script to install CUDA runtime, GCC toolchains, and Python packages on Ubuntu 20.04 (ARM64).


# Exit immediately if a command exits with a non-zero status.
set -e

# ANSI color codes
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
BOLD="\033[1m"
RESET="\033[0m"

# A function to log and run commands with color-coded output.
log_and_run() {
    echo -e "${YELLOW}==================================================${RESET}"
    echo -e "${CYAN}[INFO]${RESET} Running command: ${BOLD}$*${RESET}"
    "$@"
    local STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo -e "${RED}[ERROR]${RESET} Command failed with exit code $STATUS: ${BOLD}$*${RESET}"
        exit $STATUS
    else
        echo -e "${GREEN}[INFO]${RESET} Command succeeded: ${BOLD}$*${RESET}"
    fi
    echo -e "${YELLOW}==================================================${RESET}"
    echo
}

main() {
    echo -e "${GREEN}${BOLD}========== Starting Environment Installation ==========${RESET}"
    echo -e "${BLUE}[INFO] Timestamp: $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
    echo

    # 1. Download the NVIDIA CUDA keyring .deb
    log_and_run wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.0-1_all.deb

    # 2. Install the CUDA keyring
    log_and_run sudo dpkg -i cuda-keyring_1.0-1_all.deb

    # 3. Update APT cache
    log_and_run sudo apt-get update

    # 4. Install CUDA Runtime 11.8
    log_and_run sudo apt-get -y install cuda-runtime-11-8

    # 5. Install build-essential and other packages
    log_and_run sudo apt update
    log_and_run sudo apt install -y build-essential manpages-dev software-properties-common

    # 6. Add the ubuntu-toolchain-r/test PPA
    log_and_run sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test

    # 7. Update APT cache again and install GCC 11 and G++ 11
    log_and_run sudo apt update
    log_and_run sudo apt install -y gcc-11 g++-11

    # 8. Remove any existing alternatives configuration for 'cpp' to avoid conflicts
    log_and_run sudo update-alternatives --remove-all cpp || true

    # 9. Set up alternatives for gcc/g++ to switch between multiple versions
    #    - Register GCC 9 at priority 90, and GCC 11 at priority 110.
    log_and_run sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
       --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
       --slave /usr/bin/gcov gcov /usr/bin/gcov-9 \
       --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 \
       --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9 \
       --slave /usr/bin/cpp cpp /usr/bin/cpp-9

    log_and_run sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
       --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
       --slave /usr/bin/gcov gcov /usr/bin/gcov-11 \
       --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 \
       --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11 \
       --slave /usr/bin/cpp cpp /usr/bin/cpp-11

    # 10. Install the ONNX Runtime GPU wheel (for aarch64)
    log_and_run python3 -m pip install onnxruntime_gpu-1.18.0-cp38-cp38-linux_aarch64.whl

    # 11. Install additional Python dependencies
    log_and_run python3 -m pip install -r requirements.txt

    echo -e "${GREEN}${BOLD}========== Environment Installation Complete ==========${RESET}"
}

# Run the main function
main
