# Multi-stage Dockerfile for cocotb testing with SystemVerilog
# Uses oss-cad-suite which includes Verilator and other EDA tools

FROM ubuntu:22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
  curl \
  wget \
  git \
  make \
  build-essential \
  python3 \
  python3-pip \
  python3-venv \
  libffi-dev \
  && rm -rf /var/lib/apt/lists/*

# Stage for oss-cad-suite installation
FROM base AS oss-cad-suite

# Install oss-cad-suite (includes Verilator, Yosys, cocotb, and other tools)
# This layer is cached unless the version changes
# Note: The tarball filename uses format YYYYMMDD (no dashes) while the release tag uses YYYY-MM-DD
ARG OSS_CAD_SUITE_VERSION=2024-12-13
ARG OSS_CAD_SUITE_DATE=20241213
RUN wget -q https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE_VERSION}/oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz \
  && tar -xzf oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz -C /opt \
  && rm oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz

# Add oss-cad-suite to PATH
ENV PATH="/opt/oss-cad-suite/bin:${PATH}"

# Final stage
FROM oss-cad-suite AS cocotb

# Verify Verilator installation (cocotb will be verified when tests run)
RUN echo "=== Verifying installations ===" && \
  verilator --version && \
  python3 --version

# Create a non-root user matching the host user's UID/GID
# This ensures files created in mounted volumes have correct ownership on the host
ARG USER_ID=1000
ARG GROUP_ID=1000
# Use --non-unique flag to allow GID conflicts (e.g., macOS GID 20 = staff)
# and || true to continue if group already exists
RUN groupadd --non-unique -g ${GROUP_ID} cocotb_user || true && \
  useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash cocotb_user && \
  chown -R cocotb_user:cocotb_user /opt/oss-cad-suite

# Set working directory and ensure cocotb_user owns it
WORKDIR /workspace
RUN chown cocotb_user:cocotb_user /workspace

# Switch to non-root user
USER cocotb_user

# Set environment variables for cocotb
ENV COCOTB_REDUCED_LOG_FMT=1
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
