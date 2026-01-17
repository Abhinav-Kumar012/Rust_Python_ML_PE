# Build Stage
FROM rust:1.80-slim-bookworm as builder

WORKDIR /app

# Create a blank project to cache dependencies
RUN USER=root cargo new --bin rust_ml

WORKDIR /app/rust_ml

# Copy workspace Cargo.toml to help with workspace resolution (optional, but good practice)
# But here we are building a specific crate inside a workspace potentially.
# Let's assume we copy the whole workspace context or at least the relevant parts.
# Since we are inside `rust_ml` workspace, we should copy the root Cargo.toml and the crates.

# Actually, to keep it simple and robust:
# We'll copy the entire rust_ml directory context into the build container.
# This assumes the build context is the `Rest_Python_ML_PE` root or `rust_ml` root.
# Given the user wants to `containerize this pipeline`, let's assume the build context is the repo root.

COPY ./rust_ml/Cargo.toml ./Cargo.toml
COPY ./rust_ml/Cargo.lock ./Cargo.lock
COPY ./rust_ml/mnist_infer ./mnist_infer
COPY ./rust_ml/mnist_ml ./mnist_ml

# Build the release binary for mnist_infer
RUN cargo build --release --bin mnist_infer

# Runtime Stage
FROM debian:bookworm-slim

# Install OpenSSL if needed (Axum/Tokio might need it) and ca-certificates
RUN apt-get update && apt-get install -y libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /app/rust_ml/target/release/mnist_infer /app/mnist_infer

# Copy Model
# Assuming build context is repo root, we need to copy model/mnist_rust/model.mpk
# We'll expect the model to be copied into the container at runtime or build time.
# Let's copy it at build time as requested "containerized pipeline".
COPY model/mnist_rust/model.mpk /app/model/mnist_rust/model.mpk

# Environment variables
ENV RUST_LOG=info
ENV MODEL_PATH=model/mnist_rust/model.mpk

EXPOSE 8080

CMD ["./mnist_infer"]
