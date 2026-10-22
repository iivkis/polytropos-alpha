{
  description = "polytropos-alpha: Rust, LLVM NVPTX, CUDA, Lean 4 and Task development environment";

  # Keep the project's existing nixpkgs / flake.lock pin.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      # VeriNeural currently targets NVIDIA GPU development on Linux x86_64.
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          git
          rustc
          cargo
          rustfmt
          clippy

          go-task
          coreutils # sha256sum for evidence manifests
          binutils

          llvmPackages.llvm
          llvmPackages.clang
          cmake
          ninja
          pkg-config

          lean4

          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
          cudaPackages.nsight_systems
          cudaPackages.nsight_compute
        ];

        shellHook = ''
          # nvcc and native CUDA runtime come from pinned nixpkgs;
          # libcuda.so.1 must come from the host NVIDIA driver.
          export CUDA_LIB_DIR="${pkgs.cudaPackages.cuda_cudart}/lib"
          export NVCC="${pkgs.cudaPackages.cuda_nvcc}/bin/nvcc"
          # GeForce RTX 5060 (Blackwell, compute capability 12.0).
          # Note: VeriNeural's current GpuEngine default arch is set in Rust.
          export CUDA_ARCH=120

          cuda_driver_dir=""
          if test -f /usr/lib/wsl/lib/libcuda.so.1; then
            cuda_driver_dir="/usr/lib/wsl/lib"
          elif test -f /run/opengl-driver/lib/libcuda.so.1; then
            cuda_driver_dir="/run/opengl-driver/lib"
          fi

          if test -n "$cuda_driver_dir"; then
            export CUDA_DRIVER_LIB_DIR="$cuda_driver_dir"
            # Host driver must win over CUDA toolkit stubs / other libcuda copies.
            if test -n "$LD_LIBRARY_PATH"; then
              export LD_LIBRARY_PATH="$cuda_driver_dir:$CUDA_LIB_DIR:$LD_LIBRARY_PATH"
            else
              export LD_LIBRARY_PATH="$cuda_driver_dir:$CUDA_LIB_DIR"
            fi
            if test -x "$cuda_driver_dir/nvidia-smi"; then
              export PATH="$cuda_driver_dir:$PATH"
            elif test -x /run/opengl-driver/bin/nvidia-smi; then
              export PATH="/run/opengl-driver/bin:$PATH"
            fi
            echo "CUDA driver libraries: $cuda_driver_dir"
          else
            unset CUDA_DRIVER_LIB_DIR
            if test -n "$LD_LIBRARY_PATH"; then
              export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$LD_LIBRARY_PATH"
            else
              export LD_LIBRARY_PATH="$CUDA_LIB_DIR"
            fi
            echo "CUDA driver libraries not found in WSL/NixOS host paths; GPU tasks may fail"
          fi
          unset cuda_driver_dir

          echo "VeriNeural devShell: Rust / LLVM / Lean 4 / CUDA / Task"
          echo "GPU execution requires a compatible host NVIDIA driver."
          echo "nsys=$(command -v nsys || echo unavailable)"
          echo "ncu=$(command -v ncu || echo unavailable)"
          echo "Use 'task --list' to see local commands."
        '';
      };
    };
}
