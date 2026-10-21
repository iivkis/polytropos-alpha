{
  description = "polytropos-alpha: Rust, LLVM NVPTX, CUDA, Lean 4 and Task development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      # NVIDIA GPU development is targeted at Linux x86_64 in the initial phase.
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # CUDA packages require an unfree license.
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          rustc
          cargo
          rustfmt
          clippy

          go-task

          llvmPackages.llvm
          llvmPackages.clang
          cmake
          ninja
          pkg-config

          lean4

          cudaPackages.cuda_nvcc # Includes PTX tooling such as ptxas.
          cudaPackages.cuda_cudart
        ];

        shellHook = ''
          echo "VeriGPU devShell: Rust / LLVM / Lean 4 / CUDA / Task"
          echo "GPU execution requires a compatible NVIDIA driver on the host."
          echo "Use 'task --list' to see local commands."
        '';
      };
    };
}
