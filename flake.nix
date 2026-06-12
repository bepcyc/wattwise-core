{
  description = "Agent-ready project shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          git
          gh
          jq
          curl
          coreutils
          gnused
          gnugrep
          findutils
          ripgrep
          # CI scanner toolchain (CI-R0): local CI mirror needs these to avoid
          # on-the-fly downloads that are fragile under rate limits.
          trivy
          gitleaks
          syft
          docker
        ];

        shellHook = ''
          mkdir -p .agent/bin

          cat > .agent/bin/git-askpass <<'EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' "x-access-token" ;;
  *Password*) printf '%s\n' "${GH_TOKEN:?GH_TOKEN is not set}" ;;
  *) printf '%s\n' "" ;;
esac
EOF

          chmod 700 .agent/bin/git-askpass

          export GIT_ASKPASS="$PWD/.agent/bin/git-askpass"
          export GIT_TERMINAL_PROMPT=0
        '';
      };
    };
}
