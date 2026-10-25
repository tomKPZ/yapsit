{
  description = "Yet Another Pokémon Sprite In your Terminal";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = self.packages.${system}.yapsit;
          yapsit = pkgs.clangStdenv.mkDerivation {
            pname = "yapsit";
            version = "0-unstable";
            src = self;

            nativeBuildInputs = [
              pkgs.cmake
              (pkgs.pypy3.withPackages (ps: [
                # Avoid broken optional PyPy test dependencies in Pillow/pybind11.
                ((ps.pillow.override {
                  pybind11 = (ps.pybind11.override {
                    buildTests = false;
                  }).overridePythonAttrs (_: { doCheck = false; });
                }).overridePythonAttrs (_: {
                  doCheck = false;
                  doInstallCheck = false;
                }))
                ps.cffi
              ]))
            ];

            # The generator writes beside its source and loads liblz77.so there.
            dontUseCmakeBuildDir = true;

            # Keep sprite generation fast; this only reduces compression ratio.
            FAST_COMPRESS = "1";

            doInstallCheck = true;
            installCheckPhase = ''
              runHook preInstallCheck
              "$out/bin/yapsit" --help
              "$out/bin/yapsit" -i 1 -n 1 > /dev/null
              runHook postInstallCheck
            '';

            meta = {
              description = "Display Pokémon sprites in your terminal";
              license = pkgs.lib.licenses.gpl3Only;
              platforms = systems;
              mainProgram = "yapsit";
            };
          };
        });
    };
}
