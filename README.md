# yapsit
Yet Another Pokémon Sprite In your Terminal

Build with Nix:

```sh
nix build
./result/bin/yapsit --help
```

Or run directly with `nix run -- -i 1`.

The flake supports x86_64 and ARM Linux. It uses the generator's fast lossless
compression mode to reduce build time at the cost of a larger executable.
