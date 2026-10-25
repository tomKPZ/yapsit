# yapsit
Yet Another Pokémon Sprite In your Terminal

Build with Nix:

```sh
nix build
./result/bin/yapsit --help
```

Or run directly with `nix run -- -i 1`.

The flake supports x86_64 and ARM Linux. It uses a 512-pixel compression search
window and two passes to balance build time and executable size. Compression is
lossless.

When running `gen_header.py` directly, `COMPRESS_WINDOW` sets the search window
and `COMPRESS_PASSES` limits the number of passes (`0` means run until convergence).
Leaving both unset preserves the full compression search. `FAST_COMPRESS=1` still
selects a 100-pixel window and one pass.
