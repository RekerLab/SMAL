Processed figure inputs for the SMAL manuscript.

`figure_svg_payloads.csv` contains one row per generated manuscript figure
(`figure2` through `figure6` and `figureS1` through `figureS14`). Each row stores
the source SVG as a gzip-compressed, base64-encoded payload plus the source SVG
SHA-256 checksum.

The notebooks in `../figures/` read this CSV and regenerate one SVG file each.
