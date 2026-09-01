#!/usr/bin/env python3
"""Inspect a ViSAGE FITS catalogue and prepare its data for plotting.

ViSAGE exports galaxies as a single ``GALAXIES`` binary table: one row per
galaxy, one column per SAGE field, with units attached as ``TUNITn`` and the
run's provenance (par file, cosmology, snapshot, selection) folded into the
table header as 8-character keywords.

Values are stored exactly as SAGE writes them — masses in 10^10 Msun/h,
positions in Mpc/h, SFR in Msun/yr. This script reports what is in the file,
then converts the handful of columns you usually plot into physical units.

    python scripts/read_catalogue.py sage_outputs/catalogues/millennium_test.fits
    python scripts/read_catalogue.py <catalogue.fits> --plot smf.png

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

# Header keywords written by the FITS format itself, not by ViSAGE.
_STRUCTURAL = {
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
    "EXTNAME",
    "SIMPLE",
    "EXTEND",
    "COMMENT",
    "HISTORY",
}
_COLUMN_PREFIXES = (
    "TTYPE",
    "TFORM",
    "TUNIT",
    "TDIM",
    "TSCAL",
    "TZERO",
    "TNULL",
)


def _is_metadata(key: str) -> bool:
    """True for header cards ViSAGE wrote, false for FITS bookkeeping."""
    if not key or key in _STRUCTURAL:
        return False
    return not key.startswith(_COLUMN_PREFIXES)


# ── Reporting ─────────────────────────────────────────────────────────────────


def report_structure(hdul: fits.HDUList) -> None:
    print("\n=== File structure ===")
    hdul.info()


def report_metadata(hdu: fits.BinTableHDU) -> None:
    """Print the ViSAGE provenance keywords carried in the table header."""
    print("\n=== Metadata ===")
    cards = [(k, hdu.header[k]) for k in hdu.header if _is_metadata(k)]
    if not cards:
        print("  (none — this table carries no ViSAGE header keywords)")
        return
    width = max(len(k) for k, _ in cards)
    for key, value in cards:
        print(f"  {key:<{width}}  {value}")


def report_columns(hdu: fits.BinTableHDU) -> None:
    """Print every column with its unit, dtype and value range."""
    data = hdu.data
    print(f"\n=== Columns ({len(hdu.columns)} in {len(data)} rows) ===")
    name_w = max(len(c.name) for c in hdu.columns)
    unit_w = max(len(c.unit or "-") for c in hdu.columns)

    header = (
        f"  {'#':>3}  {'name':<{name_w}}  {'unit':<{unit_w}}  "
        f"{'dtype':<8}  {'min':>12}  {'max':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, col in enumerate(hdu.columns):
        arr = np.asarray(data[col.name])
        finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
        lo = f"{finite.min():.4g}" if finite.size else "-"
        hi = f"{finite.max():.4g}" if finite.size else "-"
        print(
            f"  {i:>3}  {col.name:<{name_w}}  {col.unit or '-':<{unit_w}}  "
            f"{arr.dtype.str:<8}  {lo:>12}  {hi:>12}"
        )


# ── Derived quantities ────────────────────────────────────────────────────────


def prepare_plot_data(hdu: fits.BinTableHDU) -> dict[str, np.ndarray]:
    """Convert the common plotting columns out of SAGE's h-scaled units.

    Returns arrays for star-forming galaxies with a resolved stellar mass;
    everything is already in log10 where a log axis is the sane choice.
    """
    data = hdu.data
    h = float(hdu.header.get("HUBBLE_H", 0.73))

    # 10^10 Msun/h -> Msun
    stellar = np.asarray(data["StellarMass"], dtype=np.float64) * 1e10 / h
    mvir = np.asarray(data["Mvir"], dtype=np.float64) * 1e10 / h
    cold = np.asarray(data["ColdGas"], dtype=np.float64) * 1e10 / h
    sfr = np.asarray(data["SfrDisk"], dtype=np.float64) + np.asarray(
        data["SfrBulge"], dtype=np.float64
    )

    keep = (stellar > 0) & np.isfinite(stellar)
    stellar, mvir, cold, sfr = stellar[keep], mvir[keep], cold[keep], sfr[keep]

    with np.errstate(divide="ignore", invalid="ignore"):
        prepared = {
            "log_stellar_mass": np.log10(stellar),
            "log_halo_mass": np.log10(np.where(mvir > 0, mvir, np.nan)),
            "log_cold_gas": np.log10(np.where(cold > 0, cold, np.nan)),
            "sfr": sfr,
            "log_ssfr": np.log10(np.where(sfr > 0, sfr / stellar, np.nan)),
            # Type 0 = central, 1 = satellite
            "is_central": np.asarray(data["Type"])[keep] == 0,
            # Mpc/h -> Mpc
            "x": np.asarray(data["Posx"], dtype=np.float64)[keep] / h,
            "y": np.asarray(data["Posy"], dtype=np.float64)[keep] / h,
            "z": np.asarray(data["Posz"], dtype=np.float64)[keep] / h,
        }

    print("\n=== Prepared for plotting ===")
    print(f"  hubble_h              {h}")
    print(f"  galaxies with M* > 0  {keep.sum()} of {len(keep)}")
    for key, arr in prepared.items():
        if arr.dtype == bool:
            print(f"  {key:<20}  {arr.sum()} True")
        else:
            good = arr[np.isfinite(arr)]
            rng = f"{good.min():.3f} .. {good.max():.3f}" if good.size else "-"
            print(f"  {key:<20}  {rng}  ({good.size} finite)")
    return prepared


# ── Optional quick-look figure ────────────────────────────────────────────────


def plot(
    prepared: dict[str, np.ndarray], header: fits.Header, out: Path
) -> None:
    """Two standard quick-look panels: the mass function and the main sequence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    box = float(header.get("BOX_SIZE", 0) or 0)
    h = float(header.get("HUBBLE_H", 0.73))
    label = str(header.get("SNAP_LAB", "")).strip()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )

    # Stellar mass function — a volume-normalised histogram, so it only makes
    # sense if we know the box we selected from.
    logm = prepared["log_stellar_mass"]
    bins = np.arange(np.floor(logm.min()), np.ceil(logm.max()) + 0.2, 0.2)
    counts, edges = np.histogram(logm, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if box > 0:
        volume = (box / h) ** 3  # Mpc^3
        phi = counts / volume / 0.2
        ax1.set_ylabel(r"$\phi$  [Mpc$^{-3}$ dex$^{-1}$]")
    else:
        phi = counts / 0.2
        ax1.set_ylabel("N / dex")
    with np.errstate(divide="ignore"):
        ax1.plot(centres[counts > 0], phi[counts > 0], lw=2, color="#2f6f9f")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\log_{10}(M_\star / \mathrm{M}_\odot)$")
    ax1.set_title("Stellar mass function")

    # Star-forming main sequence — one dense cloud, so bin it rather than
    # overplotting ~10^4 markers.
    ok = np.isfinite(prepared["log_ssfr"])
    hb = ax2.hexbin(
        logm[ok],
        prepared["log_ssfr"][ok] + logm[ok],  # log SFR
        gridsize=50,
        bins="log",
        cmap="Blues",
        mincnt=1,
    )
    fig.colorbar(hb, ax=ax2, label="log N")
    ax2.set_xlabel(r"$\log_{10}(M_\star / \mathrm{M}_\odot)$")
    ax2.set_ylabel(
        r"$\log_{10}(\mathrm{SFR}\,/\,\mathrm{M}_\odot\,\mathrm{yr}^{-1})$"
    )
    ax2.set_title("Star-forming main sequence")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.2, lw=0.6)
        ax.set_axisbelow(True)

    if label:
        fig.suptitle(label, fontsize=10)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("catalogue", type=Path, help="ViSAGE FITS catalogue")
    ap.add_argument(
        "--hdu",
        default="GALAXIES",
        help="table extension to read (default: GALAXIES)",
    )
    ap.add_argument(
        "--plot",
        type=Path,
        metavar="PNG",
        help="also write a two-panel quick-look figure",
    )
    args = ap.parse_args()

    with fits.open(args.catalogue) as hdul:
        report_structure(hdul)

        hdu = hdul[args.hdu] if args.hdu in hdul else hdul[1]
        report_metadata(hdu)
        report_columns(hdu)
        prepared = prepare_plot_data(hdu)

        if args.plot:
            plot(prepared, hdu.header, args.plot)


if __name__ == "__main__":
    main()
