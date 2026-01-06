#!/usr/bin/env python3
"""
Visualize histone modification tracks using pyGenomeTracks.

This script takes cell types and genomic regions as input, then generates
visualizations of histone marks (H3K27ac, H3K4me1, H3K27me3, H3K9me3) using
bigWig files from Data/Histone/DNAbw/.

Usage:
    python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut,IT_EP_CLA_Glut
"""

import argparse
import os
import sys
import glob
import tempfile
import subprocess
import re
import math
from pathlib import Path

try:
    import pyBigWig
    PYBIGWIG_AVAILABLE = True
except ImportError:
    PYBIGWIG_AVAILABLE = False
    print("Warning: pyBigWig not available. Install with: pip install pyBigWig")
    print("Auto-scaling will be disabled without pyBigWig.")


def load_cre_bed(bed_file="Data/CRE.bed"):
    """
    Load CRE regions from BED file.

    Args:
        bed_file: Path to BED file containing CRE regions

    Returns:
        Dictionary mapping CRE IDs to (chrom, start, end) tuples
    """
    cre_dict = {}

    if not os.path.exists(bed_file):
        print(f"Warning: CRE BED file not found: {bed_file}", file=sys.stderr)
        return cre_dict

    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue

            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            cre_id = fields[4]

            cre_dict[cre_id] = (chrom, start, end)

    return cre_dict


def parse_region(region_str, cre_dict=None):
    """
    Parse region string - either genomic coordinates or CRE ID.

    Args:
        region_str: Region string (e.g., "chr1:1000000-2000000" or "CRE001")
        cre_dict: Dictionary mapping CRE IDs to coordinates (optional)

    Returns:
        Genomic region string in format "chr:start-end"
    """
    # Check if it's a CRE ID
    if region_str.startswith('CRE') and cre_dict:
        if region_str in cre_dict:
            chrom, start, end = cre_dict[region_str]
            genomic_region = f"{chrom}:{start}-{end}"
            print(f"  {region_str} → {genomic_region}")
            return genomic_region
        else:
            print(f"Warning: CRE ID '{region_str}' not found in BED file", file=sys.stderr)
            return None

    # Otherwise, assume it's already a genomic coordinate
    return region_str


def parse_modality_values(value_str, histone_marks=None):
    """
    Parse comma-separated values for histone marks.

    Args:
        value_str: Comma-separated values (e.g., "10,20,30,40" or "10")
        histone_marks: List of histone mark names (default: ["H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"])

    Returns:
        Dictionary mapping histone marks to values, or None if value_str is None
    """
    if value_str is None:
        return None

    if histone_marks is None:
        histone_marks = ["H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]

    # Split by comma and convert to float
    values = [float(v.strip()) for v in value_str.split(',')]

    if len(values) == 1:
        # Single value - apply to all marks
        return {mark: values[0] for mark in histone_marks}
    elif len(values) == len(histone_marks):
        # One value per mark
        return {mark: val for mark, val in zip(histone_marks, values)}
    else:
        print(f"Error: Expected 1 or {len(histone_marks)} values, got {len(values)}", file=sys.stderr)
        print(f"Histone marks order: {', '.join(histone_marks)}", file=sys.stderr)
        sys.exit(1)


def expand_region(region_str, expand_bp=50000):
    """
    Expand a genomic region by specified base pairs on each side.

    Args:
        region_str: Region string in format "chr:start-end"
        expand_bp: Base pairs to expand on each side (default: 50000 for 100kb total)

    Returns:
        Tuple of (expanded_region_str, original_coords)
        - expanded_region_str: Expanded region as "chr:start-end"
        - original_coords: Original coordinates as (chrom, start, end)
    """
    try:
        chrom, coords = region_str.split(':')
        start, end = coords.split('-')
        start, end = int(start), int(end)
    except (ValueError, AttributeError):
        print(f"Warning: Invalid region format '{region_str}'", file=sys.stderr)
        return region_str, None

    # Calculate center and expand
    center = (start + end) // 2
    expanded_start = max(0, center - expand_bp)
    expanded_end = center + expand_bp

    expanded_region = f"{chrom}:{expanded_start}-{expanded_end}"
    original_coords = (chrom, start, end)

    return expanded_region, original_coords


def extract_numeric_prefix(filename):
    """
    Extract numeric prefix from filename.

    Args:
        filename: Filename string (e.g., "001_CLA_EPd_CTX_Car3_Glut.H3K27ac.bw")

    Returns:
        Integer prefix if found, otherwise 999999 (for sorting)
    """
    match = re.match(r'^(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 999999


def find_bw_files(celltype, data_dir="Data/Histone/DNAbw"):
    """
    Find bigWig files for a given cell type and histone marks.

    Args:
        celltype: Cell type name (e.g., "CLA_EPd_CTX_Car3_Glut")
        data_dir: Directory containing bigWig files

    Returns:
        Dictionary mapping histone marks to file paths
    """
    histone_marks = ["H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]
    bw_files = {}

    for mark in histone_marks:
        # Pattern: *_{celltype}.{mark}.*.bw
        pattern = f"{data_dir}/*_{celltype}.{mark}.*.bw"
        matches = glob.glob(pattern)

        if not matches:
            print(f"Warning: No file found for {celltype} - {mark}", file=sys.stderr)
            print(f"  Pattern searched: {pattern}", file=sys.stderr)
            continue

        if len(matches) > 1:
            print(f"Warning: Multiple files found for {celltype} - {mark}, using first one:", file=sys.stderr)
            for m in matches:
                print(f"  {m}", file=sys.stderr)

        bw_files[mark] = matches[0]

    return bw_files


def sort_celltypes_by_prefix(celltypes, data_dir="Data/Histone/DNAbw"):
    """
    Sort cell types by their numeric prefix in the filename.

    Args:
        celltypes: List of cell type names
        data_dir: Directory containing bigWig files

    Returns:
        List of cell types sorted by numeric prefix
    """
    celltype_with_prefix = []

    for celltype in celltypes:
        # Find any file for this celltype to get the prefix
        pattern = f"{data_dir}/*_{celltype}.*.bw"
        matches = glob.glob(pattern)

        if matches:
            # Extract prefix from first match
            filename = os.path.basename(matches[0])
            prefix = extract_numeric_prefix(filename)
            celltype_with_prefix.append((prefix, celltype))
        else:
            # If no files found, put at end
            print(f"Warning: No files found for cell type: {celltype}", file=sys.stderr)
            celltype_with_prefix.append((999999, celltype))

    # Sort by prefix
    celltype_with_prefix.sort(key=lambda x: x[0])

    return [ct for prefix, ct in celltype_with_prefix]


def compute_modality_max_values(celltypes, region, data_dir="Data/Histone/DNAbw", num_bins=700):
    """
    Compute max values for each histone mark across all cell types in the region.

    Args:
        celltypes: List of cell type names
        region: Genomic region string (e.g., "chr1:1000000-2000000")
        data_dir: Directory containing bigWig files
        num_bins: Number of bins for binning (default: 700, same as pyGenomeTracks)

    Returns:
        Dictionary mapping histone marks to max values
    """
    if not PYBIGWIG_AVAILABLE:
        print("Warning: pyBigWig not available. Skipping auto-scaling.")
        return {}

    # Parse region
    try:
        chrom, coords = region.split(':')
        start, end = coords.split('-')
        start, end = int(start), int(end)
    except (ValueError, AttributeError):
        print(f"Warning: Invalid region format '{region}', skipping auto-scaling")
        return {}

    # Ensure chromosome has 'chr' prefix
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    print(f"\nComputing max values for each histone mark in region {chrom}:{start}-{end}...")

    histone_marks = ["H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]
    modality_values = {mark: [] for mark in histone_marks}

    # Collect values for each histone mark across all cell types
    for celltype in celltypes:
        bw_files = find_bw_files(celltype, data_dir)

        for mark in histone_marks:
            if mark not in bw_files:
                continue

            bw_file = bw_files[mark]

            try:
                bw = pyBigWig.open(str(bw_file))

                # Check if chromosome exists in bigwig
                if chrom not in bw.chroms():
                    # Try without 'chr' prefix
                    chrom_alt = chrom.replace('chr', '')
                    if chrom_alt in bw.chroms():
                        chrom_query = chrom_alt
                    else:
                        print(f"  Warning: Chromosome {chrom} not found in {os.path.basename(bw_file)}")
                        bw.close()
                        continue
                else:
                    chrom_query = chrom

                # Get binned statistics
                binned_values = bw.stats(chrom_query, start, end, type="mean", nBins=num_bins)
                bw.close()

                # Filter out None/NaN values
                if binned_values:
                    binned_values = [v for v in binned_values
                                   if v is not None and not (isinstance(v, float) and math.isnan(v))]

                if binned_values and len(binned_values) > 0:
                    modality_values[mark].extend(binned_values)

            except Exception as e:
                print(f"  Warning: Could not read {os.path.basename(bw_file)}: {e}")
                continue

    # Compute max for each modality
    modality_max = {}
    for mark in histone_marks:
        if modality_values[mark] and len(modality_values[mark]) > 0:
            # Filter again to ensure no None values snuck through
            valid_values = [v for v in modality_values[mark] if v is not None]
            if valid_values:
                max_val = max(valid_values)
                modality_max[mark] = round(max_val, 3)
                print(f"  {mark}: max = {modality_max[mark]:.4f} (from {len(valid_values)} bins)")
            else:
                print(f"  {mark}: No valid data found (all values were None/NaN)")
        else:
            print(f"  {mark}: No data found")

    return modality_max


def create_tracks_config(celltypes, region, data_dir="Data/Histone/DNAbw",
                        height=2, spacer_height=0.2,
                        fontsize=10, axis_fontsize=12,
                        modality_max=None, modality_min=None,
                        user_max_values=None, user_min_values=None,
                        gtf_file=None, gtf_height=5, gtf_fontsize=10,
                        highlight_coords=None):
    """
    Create a pyGenomeTracks configuration file.

    Args:
        celltypes: List of cell type names (will be sorted by numeric prefix)
        region: Genomic region for computing max values
        data_dir: Directory containing bigWig files
        height: Track height (default: 2)
        spacer_height: Height of spacer between tracks (default: 0.2)
        fontsize: Font size for track titles (default: 10)
        axis_fontsize: Font size for x-axis (default: 12)
        modality_max: Dictionary of auto-computed max values per histone mark (optional)
        modality_min: Dictionary of auto-computed min values per histone mark (optional)
        user_max_values: Dictionary of user-specified max values per histone mark (optional)
        user_min_values: Dictionary of user-specified min values per histone mark (optional)
        gtf_file: Path to GTF file for gene annotation track (optional)
        gtf_height: Height of GTF track (default: 5)
        gtf_fontsize: Font size for GTF gene labels (default: 10)
        highlight_coords: Tuple of (chrom, start, end) for highlighting original region (optional)

    Returns:
        Path to the configuration file
    """
    config_lines = []
    histone_marks = ["H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]

    # Colors for each histone mark
    colors = {
        "H3K27ac": "#E41A1C",    # Red (active enhancer)
        "H3K4me1": "#377EB8",    # Blue (enhancer)
        "H3K27me3": "#4DAF4A",   # Green (repressive)
        "H3K9me3": "#984EA3"     # Purple (heterochromatin)
    }

    # Sort cell types by numeric prefix
    sorted_celltypes = sort_celltypes_by_prefix(celltypes, data_dir)

    print(f"\nCell types will be displayed in this order:")
    for i, ct in enumerate(sorted_celltypes, 1):
        print(f"  {i}. {ct}")

    for celltype in sorted_celltypes:
        bw_files = find_bw_files(celltype, data_dir)

        if not bw_files:
            print(f"Error: No bigWig files found for cell type: {celltype}", file=sys.stderr)
            continue

        # Add tracks for each histone mark in specified order
        for mark in histone_marks:
            if mark not in bw_files:
                continue

            config_lines.append(f"[{celltype}_{mark}]")
            config_lines.append(f"file = {bw_files[mark]}")
            config_lines.append(f"title = {celltype} - {mark}")
            config_lines.append(f"color = {colors[mark]}")
            config_lines.append(f"height = {height}")
            config_lines.append(f"fontsize = {fontsize}")

            # Min value: user-specified > modality-specific > default (0)
            if user_min_values and mark in user_min_values:
                config_lines.append(f"min_value = {user_min_values[mark]}")
            elif modality_min and mark in modality_min:
                config_lines.append(f"min_value = {modality_min[mark]}")
            else:
                config_lines.append("min_value = 0")

            # Max value: user-specified > modality-specific > auto-scale
            if user_max_values and mark in user_max_values:
                config_lines.append(f"max_value = {user_max_values[mark]}")
            elif modality_max and mark in modality_max:
                config_lines.append(f"max_value = {modality_max[mark]}")

            config_lines.append("file_type = bigwig")
            config_lines.append("number_of_bins = 700")
            config_lines.append("")

            # Add spacer between tracks
            config_lines.append("[spacer]")
            config_lines.append(f"height = {spacer_height}")
            config_lines.append("")

    # Add GTF track if provided
    if gtf_file:
        gtf_path = Path(gtf_file)
        if gtf_path.exists():
            config_lines.append("[genes]")
            config_lines.append(f"file = {gtf_path.absolute()}")
            config_lines.append("title = Genes")
            config_lines.append(f"height = {gtf_height}")
            config_lines.append("file_type = gtf")
            config_lines.append("prefered_name = gene_name")
            config_lines.append("merge_transcripts = true")
            config_lines.append("style = flybase")
            config_lines.append(f"fontsize = {gtf_fontsize}")
            config_lines.append("")

            config_lines.append("[spacer]")
            config_lines.append(f"height = {spacer_height}")
            config_lines.append("")
        else:
            print(f"Warning: GTF file not found: {gtf_file}", file=sys.stderr)

    # Add highlight region if provided
    if highlight_coords:
        chrom, start, end = highlight_coords
        # Create a temporary BED file for the highlight region
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
            f.write(f"{chrom}\t{start}\t{end}\tCRE_region\n")
            highlight_bed_file = f.name

        config_lines.append("[vhighlight]")
        config_lines.append(f"file = {highlight_bed_file}")
        config_lines.append("type = vhighlight")
        config_lines.append("color = #FFD700")
        config_lines.append("alpha = 0.3")
        config_lines.append("")

    # Add x-axis track at the end
    config_lines.append("[x-axis]")
    config_lines.append("where = bottom")
    config_lines.append(f"fontsize = {axis_fontsize}")
    config_lines.append("")

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('\n'.join(config_lines))
        config_file = f.name

    return config_file


def generate_plots(regions, region_names, celltypes, output_prefix="histone_tracks",
                   data_dir="Data/Histone/DNAbw",
                   width=40, dpi=300, height=2, spacer_height=0.2,
                   fontsize=10, axis_fontsize=12, gtf_fontsize=10,
                   user_max_values=None, user_min_values=None, no_auto_scale=False,
                   gtf_file=None, gtf_height=5):
    """
    Generate plots for all regions using pyGenomeTracks.

    Args:
        regions: List of genomic regions (e.g., ["chr1:1-1000", "chr2:2-1000"])
        region_names: List of original region names for filenames (e.g., ["CRE001", "chr1:1-1000"])
        celltypes: List of cell type names
        output_prefix: Prefix for output files
        data_dir: Directory containing bigWig files
        width: Figure width in cm (default: 40)
        dpi: Image resolution (default: 300)
        height: Track height (default: 2)
        spacer_height: Height of spacer between tracks (default: 0.2)
        fontsize: Font size for track titles (default: 10)
        axis_fontsize: Font size for x-axis (default: 12)
        gtf_fontsize: Font size for GTF gene labels (default: 10)
        user_max_values: Dictionary of user-specified max values per histone mark (optional)
        user_min_values: Dictionary of user-specified min values per histone mark (optional)
        no_auto_scale: Disable auto-scaling (default: False)
        gtf_file: Path to GTF file for gene annotation track (optional)
        gtf_height: Height of GTF track (default: 5)
    """
    # Create output directory if needed
    output_path = Path(output_prefix)
    if output_prefix.endswith('/') or output_prefix.endswith(os.sep):
        # User specified a directory
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = output_path
        base_prefix = "histone_tracks"
    else:
        # User specified a file prefix
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        base_prefix = output_path.name

    # Generate a plot for each region
    for i, (region, region_name) in enumerate(zip(regions, region_names)):
        # Expand region to 100kb and keep original coordinates for highlighting
        expanded_region, original_coords = expand_region(region, expand_bp=50000)
        print(f"\nRegion {i+1} ({region_name}):")
        print(f"  Original: {region}")
        print(f"  Expanded to 100kb: {expanded_region}")

        # Compute modality-specific max values for the EXPANDED region
        modality_max = None
        if not no_auto_scale and user_max_values is None and PYBIGWIG_AVAILABLE:
            modality_max = compute_modality_max_values(celltypes, expanded_region, data_dir)

        # Create tracks configuration with highlight
        config_file = create_tracks_config(
            celltypes, expanded_region, data_dir,
            height=height, spacer_height=spacer_height,
            fontsize=fontsize, axis_fontsize=axis_fontsize,
            modality_max=modality_max, modality_min=None,
            user_max_values=user_max_values, user_min_values=user_min_values,
            gtf_file=gtf_file, gtf_height=gtf_height,
            gtf_fontsize=gtf_fontsize,
            highlight_coords=original_coords
        )
        print(f"Created configuration file: {config_file}")

        # Output as PDF - use original region name for filename
        # Clean up region name for use in filename
        safe_region_name = region_name.replace(':', '_').replace('-', '_')
        if output_prefix.endswith('/') or output_prefix.endswith(os.sep):
            output_file = output_dir / f"{base_prefix}_region{i+1}_{safe_region_name}.pdf"
        else:
            output_file = output_dir / f"{base_prefix}_region{i+1}_{safe_region_name}.pdf"
        output_file = str(output_file)

        cmd = [
            "pyGenomeTracks",
            "--tracks", config_file,
            "--region", expanded_region,
            "--outFileName", output_file,
            "--width", str(width),
            "--dpi", str(dpi)
        ]

        print(f"\nGenerating plot for expanded region {expanded_region}...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Plot saved to: {output_file}")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error generating plot for region {expanded_region}:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            continue
        except FileNotFoundError:
            print("Error: pyGenomeTracks not found. Please install it with:", file=sys.stderr)
            print("  pip install pyGenomeTracks", file=sys.stderr)
            sys.exit(1)

        print(f"Configuration file for this region: {config_file}")
        if original_coords:
            print(f"Highlighted region: {original_coords[0]}:{original_coords[1]}-{original_coords[2]}")

    print("\nYou can reuse configuration files with: pyGenomeTracks --tracks <config> --region <region> --outFileName <output>")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize histone modification tracks using pyGenomeTracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - single region and cell type (genomic coordinates)
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut

  # Using CRE IDs instead of genomic coordinates
  python visualize_histones.py --region CRE001 --celltype CLA_EPd_CTX_Car3_Glut

  # Multiple CRE IDs
  python visualize_histones.py --region CRE001,CRE002,CRE003 --celltype IT_EP_CLA_Glut

  # Mix of CRE IDs and genomic coordinates
  python visualize_histones.py --region CRE001,chr1:1000000-2000000,CRE002 --celltype CLA_EPd_CTX_Car3_Glut

  # Multiple cell types (will be automatically sorted by numeric prefix)
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut,IT_EP_CLA_Glut

  # Custom track heights and spacing
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut \\
      --height 3 --spacer-height 0.5

  # Custom font sizes
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut \\
      --fontsize 12 --axis-fontsize 14

  # Set max value for all tracks (overrides auto-scaling)
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut \\
      --max 10

  # Set different max values for each histone mark (H3K27ac, H3K4me1, H3K27me3, H3K9me3)
  python visualize_histones.py --region CRE001 --celltype CLA_EPd_CTX_Car3_Glut \\
      --max 10,20,30,40

  # Set both min and max values per mark
  python visualize_histones.py --region CRE001 --celltype CLA_EPd_CTX_Car3_Glut \\
      --min 0,0,0,0 --max 15,25,35,45

  # Disable auto-scaling (pyGenomeTracks will auto-scale each track independently)
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut \\
      --no-auto-scale

  # Include gene annotation track
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut \\
      --gtf Data/source/gencode.v48.annotation.gtf.gz --gtf-height 7

  # Custom figure size and resolution
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut \\
      --width 50 --dpi 600

  # Custom output prefix
  python visualize_histones.py --region chr1:1000000-2000000 --celltype CLA_EPd_CTX_Car3_Glut \\
      --output my_histone_tracks

Note:
  - All regions are automatically expanded to 100kb for better context visualization
  - The original ~1kb region is highlighted with a gold semi-transparent overlay
  - By default, the script auto-computes max values for each histone mark separately,
    so all H3K27ac tracks share the same scale, all H3K4me1 tracks share the same scale, etc.
  - Min is always set to 0
        """
    )

    # Required arguments
    parser.add_argument(
        "--region",
        required=True,
        help="Comma-separated genomic regions or CRE IDs. "
             "Genomic format: chr1:1000000-2000000. "
             "CRE format: CRE001. "
             "You can mix both formats: CRE001,chr1:1000000-2000000,CRE002"
    )

    parser.add_argument(
        "--celltype",
        required=True,
        help="Comma-separated cell type names (e.g., CLA_EPd_CTX_Car3_Glut,IT_EP_CLA_Glut). "
             "Will be automatically sorted by numeric prefix in filenames."
    )

    # Optional arguments - file paths
    parser.add_argument(
        "--output",
        default="histone_tracks",
        help="Output file prefix (default: histone_tracks)"
    )

    parser.add_argument(
        "--data-dir",
        default="Data/Histone/DNAbw",
        help="Directory containing bigWig files (default: Data/Histone/DNAbw)"
    )

    parser.add_argument(
        "--cre-bed",
        default="Data/CRE.bed",
        help="Path to CRE BED file for looking up CRE IDs (default: Data/CRE.bed)"
    )

    # Track appearance
    parser.add_argument(
        "--height",
        type=float,
        default=2,
        help="BigWig track height (default: 2)"
    )

    parser.add_argument(
        "--spacer-height",
        type=float,
        default=0.2,
        help="Height of spacer between tracks (default: 0.2, smaller = tighter)"
    )

    # Font sizes
    parser.add_argument(
        "--fontsize",
        type=int,
        default=10,
        help="Font size for track titles (default: 10)"
    )

    parser.add_argument(
        "--axis-fontsize",
        type=int,
        default=12,
        help="Font size for x-axis labels (default: 12)"
    )

    parser.add_argument(
        "--gtf-fontsize",
        type=int,
        default=10,
        help="Font size for GTF gene labels (default: 10)"
    )

    # Scale settings
    parser.add_argument(
        "--max",
        type=str,
        default=None,
        help="Maximum values for histone marks. "
             "Single value (applies to all): --max 10. "
             "Per-mark values (H3K27ac,H3K4me1,H3K27me3,H3K9me3): --max 10,20,30,40"
    )

    parser.add_argument(
        "--min",
        type=str,
        default=None,
        help="Minimum values for histone marks. "
             "Single value (applies to all): --min 0. "
             "Per-mark values (H3K27ac,H3K4me1,H3K27me3,H3K9me3): --min 0,0,0,0"
    )

    parser.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable auto-scaling (pyGenomeTracks will auto-scale each track independently)"
    )

    # GTF annotation track
    parser.add_argument(
        "--gtf",
        type=str,
        default=None,
        help="Path to GTF file for gene annotation track (optional)"
    )

    parser.add_argument(
        "--gtf-height",
        type=float,
        default=5,
        help="Height of GTF annotation track (default: 5)"
    )

    # Figure settings
    parser.add_argument(
        "--width",
        type=float,
        default=40,
        help="Figure width in cm (default: 40)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution (default: 300)"
    )

    args = parser.parse_args()

    # Parse min/max values
    user_max_values = parse_modality_values(args.max)
    user_min_values = parse_modality_values(args.min)

    # Load CRE BED file for CRE ID lookup
    cre_dict = load_cre_bed(args.cre_bed)
    if cre_dict:
        print(f"Loaded {len(cre_dict)} CRE regions from {args.cre_bed}")

    # Parse comma-separated inputs
    raw_regions = [r.strip() for r in args.region.split(',')]
    celltypes = [c.strip() for c in args.celltype.split(',')]

    # Convert CRE IDs to genomic coordinates (keep both original names and converted coordinates)
    print("\nParsing regions:")
    regions = []  # Genomic coordinates for plotting
    region_names = []  # Original names for filenames
    for region_str in raw_regions:
        genomic_region = parse_region(region_str, cre_dict)
        if genomic_region:
            regions.append(genomic_region)
            region_names.append(region_str)  # Keep original name (CRE001 or chr:start-end)
        else:
            print(f"Error: Could not parse region '{region_str}'", file=sys.stderr)
            sys.exit(1)

    print("\n" + "=" * 80)
    print("Histone Modification Track Visualization")
    print("=" * 80)
    print(f"Regions: {regions}")
    print(f"Cell types (will be sorted by numeric prefix): {celltypes}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output prefix: {args.output}")
    print(f"Track height: {args.height}")
    print(f"Spacer height: {args.spacer_height}")
    print(f"Font size: {args.fontsize}")
    print(f"Axis font size: {args.axis_fontsize}")

    # Print min/max values
    if user_max_values:
        print("Max values (user-specified):")
        for mark in ["H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]:
            if mark in user_max_values:
                print(f"  {mark}: {user_max_values[mark]}")
    elif args.no_auto_scale:
        print("Auto-scaling: DISABLED (each track scales independently)")
    else:
        print("Auto-scaling: ENABLED (per histone mark)")

    if user_min_values:
        print("Min values (user-specified):")
        for mark in ["H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]:
            if mark in user_min_values:
                print(f"  {mark}: {user_min_values[mark]}")

    if args.gtf:
        print(f"GTF file: {args.gtf}")
        print(f"GTF height: {args.gtf_height}")
    print(f"Figure width: {args.width} cm")
    print(f"DPI: {args.dpi}")
    print(f"Output format: PDF (AI-editable)")
    print("=" * 80)

    # Validate data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate plots
    generate_plots(
        regions, region_names, celltypes, args.output, args.data_dir,
        width=args.width, dpi=args.dpi,
        height=args.height, spacer_height=args.spacer_height,
        fontsize=args.fontsize, axis_fontsize=args.axis_fontsize,
        gtf_fontsize=args.gtf_fontsize,
        user_max_values=user_max_values, user_min_values=user_min_values,
        no_auto_scale=args.no_auto_scale,
        gtf_file=args.gtf, gtf_height=args.gtf_height
    )


if __name__ == "__main__":
    main()
