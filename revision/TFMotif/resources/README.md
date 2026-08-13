# HOCOMOCO resources

The analysis uses the current HOCOMOCO v14 CORE collection and its
mouse-specific annotation:

- `H14CORE_meme_format.meme`
- `H14CORE-MOUSE_annotation.jsonl`

They were downloaded from the official HOCOMOCO v14 release:

```text
https://hocomoco14.autosome.org/final_bundle/hocomoco14/H14CORE/
  formatted_motifs/H14CORE_meme_format.meme
https://hocomoco14.autosome.org/final_bundle/hocomoco14/H14CORE/
  H14CORE-MOUSE_annotation.jsonl
```

The annotation maps each motif to its mouse gene symbol, synonyms, MGI,
Entrez, and UniProt identifiers as well as TFClass metadata. HOCOMOCO states
that the motif collection is distributed under WTFPL and may alternatively be
treated as CC-BY. See:

```text
https://hocomoco14.autosome.org/downloads_v14
```

`results/run_manifest.json` records SHA-256 hashes of both resource files used
in a run.
