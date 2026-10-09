# Figure replacement and layout check

- Original source edited in place: E:/main.tex.
- Final PDF: E:/main.pdf, 18 pages; main figures 1--10, tables 1--8, equations 1--31.
- Figure 1 retains the Overthrust source-model sections. Figure 2 uses the user-supplied workflow, single column, with display-only cropping of white margins; the original PDF remains intact.
- Figures 3--10 now show source recovery, event errors, full-record waveform residuals, the blind transfer audit, noise/receiver removal, velocity bias, independent geology and the separate carrier-ensemble audit.
- Archived figure PDF files remain intact; supplementary material is unchanged. Old figure 10 is not the new main-text Figure 10.
- Waveform visualization is post hoc, does not alter training/search or frozen confirmation statistics. It reports substantial residuals transparently.
- Plot inputs and SHA-256 values: figure_provenance.json, new_figure_provenance.json, final_diagnostic_provenance.json, figure_replacement_manifest.json.
- Local XeLaTeX -> BibTeX -> XeLaTeX builds pass. No undefined citations/references, placeholder marks or overfull boxes remain. Figure captions, panel labels and page rendering were inspected.
- Built-in isolated compiler cannot read this multi-file project's figures/fig1.pdf. This limitation does not affect the locally compiled PDF.
- This check supersedes the 10-page layout record in submission_check_20261008/final_check.md.
