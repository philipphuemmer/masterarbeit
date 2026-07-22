#!/bin/bash

echo "=== THESIS QUALITY CHECKS ==="

THESIS="thesis/main.tex"

# 1. Check for unmatched braces
echo -e "\n[1] Brace Balance:"
OPEN=$(grep -o '{' "$THESIS" | wc -l)
CLOSE=$(grep -o '}' "$THESIS" | wc -l)
echo "  Opening: $OPEN, Closing: $CLOSE"
if [ "$OPEN" != "$CLOSE" ]; then echo "  ⚠️ MISMATCH"; fi

# 2. Duplicate labels
echo -e "\n[2] Duplicate Labels:"
DUP=$(grep '\\label{' "$THESIS" | sed 's/.*\\label{\(.*\)}.*/\1/' | sort | uniq -d)
if [ -z "$DUP" ]; then echo "  ✓ None"; else echo "$DUP" | head -5; fi

# 3. References without labels
echo -e "\n[3] Orphaned References:"
grep -o '\\ref{[^}]*}' "$THESIS" | sed 's/.*{\(.*\)}/\1/' | sort -u > /tmp/refs_used
grep -o '\\label{[^}]*}' "$THESIS" | sed 's/.*{\(.*\)}/\1/' | sort -u > /tmp/labels_def
MISSING=$(comm -23 /tmp/refs_used /tmp/labels_def)
if [ -z "$MISSING" ]; then echo "  ✓ All references have labels"; else echo "$MISSING" | head -5; fi

# 4. Cite without bib
echo -e "\n[4] Citations:"
CITES=$(grep -o '\\cite{[^}]*}' "$THESIS" | wc -l)
NATBIBS=$(grep -o '\\citet{[^}]*}\|\\citep{[^}]*}' "$THESIS" | wc -l)
echo "  Total citations (natbib): $((CITES + NATBIBS))"

# 5. Formatting issues
echo -e "\n[5] Potential Formatting Issues:"
echo "  Lines with three+ spaces: $(grep -E '   +' "$THESIS" | wc -l)"
echo "  Lines with missing tilde in references: $(grep -E 'Chapter *[0-9]|Section *[0-9]|Figure *[0-9]' "$THESIS" | head -3 | wc -l) (should use ~)"
echo "  Lines with incorrect quote usage: $(grep -E '^[^%]*"[^"]*"' "$THESIS" | wc -l) (might need `` or '')"

# 6. Section consistency
echo -e "\n[6] Section Structure:"
grep '\\section{\|\\subsection{\|\\subsubsection{' "$THESIS" | wc -l
echo "  Sections: $(grep '\\section{' "$THESIS" | wc -l)"
echo "  Subsections: $(grep '\\subsection{' "$THESIS" | wc -l)"
echo "  Subsubsections: $(grep '\\subsubsection{' "$THESIS" | wc -l)"

# 7. Table/Figure references
echo -e "\n[7] Floats (Tables/Figures):"
echo "  Figures defined: $(grep -c '\\\\begin{figure}' "$THESIS")"
echo "  Tables defined: $(grep -c '\\\\begin{table}' "$THESIS")"
echo "  Figure refs: $(grep -o '\\ref{fig:[^}]*}' "$THESIS" | wc -l)"
echo "  Table refs: $(grep -o '\\ref{tab:[^}]*}' "$THESIS" | wc -l)"

# 8. Acronyms
echo -e "\n[8] Acronyms:"
echo "  Defined: $(grep '\\acrodef{' "$THESIS" | wc -l)"
echo "  Used (first): $(grep -c '\\ac{' "$THESIS")"
echo "  First plural use: $(grep -c '\\acp{' "$THESIS")"
