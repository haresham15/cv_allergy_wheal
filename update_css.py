import sys

with open('frontend/cv_allergy_wheal/app/globals.css', 'a', encoding='utf-8') as f:
    f.write("""
/* ─── Hover Tooltips ────────────────────────────────────────────── */
.cyst-hover-zone {
  position: absolute;
  border-radius: 50%;
  cursor: crosshair;
  transform: translate(-50%, -50%);
  z-index: 10;
}
.cyst-tooltip {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  margin-top: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border-accent);
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-card);
  color: var(--text-primary);
  font-size: 0.8rem;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity var(--transition);
  z-index: 20;
  text-align: left;
}
.cyst-hover-zone:hover .cyst-tooltip {
  opacity: 1;
}
""")

print("globals.css updated")
