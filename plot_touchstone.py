#!/usr/bin/env python3
"""
Touchstone File Viewer with scikit-rf

Features
- Prompts for one or more Touchstone files (.s2p, .sNp)
- Lets you choose which S-parameters to plot (S11, S12, S21, S22)
- Plots magnitude in dB vs linear frequency covering the overall range

Requirements
    pip install scikit-rf matplotlib
(If tkinter is missing on your OS, install python-tk / python3-tk via your package manager.)
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import skrf as rf
except Exception as e:
    raise SystemExit(
        "Error: scikit-rf is required. Install with `pip install scikit-rf`.\n"
        f"Underlying error: {e}"
    )


# ------------------------------
# UI helpers
# ------------------------------
class ParamSelector(tk.Toplevel):
    """Simple checkbox dialog to pick S-parameters for a 2-port plot."""

    def __init__(self, master: tk.Tk | tk.Toplevel):
        super().__init__(master)
        self.title("Select S-parameters to plot")
        self.resizable(False, False)
        self.grab_set()

        self.vars = {p: tk.IntVar(value=1 if p == "S11" else 0) for p in ("S11", "S12", "S21", "S22")}

        tk.Label(self, text="Choose one or more S-parameters:").grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6), sticky="w")

        for r, p in enumerate(("S11", "S12", "S21", "S22"), start=1):
            tk.Checkbutton(self, text=p, variable=self.vars[p]).grid(row=r, column=0, sticky="w", padx=16)

        btns = tk.Frame(self)
        btns.grid(row=5, column=0, columnspan=2, pady=12)
        tk.Button(btns, text="Plot", width=10, command=self._ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Cancel", width=10, command=self._cancel).pack(side=tk.LEFT, padx=6)

        self.selected: List[str] | None = None

        # Center dialog over parent
        self.update_idletasks()
        if master.winfo_viewable():
            x = master.winfo_rootx() + (master.winfo_width() // 2) - (self.winfo_width() // 2)
            y = master.winfo_rooty() + (master.winfo_height() // 2) - (self.winfo_height() // 2)
            self.geometry(f"+{x}+{y}")

    def _ok(self):
        sel = [k for k, v in self.vars.items() if v.get() == 1]
        if not sel:
            messagebox.showwarning("No selection", "Please select at least one S-parameter.")
            return
        self.selected = sel
        self.destroy()

    def _cancel(self):
        self.selected = None
        self.destroy()


# ------------------------------
# Core logic
# ------------------------------

def parse_param(param: str) -> Tuple[int, int]:
    """Convert 'Sij' -> (i-1, j-1) indices, validate format."""
    p = param.strip().upper()
    if len(p) != 3 or p[0] != "S" or not p[1:].isdigit():
        raise ValueError(f"Invalid S-parameter: {param}")
    i = int(p[1]) - 1
    j = int(p[2]) - 1
    if i < 0 or j < 0:
        raise ValueError(f"Invalid S-parameter: {param}")
    return i, j


def choose_freq_unit(freqs_hz: np.ndarray) -> Tuple[float, str]:
    """Pick a nice unit (Hz, kHz, MHz, GHz) for x-axis based on max frequency."""
    fmax = float(np.nanmax(freqs_hz)) if freqs_hz.size else 0.0
    if fmax >= 1e9:
        return 1e9, "GHz"
    if fmax >= 1e6:
        return 1e6, "MHz"
    if fmax >= 1e3:
        return 1e3, "kHz"
    return 1.0, "Hz"


def read_networks(paths: List[str]) -> List[rf.Network]:
    nets: List[rf.Network] = []
    errors: List[str] = []
    for p in paths:
        try:
            n = rf.Network(p)
            nets.append(n)
        except Exception as e:
            errors.append(f"{os.path.basename(p)} — {e}")
    if errors:
        messagebox.showwarning("Some files could not be read", "\n".join(errors))
    return nets


def plot_sparams(nets: List[rf.Network], params: List[str]):
    if not nets:
        messagebox.showerror("No data", "No valid Touchstone files to plot.")
        return

    # Compute overall frequency range
    all_f = np.concatenate([n.f for n in nets if hasattr(n, "f") and n.f is not None])
    scale, unit = choose_freq_unit(all_f)

    plt.figure()

    # Build the plot
    for n in nets:
        for p in params:
            try:
                i, j = parse_param(p)
            except ValueError:
                continue

            if max(i, j) >= n.nports:
                # Skip if Sij not available for this network
                continue

            sij = n.s[:, i, j]
            mag_db = 20 * np.log10(np.abs(sij))
            label = f"{p} | {os.path.basename(n.name or n.filename or 'network')}"
            plt.plot(n.f / scale, mag_db, label=label)

    plt.xlabel(f"Frequency ({unit})")
    plt.ylabel("|S| (dB)")
    plt.title("Touchstone Viewer — Magnitude (dB) vs Linear Frequency")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


# ------------------------------
# Main entry
# ------------------------------

def main():
    root = tk.Tk()
    root.withdraw()  # hide main window — we only need dialogs

    filetypes = [
        ("Touchstone 2-port (*.s2p)", "*.s2p *.S2P"),
        ("Touchstone N-port (*.sNp)", "*.s*p *.S*P"),
        ("All files", "*.*"),
    ]

    paths = filedialog.askopenfilenames(
        title="Select Touchstone file(s)",
        filetypes=filetypes,
    )

    if not paths:
        messagebox.showinfo("Cancelled", "No files selected.")
        return

    nets = read_networks(list(paths))
    if not nets:
        return

    # Prompt for S-parameter selection (2-port oriented UI)
    selector = ParamSelector(root)
    root.wait_window(selector)
    if not selector.selected:
        messagebox.showinfo("Cancelled", "No S-parameters selected.")
        return

    plot_sparams(nets, selector.selected)


if __name__ == "__main__":
    main()
