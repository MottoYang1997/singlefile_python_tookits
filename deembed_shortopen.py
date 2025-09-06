#!/usr/bin/env python3
"""
Short-Open De-embedder (scikit-rf)

Workflow
1) Pick a SHORT Touchstone file (.s2p / .sNp)
2) Pick an OPEN Touchstone file
3) Pick one or more DUT Touchstone files
4) Choose S-parameters to plot (S11, S12, S21, S22)
5) The app de-embeds each DUT using Short-Open and plots |S| (dB) vs linear frequency
6) Optionally saves de-embedded DUT(s) next to the originals with suffix "_deembedded.s2p"

Notes
- This performs *de-embedding* of a 2-port fixture using scikit-rf's `ShortOpen` class.
- If you actually need a *one-port VNA calibration* with Short+Open only, that's under-determined; consider adding a Load (SOL) or using PHN with half-knowns.

Requirements
    pip install scikit-rf matplotlib
(If tkinter is missing on your OS, install python3-tk via your package manager.)
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
    from skrf.calibration import ShortOpen
except Exception as e:
    raise SystemExit(
        "Error: scikit-rf is required. Install with `pip install scikit-rf`.\n"
        f"Underlying error: {e}"
    )


# ------------------------------
# UI helpers
# ------------------------------
class ParamSelector(tk.Toplevel):
    def __init__(self, master: tk.Tk | tk.Toplevel):
        super().__init__(master)
        self.title("Select S-parameters to plot")
        self.resizable(False, False)
        self.grab_set()

        self.vars = {p: tk.IntVar(value=1 if p in ("S11", "S21") else 0) for p in ("S11", "S12", "S21", "S22")}

        tk.Label(self, text="Choose one or more S-parameters:").grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6), sticky="w")

        for r, p in enumerate(("S11", "S12", "S21", "S22"), start=1):
            tk.Checkbutton(self, text=p, variable=self.vars[p]).grid(row=r, column=0, sticky="w", padx=16)

        self.save_var = tk.IntVar(value=1)
        tk.Checkbutton(self, text="Save de-embedded .s2p next to DUT", variable=self.save_var).grid(row=5, column=0, sticky="w", padx=16, pady=(6, 0))

        btns = tk.Frame(self)
        btns.grid(row=6, column=0, columnspan=2, pady=12)
        tk.Button(btns, text="Run", width=10, command=self._ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Cancel", width=10, command=self._cancel).pack(side=tk.LEFT, padx=6)

        self.selected: List[str] | None = None
        self.save_choice: bool = False

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
        self.save_choice = bool(self.save_var.get())
        self.destroy()

    def _cancel(self):
        self.selected = None
        self.destroy()


# ------------------------------
# Core helpers
# ------------------------------

def parse_param(param: str) -> Tuple[int, int]:
    p = param.strip().upper()
    if len(p) != 3 or p[0] != "S" or not p[1:].isdigit():
        raise ValueError(f"Invalid S-parameter: {param}")
    i = int(p[1]) - 1
    j = int(p[2]) - 1
    if i < 0 or j < 0:
        raise ValueError(f"Invalid S-parameter: {param}")
    return i, j


def choose_freq_unit(freqs_hz: np.ndarray) -> Tuple[float, str]:
    fmax = float(np.nanmax(freqs_hz)) if freqs_hz.size else 0.0
    if fmax >= 1e9:
        return 1e9, "GHz"
    if fmax >= 1e6:
        return 1e6, "MHz"
    if fmax >= 1e3:
        return 1e3, "kHz"
    return 1.0, "Hz"


def read_network(path: str) -> rf.Network | None:
    try:
        return rf.Network(path)
    except Exception as e:
        messagebox.showerror("Read error", f"{os.path.basename(path)} — {e}")
        return None


def read_networks(paths: List[str]) -> List[rf.Network]:
    nets: List[rf.Network] = []
    errors: List[str] = []
    for p in paths:
        try:
            nets.append(rf.Network(p))
        except Exception as e:
            errors.append(f"{os.path.basename(p)} — {e}")
    if errors:
        messagebox.showwarning("Some DUT files could not be read", "\n".join(errors))
    return nets


def ensure_two_port(n: rf.Network, label: str) -> bool:
    if getattr(n, 'nports', None) != 2:
        messagebox.showwarning("Non 2-port skipped", f"{label} is {getattr(n, 'nports', '?')}-port. This tool expects 2-port networks.")
        return False
    return True


def interpolate_like(ref: rf.Network, target: rf.Network) -> rf.Network:
    if ref.f.shape == target.f.shape and np.allclose(ref.f, target.f, rtol=0, atol=1e-9):
        return ref
    try:
        return ref.interpolate_to_frequency(target.frequency)
    except Exception:
        return ref.interpolate_to_frequency(rf.Frequency.from_f(target.f, unit='hz'))


def safe_basename(n: rf.Network) -> str:
    return os.path.basename((n.name or n.filename or 'network')).replace(os.sep, '_')


# ------------------------------
# Plotting and export
# ------------------------------

def plot_deembedded(dut_pairs: List[Tuple[rf.Network, rf.Network]], params: List[str]):
    if not dut_pairs:
        messagebox.showerror("No data", "Nothing to plot.")
        return

    all_f = np.concatenate([de.f for _, de in dut_pairs])
    scale, unit = choose_freq_unit(all_f)

    plt.figure()

    for raw, de in dut_pairs:
        for p in params:
            try:
                i, j = parse_param(p)
            except ValueError:
                continue
            if max(i, j) >= de.nports:
                continue
            mag_db = 20 * np.log10(np.abs(de.s[:, i, j]))
            label = f"{p} | {safe_basename(raw)} (de-emb)"
            plt.plot(de.f / scale, mag_db, label=label)

    plt.xlabel(f"Frequency ({unit})")
    plt.ylabel("|S| (dB)")
    plt.title("Short-Open De-embedded |S| vs Frequency")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


def export_s2p(dut_pairs: List[Tuple[rf.Network, rf.Network]]):
    saved = []
    errors = []
    for raw, de in dut_pairs:
        try:
            raw_path = raw.filename if raw.filename else (raw.name + ".s2p")
            root, ext = os.path.splitext(raw_path)
            out_path = root + "_deembedded" + (ext if ext else ".s2p")
            de.write_touchstone(out_path)
            saved.append(out_path)
        except Exception as e:
            errors.append(f"{safe_basename(raw)} — {e}")
    if saved:
        messagebox.showinfo("Saved", "\n".join(saved))
    if errors:
        messagebox.showwarning("Some files not saved", "\n".join(errors))


# ------------------------------
# Main app
# ------------------------------

def main():
    root = tk.Tk()
    root.withdraw()

    # 1) Select SHORT
    short_path = filedialog.askopenfilename(
        title="Select SHORT Touchstone",
        filetypes=[("Touchstone (*.s2p, *.sNp)", "*.s2p *.s*p *.S2P *.S*P"), ("All files", "*.*")],
    )
    if not short_path:
        messagebox.showinfo("Cancelled", "No SHORT selected.")
        return

    short_net = read_network(short_path)
    if short_net is None or not ensure_two_port(short_net, "SHORT"):
        return

    # 2) Select OPEN
    open_path = filedialog.askopenfilename(
        title="Select OPEN Touchstone",
        filetypes=[("Touchstone (*.s2p, *.sNp)", "*.s2p *.s*p *.S2P *.S*P"), ("All files", "*.*")],
    )
    if not open_path:
        messagebox.showinfo("Cancelled", "No OPEN selected.")
        return

    open_net = read_network(open_path)
    if open_net is None or not ensure_two_port(open_net, "OPEN"):
        return

    # 3) Select DUT(s)
    dut_paths = filedialog.askopenfilenames(
        title="Select DUT file(s)",
        filetypes=[("Touchstone (*.s2p, *.sNp)", "*.s2p *.s*p *.S2P *.S*P"), ("All files", "*.*")],
    )
    if not dut_paths:
        messagebox.showinfo("Cancelled", "No DUT files selected.")
        return

    duts = [n for n in read_networks(list(dut_paths)) if ensure_two_port(n, safe_basename(n))]
    if not duts:
        return

    # 4) Parameter selection + save option
    selector = ParamSelector(root)
    root.wait_window(selector)
    if not selector.selected:
        messagebox.showinfo("Cancelled", "No S-parameters selected.")
        return

    params = selector.selected
    do_save = selector.save_choice

    # 5) De-embed each DUT using ShortOpen
    dut_pairs: List[Tuple[rf.Network, rf.Network]] = []
    for dut in duts:
        try:
            s = interpolate_like(short_net, dut)
            o = interpolate_like(open_net, dut)
            so = ShortOpen(s, o)
            de = so.deembed(dut)
            de.name = (dut.name or safe_basename(dut)) + "_deembedded"
            dut_pairs.append((dut, de))
        except Exception as e:
            messagebox.showwarning("De-embed error", f"{safe_basename(dut)} — {e}")

    if not dut_pairs:
        messagebox.showerror("Failed", "No DUTs were de-embedded.")
        return

    # 6) Plot
    plot_deembedded(dut_pairs, params)

    # 7) Optional export
    if do_save:
        export_s2p(dut_pairs)


if __name__ == "__main__":
    main()
