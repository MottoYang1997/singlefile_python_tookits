# gds2bmp_gui.py
import math
import os
import tempfile
import threading
import traceback
from typing import Iterable, Tuple, Optional, Dict, List

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

try:
    import gdstk
except ImportError:
    raise SystemExit("未安装 gdstk，请先 `pip install gdstk`")


# ===========================
#   核心渲染：喷墨网格量化
# ===========================
def rasterize_gds_to_bmp(
    gds_path: str,
    out_path: str,
    cell_name: Optional[str] = None,
    spacing_um: float = 40.0,               # 墨滴/网格间距（μm）
    layers: Optional[Iterable[Tuple[int, int]]] = None,
    margin_um: float = 0.0,
    invert: bool = False,
) -> Dict:
    if spacing_um <= 0:
        raise ValueError("spacing_um 必须为正数（μm）。")

    lib = gdstk.read_gds(gds_path)

    # 物理单位：每 1 坐标单位对应多少米
    unit_m = lib.unit
    um_per_unit = unit_m * 1e6  # 每单位多少微米

    # 顶层 cell
    if cell_name:
        if cell_name not in lib.cells:
            raise RuntimeError(f"未找到 cell: {cell_name}")
        top_cells = [lib.cells[cell_name]]
    else:
        top_cells = lib.top_level()

    if not top_cells:
        raise RuntimeError("未找到顶层 cell。请检查 GDS 或指定 cell_name。")

    # 收集多边形并按层过滤
    polys_by_spec: Dict[Tuple[int, int], List[np.ndarray]] = {}
    layer_set = set(layers) if layers is not None else None

    for cell in top_cells:
        polys = cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
        for poly in polys:  # gdstk.Polygon
            key = (poly.layer, poly.datatype)
            if (layer_set is None) or (key in layer_set):
                polys_by_spec.setdefault(key, []).append(poly.points)

    all_polys: List[np.ndarray] = [p for plist in polys_by_spec.values() for p in plist]
    if not all_polys:
        raise RuntimeError("没有多边形（按所选图层过滤后为空）。")

    # 边界（坐标单位）
    mins = np.min([np.min(p, axis=0) for p in all_polys], axis=0)
    maxs = np.max([np.max(p, axis=0) for p in all_polys], axis=0)

    # 外扩边距（坐标单位）
    if margin_um > 0:
        margin_units = margin_um / um_per_unit
        mins = mins - margin_units
        maxs = maxs + margin_units

    width_units = float(maxs[0] - mins[0])
    height_units = float(maxs[1] - mins[1])

    # 网格步长（坐标单位）
    step_units = spacing_um / um_per_unit

    # 像素数量（网格数量）
    px_w = max(1, int(math.ceil(width_units / step_units)))
    px_h = max(1, int(math.ceil(height_units / step_units)))

    # 像素中心坐标
    ix = np.arange(px_w, dtype=np.float64)
    jx = np.arange(px_h, dtype=np.float64)
    xs = mins[0] + (ix + 0.5) * step_units
    ys = maxs[1] - (jx + 0.5) * step_units

    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack((XX.ravel(), YY.ravel()))

    result = np.array(gdstk.inside(pts, all_polys))
    mask = result.reshape((px_h, px_w))

    # 1-bit 图（PIL '1'：0=黑，1=白）
    bg = 0 if invert else 1
    fg = 1 - bg
    img_np = np.full((px_h, px_w), fill_value=bg, dtype=np.uint8)
    img_np[mask] = fg

    img = Image.fromarray(img_np * 255, mode="L").convert("1")
    img.save(out_path, format="BMP")

    # 物理尺寸与等效 DPI
    width_in = (width_units * unit_m) / 0.0254
    height_in = (height_units * unit_m) / 0.0254
    effective_dpi = 25400.0 / spacing_um

    return {
        "output": out_path,
        "pixel_size": (px_w, px_h),
        "physical_size_in": (width_in, height_in),
        "spacing_um": spacing_um,
        "effective_dpi": effective_dpi,
        "layers_rendered": sorted(list(polys_by_spec.keys())),
        "unit_meter_per_coord": unit_m,
        "note": "按墨滴中心点采样量化（inside test, 边界记作 inside）。",
    }


# ===========================
#          GUI
# ===========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GDS → BMP（喷墨量化 / 40µm 网格）")
        self.geometry("980x680")

        self.gds_path = tk.StringVar()
        self.out_path = tk.StringVar()
        self.spacing_um = tk.DoubleVar(value=40.0)
        self.margin_um = tk.DoubleVar(value=0.0)
        self.invert = tk.BooleanVar(value=False)

        self.cell_name = tk.StringVar()
        self.layer_filter_enabled = tk.BooleanVar(value=False)

        self.lib = None
        self.top_cells = []
        self.layers_available: List[Tuple[int, int]] = []

        # 预览相关
        self.preview_image_tk = None
        self.preview_pil = None
        self.preview_win: Optional[tk.Toplevel] = None
        self.preview_canvas: Optional[tk.Canvas] = None
        self.preview_canvas_img = None

        # 缓存文件追踪（预览 BMP）
        self._temp_files = set()

        self._build_ui()

        # 退出前清理缓存
        self.protocol("WM_DELETE_WINDOW", self.on_app_close)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", **pad)

        # GDS 选择
        ttk.Label(frm_top, text="GDS 文件：").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm_top, textvariable=self.gds_path, width=60).grid(row=0, column=1, sticky="we")
        ttk.Button(frm_top, text="浏览…", command=self.browse_gds).grid(row=0, column=2, sticky="w")

        # 输出
        ttk.Label(frm_top, text="输出 BMP：").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm_top, textvariable=self.out_path, width=60).grid(row=1, column=1, sticky="we")
        ttk.Button(frm_top, text="保存为…", command=self.choose_out).grid(row=1, column=2, sticky="w")

        frm_top.columnconfigure(1, weight=1)

        # 参数
        frm_params = ttk.LabelFrame(self, text="参数")
        frm_params.pack(fill="x", **pad)

        ttk.Label(frm_params, text="墨滴间距 spacing (µm)：").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm_params, textvariable=self.spacing_um, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(frm_params, text="边距 margin (µm)：").grid(row=0, column=2, sticky="e")
        ttk.Entry(frm_params, textvariable=self.margin_um, width=10).grid(row=0, column=3, sticky="w")

        ttk.Checkbutton(frm_params, text="反色（图白/底黑）", variable=self.invert).grid(row=0, column=4, sticky="w")

        # cell / layer
        frm_sel = ttk.LabelFrame(self, text="Cell / 图层选择")
        frm_sel.pack(fill="both", **pad)

        ttk.Label(frm_sel, text="顶层 Cell：").grid(row=0, column=0, sticky="e")
        self.cb_cell = ttk.Combobox(frm_sel, textvariable=self.cell_name, width=40, state="readonly")
        self.cb_cell.grid(row=0, column=1, sticky="w")
        ttk.Button(frm_sel, text="刷新 GDS 信息", command=self.scan_gds).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(frm_sel, text="仅渲染选中图层", variable=self.layer_filter_enabled,
                        command=self.on_layer_toggle).grid(row=1, column=0, sticky="e")
        self.lst_layers = tk.Listbox(frm_sel, selectmode="multiple", height=8, exportselection=False)
        self.lst_layers.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=4)
        frm_sel.rowconfigure(1, weight=1)
        frm_sel.columnconfigure(1, weight=1)

        # 操作按钮
        frm_ops = ttk.Frame(self)
        frm_ops.pack(fill="x", **pad)
        ttk.Button(frm_ops, text="渲染 BMP", command=self.run_render_threaded).pack(side="left")
        ttk.Button(frm_ops, text="预览（快速）", command=self.preview_threaded).pack(side="left")
        ttk.Button(frm_ops, text="清空日志", command=self.clear_log).pack(side="right")

        # 日志
        frm_log = ttk.LabelFrame(self, text="日志 / 元数据")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(frm_log, height=20, wrap="word")
        self.txt_log.pack(fill="both", expand=True)

    # ============= 工具函数 =============
    def log(self, s: str):
        self.txt_log.insert("end", s + "\n")
        self.txt_log.see("end")

    def clear_log(self):
        self.txt_log.delete("1.0", "end")

    def browse_gds(self):
        path = filedialog.askopenfilename(
            title="选择 GDS 文件",
            filetypes=[("GDSII", "*.gds *.gdsii"), ("所有文件", "*.*")]
        )
        if path:
            self.gds_path.set(path)
            base, _ = os.path.splitext(path)
            self.out_path.set(base + "_DS.bmp")
            self.scan_gds()

    def choose_out(self):
        path = filedialog.asksaveasfilename(
            title="保存 BMP",
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp")]
        )
        if path:
            self.out_path.set(path)

    def on_layer_toggle(self):
        state = "normal" if self.layer_filter_enabled.get() else "disabled"
        self.lst_layers.configure(state=state)

    # ============= GDS 扫描 =============
    def scan_gds(self):
        path = self.gds_path.get().strip()
        if not path:
            messagebox.showwarning("提示", "请先选择 GDS 文件。")
            return
        try:
            self.lib = gdstk.read_gds(path)
            self.top_cells = self.lib.top_level()
            names = [c.name for c in self.top_cells]
            self.cb_cell["values"] = ["<全部顶层合并>"] + names
            self.cb_cell.set(self.cb_cell["values"][0])

            layer_pairs = set()
            for c in self.top_cells:
                polys = c.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
                for poly in polys:
                    layer_pairs.add((poly.layer, poly.datatype))
            self.layers_available = sorted(layer_pairs)

            self.lst_layers.delete(0, "end")
            for (layer, dtype) in self.layers_available:
                self.lst_layers.insert("end", f"Layer {layer}, Datatype {dtype}")

            self.on_layer_toggle()
            self.log(f"读取成功：{os.path.basename(path)}")
            self.log(f"顶层 cells: {', '.join(names) if names else '(无)'}")
            self.log(f"图层数：{len(self.layers_available)}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("读取失败", str(e))

    # ============= 读参数 =============
    def get_selected_layers(self) -> Optional[List[Tuple[int, int]]]:
        if not self.layer_filter_enabled.get():
            return None
        idxs = self.lst_layers.curselection()
        if not idxs:
            return []
        return [self.layers_available[i] for i in idxs]

    def get_selected_cell(self) -> Optional[str]:
        val = self.cb_cell.get()
        if not val or val == "<全部顶层合并>":
            return None
        return val

    # ============= 渲染/预览（线程） =============
    def run_render_threaded(self):
        threading.Thread(target=self.run_render, daemon=True).start()

    def preview_threaded(self):
        threading.Thread(target=self.preview, daemon=True).start()

    # ============= 渲染 BMP =============
    def run_render(self):
        gds = self.gds_path.get().strip()
        outp = self.out_path.get().strip()
        if not gds:
            messagebox.showwarning("提示", "请先选择 GDS 文件。")
            return
        if not outp:
            messagebox.showwarning("提示", "请先选择输出 BMP 路径。")
            return

        try:
            meta = rasterize_gds_to_bmp(
                gds_path=gds,
                out_path=outp,
                cell_name=self.get_selected_cell(),
                spacing_um=float(self.spacing_um.get()),
                layers=self.get_selected_layers(),
                margin_um=float(self.margin_um.get()),
                invert=bool(self.invert.get()),
            )
            self.log("渲染完成：")
            for k, v in meta.items():
                self.log(f"  {k}: {v}")
            messagebox.showinfo("完成", f"已输出：\n{outp}")
        except Exception as e:
            self.log("渲染失败：\n" + traceback.format_exc())
            messagebox.showerror("出错", str(e))

    # ============= 预览（独立可缩放窗口） =============
    def preview(self):
        gds = self.gds_path.get().strip()
        if not gds:
            messagebox.showwarning("提示", "请先选择 GDS 文件。")
            return

        try:
            # 生成临时 BMP
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
                tmp_out = tmp.name
            self._temp_files.add(tmp_out)

            meta = rasterize_gds_to_bmp(
                gds_path=gds,
                out_path=tmp_out,
                cell_name=self.get_selected_cell(),
                spacing_um=float(self.spacing_um.get()),
                layers=self.get_selected_layers(),
                margin_um=float(self.margin_um.get()),
                invert=bool(self.invert.get()),
            )

            # 打开预览窗口（若已存在则复用）
            self.open_preview_window()

            # 载入原始图像
            self.preview_pil = Image.open(tmp_out)
            self.preview_render_fit()  # 根据窗口当前大小做一次缩放绘制

            self.log("预览生成：")
            for k, v in meta.items():
                self.log(f"  {k}: {v}")

        except Exception as e:
            self.log("预览失败：\n" + traceback.format_exc())
            messagebox.showerror("出错", str(e))

    def open_preview_window(self):
        if self.preview_win is not None and tk.Toplevel.winfo_exists(self.preview_win):
            # 已存在：前置
            self.preview_win.deiconify()
            self.preview_win.lift()
            return

        self.preview_win = tk.Toplevel(self)
        self.preview_win.title("预览（可缩放）")
        self.preview_win.geometry("900x600")
        self.preview_win.minsize(300, 200)
        self.preview_win.transient(self)
        self.preview_win.focus_set()

        # 预览用 Canvas，自适应铺满
        self.preview_canvas = tk.Canvas(self.preview_win, bg="#222")
        self.preview_canvas.pack(fill="both", expand=True)

        # 窗口大小变化时，重绘缩放图
        self.preview_win.bind("<Configure>", self.on_preview_configure)

        # 关闭预览窗口时仅销毁窗口，不清理 temp（由主窗口退出统一清理）
        self.preview_win.protocol("WM_DELETE_WINDOW", self.preview_win.destroy)

    def on_preview_configure(self, event):
        # 防抖：仅当是 Canvas 或窗口尺寸改变时重绘
        if self.preview_pil is None or self.preview_canvas is None:
            return
        self.preview_render_fit()

    def preview_render_fit(self):
        """把 self.preview_pil 按窗口大小做最近邻缩放并绘制到 Canvas。"""
        if self.preview_pil is None or self.preview_canvas is None:
            return
        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())

        # 维持比例缩放到可视区域内
        W, H = self.preview_pil.size
        scale = min(cw / W, ch / H)
        tw, th = max(1, int(W * scale)), max(1, int(H * scale))

        img = self.preview_pil.resize((tw, th), Image.NEAREST)
        self.preview_image_tk = ImageTk.PhotoImage(img)

        self.preview_canvas.delete("all")
        x = (cw - tw) // 2
        y = (ch - th) // 2
        self.preview_canvas_img = self.preview_canvas.create_image(x, y, anchor="nw", image=self.preview_image_tk)

    # ============= 退出清理 =============
    def on_app_close(self):
        # 先尝试删除所有临时文件
        for p in list(self._temp_files):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
            finally:
                self._temp_files.discard(p)
        # 再正常退出
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
