# gds2bmp_quantized.py
import math
from typing import Iterable, Tuple, Optional, Dict, List
import gdstk
import numpy as np
from PIL import Image

def rasterize_gds_to_bmp(
    gds_path: str,
    out_path: str,
    cell_name: Optional[str] = None,
    spacing_um: float = 40.0,               # ☆ 每个墨滴/网格的间距（μm）
    layers: Optional[Iterable[Tuple[int, int]]] = None,
    margin_um: float = 0.0,                 # 四周物理边距（μm）
    invert: bool = False,                   # True：图形白/背景黑；False：图形黑/背景白
) -> Dict:
    """
    将 GDSII 以喷墨打印的“墨滴网格”规则量化为二值 BMP（1-bit）。
    规则：以 spacing_um 为网格步长，在每个像素中心进行点采样；
          只要像素中心落在任一目标多边形内部，则该像素置前景色。

    参数
    ----
    gds_path  : GDS 文件路径
    out_path  : 输出 BMP 路径（.bmp）
    cell_name : 要渲染的顶层 cell 名；None 时自动取库里所有顶层并合并
    spacing_um: 墨滴中心间距（μm），例如 40.0
    layers    : 需要渲染的 (layer, datatype) 列表；None 表示全部
    margin_um : 四周外扩（μm）
    invert    : True 则多边形为白，背景为黑；默认 False（多边形黑，背景白）

    返回
    ----
    一些元数据：像素尺寸、物理尺寸、等效DPI、层等。
    """

    if spacing_um <= 0:
        raise ValueError("spacing_um 必须为正数（μm）。")

    lib = gdstk.read_gds(gds_path)

    # 物理单位：每 1 坐标单位对应多少米
    unit_m = lib.unit
    um_per_unit = unit_m * 1e6  # 每单位多少微米

    # 选择顶层 cell
    if cell_name:
        if cell_name not in lib.cells:
            raise RuntimeError(f"未找到 cell: {cell_name}")
        top_cells = [lib.cells[cell_name]]
    else:
        top_cells = lib.top_level()

    if not top_cells:
        raise RuntimeError("未找到顶层 cell。请检查 GDS 或指定 cell_name。")

    # 收集多边形（合并所有顶层），并按层过滤
    polys_by_spec: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for cell in top_cells:
        polys = cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
        for poly in polys:  # poly: gdstk.Polygon
            key = (poly.layer, poly.datatype)
            if (layers is None) or (key in set(layers)):
                polys_by_spec.setdefault(key, []).append(poly.points)

    # 合并所有需要的多边形以计算边界，并供 inside 测试使用
    all_polys: List[np.ndarray] = [p for plist in polys_by_spec.values() for p in plist]
    if not all_polys:
        raise RuntimeError("没有多边形（按所选图层过滤后为空）。")

    # 计算原始边界（单位：坐标单位）
    mins = np.min([np.min(p, axis=0) for p in all_polys], axis=0)
    maxs = np.max([np.max(p, axis=0) for p in all_polys], axis=0)

    # 外扩边距（转为坐标单位）
    if margin_um > 0:
        margin_units = margin_um / um_per_unit
        mins = mins - margin_units
        maxs = maxs + margin_units

    width_units = float(maxs[0] - mins[0])
    height_units = float(maxs[1] - mins[1])

    # 将 spacing_um 转为坐标单位下的网格步长
    step_units = spacing_um / um_per_unit  # 每“像素”的物理步长（坐标单位）

    # 计算像素数量（网格数量）
    # 用 ceil 以完整覆盖边界（最后一列/行的中心可能略超边界一点，这在打印中通常可接受）
    px_w = max(1, int(math.ceil(width_units / step_units)))
    px_h = max(1, int(math.ceil(height_units / step_units)))

    # 计算每个像素中心在 GDS 坐标中的位置
    # 像素 i 的中心：x = mins_x + (i + 0.5)*step_units
    # 注意图像坐标 y 轴向下，为了保持“上->下”的视觉一致性，我们让 j=0 对应靠上（y 最大）
    # 因此中心 y = maxs_y - (j + 0.5)*step_units
    ix = np.arange(px_w, dtype=np.float64)
    jx = np.arange(px_h, dtype=np.float64)

    xs = mins[0] + (ix + 0.5) * step_units
    ys = maxs[1] - (jx + 0.5) * step_units

    # 生成所有中心点坐标（N×2），用于一次性 inside 测试
    XX, YY = np.meshgrid(xs, ys)          # YY 第0行对应最上方
    pts = np.column_stack((XX.ravel(), YY.ravel()))

    # 点是否在多边形内部（含边界）
    # gdstk.inside 支持批量判断，返回布尔数组
    inside = np.array(gdstk.inside(pts, all_polys))
    mask = inside.reshape((px_h, px_w))

    # 生成 1-bit 图（PIL '1'，0=黑，1=白）
    bg = 0 if invert else 1
    fg = 1 - bg
    img_np = np.full((px_h, px_w), fill_value=bg, dtype=np.uint8)
    img_np[mask] = fg

    img = Image.fromarray(img_np * 255, mode="L").convert("1")

    # 保存 BMP（注：BMP 不一定携带 DPI；如需嵌入，可改存 PNG/JPEG 并传入 dpi）
    img.save(out_path, format="BMP")

    # 计算物理尺寸（inch）与等效 DPI（用于记录/对比）
    meters_per_unit = unit_m
    width_m = width_units * meters_per_unit
    height_m = height_units * meters_per_unit
    width_in = width_m / 0.0254
    height_in = height_m / 0.0254
    effective_dpi = 25400.0 / spacing_um  # 每英寸像素 = 每英寸μm / spacing_um

    return {
        "output": out_path,
        "pixel_size": (px_w, px_h),
        "physical_size_in": (width_in, height_in),
        "spacing_um": spacing_um,
        "effective_dpi": effective_dpi,
        "layers_rendered": sorted(list(polys_by_spec.keys())),
        "unit_meter_per_coord": unit_m,
        "note": "按墨滴中心点采样量化（inside test, inclusive border）。",
    }

if __name__ == "__main__":
    meta = rasterize_gds_to_bmp(
        gds_path="SapphireElectrode.Export.gds",
        out_path="SapphireElectrode.ExportDS40.bmp",
        cell_name=None,
        spacing_um=40.0,             # 40 μm 网格
        layers=None,                  # 需要的话可指定 [(layer, datatype), ...]
        margin_um=10.0,
        invert=False,
    )
    print(meta)
