"""Bước 6b: giao diện web (Gradio) cho model đã chọn - dùng lại hàm dự đoán của 06_predict.py.

Chạy:
    python main.py app                 # mở http://127.0.0.1:7860
    python main.py app --port 7861
    python main.py app --share         # tạo link công khai tạm thời của Gradio (ai có link đều dùng được)

Cả hai tab cùng cơ chế: phân tích 8 aspect rồi ra một kết luận tổng (overall_verdict của 06_predict.py,
cùng ngưỡng chỉ số khen), hiển thị bằng cùng một thẻ kết luận (verdict_card).
Tab "Một comment": thẻ kết luận, chart điểm khen − không khen của 8 aspect (bar diverging, trục cố định ±5),
                   bảng chi tiết (lớp, margin, từ ảnh hưởng) trong mục "Bảng chi tiết".
Tab "File CSV"    : thẻ kết luận, tóm tắt tỉ lệ khen theo aspect (theo nhóm nếu chọn), tải file dự đoán từng dòng.
"""

import argparse
import html
import importlib
import os
import sys

from absa_common import ASPECTS, PROJECT_DIR, normalize_absa, prep_frame
from clean_data import extract_template

# File upload / cache của Gradio nằm trong repo (không ghi ra ổ hệ thống), không gửi analytics
os.environ.setdefault("GRADIO_TEMP_DIR", str(PROJECT_DIR / ".gradio_tmp"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

predict_mod = importlib.import_module("06_predict")  # tên module bắt đầu bằng số nên không import trực tiếp được
VI, OUT_DIR, VERDICT = predict_mod.VI, predict_mod.OUT_DIR, predict_mod.VERDICT
LOW_MARGIN = 0.25
PIPE, META = predict_mod.load_model()

# Ví dụ tự viết cho demo, không lấy từ dữ liệu
EXAMPLES = [
    "gõ êm tay, tiếng thocky hay, led hơi tối, giao hàng nhanh, shop tư vấn nhiệt tình",
    "giá rẻ mà dùng 2 tuần đã liệt phím, nhắn shop không trả lời",
    "kết nối bluetooth hay bị trễ, pin tụt nhanh",
    "phím bấm hơi cứng, đèn đẹp",
    "đóng gói cẩn thận, hàng đúng mô tả, trong tầm giá thì quá ok",
]

# Trục cố định để so được giữa các comment: |polarity| <= 5 ở ~98% ô có nhắc của tập test (đo ngày 2026-10-29)
AXIS = 5.0

# Cặp diverging xanh dương (khen) <-> đỏ (không khen) của bảng màu dataviz, đã chạy validator cho light và dark
CSS = """
.absa-chart, .ov, .ct-wrap, .absa-meta { --pos: #2a78d6; --neg: #e34948; --mid: #a3a29d; --grid: var(--border-color-primary);
  font-size: 14px; color: var(--body-text-color); }
.dark .absa-chart, .dark .ov, .dark .ct-wrap, .dark .absa-meta { --pos: #3987e5; --neg: #e66767; --mid: #75746f; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.dot.pos { background: var(--pos); } .dot.non_pos { background: var(--neg); } .dot.mixed { background: var(--mid); }
.dot.none { border: 1px dashed var(--body-text-color-subdued); }
.dot.big { width: 14px; height: 14px; margin-right: 10px; }
.ov { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 16px 18px; background: var(--block-background-fill); }
.ov-top { display: flex; flex-wrap: wrap; gap: 14px 36px; align-items: flex-end; }
.ov-src { font-size: 13px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border-color-primary); }
.ov-k { color: var(--body-text-color-subdued); font-size: 13px; }
.ov-verdict { font-size: 24px; font-weight: 700; margin-top: 2px; }
.ov-v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px; }
.ov-bar { display: flex; gap: 2px; height: 10px; margin: 6px 0 6px; }
.ov-seg { border-radius: 3px; min-width: 3px; }
.ov-seg.pos { background: var(--pos); } .ov-seg.non_pos { background: var(--neg); } .ov-seg.mixed { background: var(--mid); }
.ov-seg.none { background: transparent; border: 1px dashed var(--body-text-color-subdued); }
.ov-legend { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 13px; }
.ov-list { margin: 12px 0 0; padding-left: 18px; line-height: 1.6; }
.ov-how { margin-top: 8px; font-size: 13px; color: var(--body-text-color-subdued); }
.ov-how summary { cursor: pointer; } .ov-how ul { padding-left: 18px; line-height: 1.55; }
.ct-wrap { max-height: 640px; overflow: auto; border: 1px solid var(--border-color-primary); border-radius: 8px; }
.ct { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; }
.ct th { position: sticky; top: 0; z-index: 2; background: var(--block-background-fill); text-align: left; padding: 8px;
  border-bottom: 1px solid var(--border-color-primary); white-space: nowrap; color: var(--body-text-color-subdued); font-weight: 600; }
.ct td { padding: 6px 8px; border-bottom: 1px solid var(--border-color-primary); vertical-align: top; }
.ct th, .ct td { border-left: none !important; border-right: none !important; border-top: none !important; }  /* bỏ khung ô mặc định của Gradio */
.ct .clamp { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.ct tr:hover .clamp { -webkit-line-clamp: unset; }
.ct tr:hover td { background: var(--background-fill-secondary); }
.ct .num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.ct td.txt { min-width: 260px; max-width: 460px; line-height: 1.45; }
.ct td.vd { white-space: nowrap; }
.ct .muted, .absa-meta .muted { color: var(--body-text-color-subdued); }
.ac-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 16px; margin-bottom: 10px; }
.ac-title { font-weight: 600; }
.ac-sub { color: var(--body-text-color-subdued); font-size: 13px; }
.ac-legend { display: flex; gap: 14px; align-items: center; color: var(--body-text-color-subdued); font-size: 13px; }
.ac-legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.ac-legend i.pos { background: var(--pos); } .ac-legend i.non_pos { background: var(--neg); }
.ac-legend i.none { background: transparent; border: 1px dashed var(--body-text-color-subdued); }
.ac-grid { display: grid; grid-template-columns: 96px minmax(0, 1fr) 64px; column-gap: 10px; align-items: center; }
.ac-row { position: relative; height: 32px; border-radius: 6px; outline: none; }
.ac-row:hover, .ac-row:focus { background: var(--background-fill-secondary); }
.ac-name { padding-left: 6px; }
.ac-row.none .ac-name, .ac-row.none .ac-val { color: var(--body-text-color-subdued); }
.ac-track { position: relative; height: 100%; }
.ac-track::before, .ac-axis-track::before { content: ""; position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--body-text-color-subdued); opacity: .55; }
.ac-track::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(to right, transparent calc(25% - .5px), var(--grid) calc(25% - .5px), var(--grid) calc(25% + .5px), transparent calc(25% + .5px),
    transparent calc(75% - .5px), var(--grid) calc(75% - .5px), var(--grid) calc(75% + .5px), transparent calc(75% + .5px)); }
.ac-bar { position: absolute; top: 9px; height: 14px; min-width: 3px; z-index: 1; }
.ac-bar.pos { left: 50%; background: var(--pos); border-radius: 0 4px 4px 0; }
.ac-bar.non_pos { right: 50%; background: var(--neg); border-radius: 4px 0 0 4px; }
.ac-none { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 1; padding: 0 6px;
  background: var(--block-background-fill); color: var(--body-text-color-subdued); font-size: 12px; white-space: nowrap; }
.ac-row:hover .ac-none, .ac-row:focus .ac-none { background: var(--background-fill-secondary); }
.ac-val { text-align: right; padding-right: 6px; font-variant-numeric: tabular-nums; }
.ac-tip { display: none; position: absolute; left: 106px; bottom: calc(100% + 4px); z-index: 10; max-width: 320px;
  padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); box-shadow: 0 4px 14px rgba(0,0,0,.15); font-size: 13px; line-height: 1.45; }
.ac-row:hover .ac-tip, .ac-row:focus .ac-tip { display: block; }
.ac-axis { margin-top: 4px; color: var(--body-text-color-subdued); font-size: 12px; font-variant-numeric: tabular-nums; }
.ac-axis-track { position: relative; height: 18px; }
.ac-axis-track span { position: absolute; top: 2px; transform: translateX(-50%); }
.ac-axis-names { display: flex; justify-content: space-between; }
.ac-count { position: absolute; top: 50%; transform: translateY(-50%); z-index: 1; font-size: 12px;
  color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }
.ac-complaint { margin: 0 0 8px; color: var(--body-text-color); } .ac-complaint summary { cursor: pointer; }
.ac-complaint ul { margin: 6px 0 0; padding-left: 18px; font-size: 13px; line-height: 1.5; }
.ac-score { color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; margin-right: 4px; }
.absa { width: 100%; border-collapse: collapse; font-size: 14px; }
.absa th, .absa td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border-color-primary); }
.absa th { color: var(--body-text-color-subdued); font-weight: 600; }
.absa tr.none td { color: var(--body-text-color-subdued); }
.absa td.num { font-variant-numeric: tabular-nums; }
.absa-meta { color: var(--body-text-color-subdued); margin: 4px 0 12px; }
.absa-warn { margin-top: 10px; color: var(--body-text-color); }
"""


def chart(P: pd.Series, M: pd.Series, S: pd.Series, R: dict) -> str:
    """Bar diverging: polarity (điểm khen - điểm không khen) của từng aspect, trục cố định ±AXIS."""
    rows = []
    for a in ASPECTS:
        c, m, s = P[a], M[a], S[a]
        if c == "none":
            rows.append(f"<div class='ac-grid ac-row none' tabindex='0'><div class='ac-name'>{a}</div>"
                        "<div class='ac-track'><span class='ac-none'>không nhắc</span></div><div class='ac-val'>-</div>"
                        f"<div class='ac-tip'><b>{a}</b> · không nhắc</div></div>")
            continue
        width = min(abs(s), AXIS) / AXIS * 50
        flag = " ⚠" if m < LOW_MARGIN else ""
        words = html.escape(", ".join(R.get(a, []))) or "-"
        clip = f"<br>Vượt trục ±{AXIS:g}" if abs(s) > AXIS else ""
        tip = (f"<b>{a}</b> · {VI[c]}<br>Điểm khen - không khen: {s:+.2f}<br>"
               f"Margin: {m:.2f}{' (model phân vân)' if flag else ''}<br>Từ ảnh hưởng: {words}{clip}")
        rows.append(f"<div class='ac-grid ac-row' tabindex='0'><div class='ac-name'>{a}</div>"
                    f"<div class='ac-track'><div class='ac-bar {c}' style='width:{width:.2f}%'></div></div>"
                    f"<div class='ac-val'>{s:+.2f}{flag}</div><div class='ac-tip'>{tip}</div></div>")
    ticks = "".join(f"<span style='left:{(t / AXIS + 1) * 50:g}%'>{t:+g}</span>" if t else "<span style='left:50%'>0</span>"
                    for t in (-AXIS, -AXIS / 2, 0, AXIS / 2, AXIS))
    label = "; ".join(f"{a} {VI[P[a]]}" + (f" {S[a]:+.2f}" if P[a] != "none" else "") for a in ASPECTS)
    return (f"<div class='absa-chart' role='img' aria-label='Điểm khen trừ không khen theo aspect: {html.escape(label)}'>"
            "<div class='ac-head'><div><div class='ac-title'>Điểm khen − không khen theo aspect</div>"
            "<div class='ac-sub'>Phải: nghiêng về khen · trái: nghiêng về không khen · rê chuột vào từng dòng để xem chi tiết</div></div>"
            "<div class='ac-legend'><span><i class='pos'></i>khen</span><span><i class='non_pos'></i>không khen</span>"
            "<span><i class='none'></i>không nhắc</span></div></div>"
            + "".join(rows)
            + f"<div class='ac-grid ac-axis'><div></div><div><div class='ac-axis-track'>{ticks}</div>"
            "<div class='ac-axis-names'><span>← không khen</span><span>khen →</span></div></div><div></div></div></div>")


def table(P: pd.Series, M: pd.Series, S: pd.Series, R: dict) -> str:
    """Bảng chi tiết cùng số liệu với chart (thay cho người không xem được chart)."""
    rows = []
    for a in ASPECTS:
        c = P[a]
        if c == "none":
            rows.append(f"<tr class='none'><td>{a}</td><td>không nhắc</td><td></td><td></td><td></td></tr>")
            continue
        flag = " ⚠" if M[a] < LOW_MARGIN else ""
        rows.append(f"<tr><td>{a}</td><td>{VI[c]}</td><td class='num'>{S[a]:+.2f}</td>"
                    f"<td class='num'>{M[a]:.2f}{flag}</td><td>{html.escape(', '.join(R.get(a, [])))}</td></tr>")
    return ("<table class='absa'><tr><th>Aspect</th><th>Lớp</th><th>Điểm khen − không khen</th><th>Margin</th>"
            "<th>Từ ảnh hưởng</th></tr>" + "".join(rows) + "</table>")


def analyze(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "<p class='absa-meta'>Nhập một comment để dự đoán.</p>", ""
    P, M, S, R = predict_mod.predict(PIPE, [text])
    # Cùng cơ chế với tab File CSV: kết luận tổng = overall_verdict trên các aspect của comment này
    V, ov = predict_mod.comment_verdict(P, S).iloc[0], predict_mod.overall_verdict(P)
    P, M, S, R = P.iloc[0], M.iloc[0], S.iloc[0], R[0]
    norm = prep_frame([extract_template(text)[0]])["comment_norm"][0]
    notes = []
    # Đếm từ giống analyze_file: dữ liệu train chỉ gồm comment >= 4 từ
    if len(normalize_absa(extract_template(text)[0]).split()) < 4:
        notes.append("Comment dưới 4 từ: loại này không có trong dữ liệu train, kết quả kém tin cậy hơn.")
    unsure = [a for a in ASPECTS if P[a] != "none" and M[a] < LOW_MARGIN]
    if unsure:
        notes.append(f"Model phân vân (margin < {LOW_MARGIN}) ở: {', '.join(unsure)}, đánh dấu ⚠ trong chart.")
    notes.append("Model hay bỏ sót lời chê nhẹ nằm trong câu khen (bước 5).")
    head = f"<p class='absa-meta'>Chuẩn hoá: {html.escape(norm)}</p>"
    return head + comment_overview(P, S, V, ov, notes) + chart(P, M, S, R), table(P, M, S, R)


def comment_overview(P: pd.Series, S: pd.Series, V: pd.Series, ov: dict, notes: list[str]) -> str:
    """Thẻ kết luận tổng cho một comment, cùng bố cục và ngưỡng với thẻ của file."""
    pos = S[P == "pos"].sort_values(ascending=False)
    neg = S[P == "non_pos"].sort_values()
    counts = {"pos": len(pos), "non_pos": len(neg), "none": len(ASPECTS) - len(pos) - len(neg)}
    names = {"pos": "khen", "non_pos": "không khen", "none": "không nhắc"}
    tiles = [("Chỉ số khen", f"{ov['share']:.0%}" if ov["n_mentions"] else "-"),
             ("Aspect được nhắc", f"{ov['n_pos']} khen / {ov['n_mentions']}"),
             ("Điểm comment", f"{V['score']:+.2f}" if pd.notna(V["score"]) else "-")]

    def fmt(s: pd.Series) -> str:
        return ", ".join(f"{a} ({v:+.2f})" for a, v in s.items())

    items = [f"<li><b>Được khen:</b> {fmt(pos) or 'không có'}</li>",
             f"<li><b>Cần lưu ý:</b> {fmt(neg) or 'không có aspect nào bị chê'}</li>"]
    unmentioned = [a for a in ASPECTS if P[a] == "none"]
    if unmentioned:
        items.append(f"<li><b>Không nhắc:</b> {', '.join(unmentioned)}</li>")
    items.append(f"<li><b>Nhãn ở bảng từng comment (tab File CSV):</b> {VERDICT[V['verdict']]}</li>")
    how = ["<b>Điểm comment</b> = trung bình điểm các aspect được nhắc (cùng cột Điểm ở bảng từng comment của tab File CSV).",
           "<b>Chỉ số khen</b> = số aspect khen / số aspect được nhắc trong comment. Kết luận: ≥ 80% tích cực · "
           "60–79% khá tích cực · 40–59% trái chiều · &lt; 40% tiêu cực (cùng ngưỡng với kết luận file)."]
    return verdict_card("", "Kết luận cho comment", ov["label"], tiles, "Aspect của comment", counts, names,
                        items, how, notes) + "<div style='height:14px'></div>"


def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file if isinstance(file, str) else file.name, encoding="utf-8-sig")


def on_upload(file):
    if file is None:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    cols = list(read_csv(file).columns)
    return (gr.update(choices=cols, value="comment" if "comment" in cols else cols[0]),
            gr.update(choices=["(không)", *cols], value="model" if "model" in cols else "(không)"))


def analyze_file(file, col, group):
    if file is None:
        raise gr.Error("Chưa chọn file CSV.")
    df = read_csv(file)
    if col not in df.columns:
        raise gr.Error(f"File không có cột '{col}'.")
    # Nhóm theo chính cột comment thì mỗi comment thành một nhóm -> coi như không nhóm
    group = None if group in (None, "(không)", col) else group
    source = describe_source(df)
    # Review chỉ chấm sao, không viết chữ -> bỏ, không đưa vào model
    text = df[col].fillna("").astype(str).str.strip()
    n_blank, n_dup = int((text == "").sum()), int(text[text != ""].duplicated().sum())
    df = df[text != ""].reset_index(drop=True)
    if df.empty:
        raise gr.Error(f"Cột '{col}' không có comment nào có chữ.")
    texts = df[col].astype(str).str.strip()
    P, M, S, _ = predict_mod.predict(PIPE, texts.tolist())
    V = predict_mod.comment_verdict(P, S)
    ov = predict_mod.overall_verdict(P)

    out = df.copy()
    for a in ASPECTS:
        out[f"{a}_pred"] = P[a].values
        out[f"{a}_score"] = S[a].where(P[a] != "none").round(2).values
    out["comment_score"] = V["score"].round(2).values
    out["comment_verdict"] = V["verdict"].map(VERDICT).values
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "app_pred.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    groups = df[group].astype(str).rename(group) if group else None
    rates = predict_mod.rate_table(P, groups).reset_index()
    rates.columns = [group or "", *ASPECTS]
    best = predict_mod.strengths(P, groups).reset_index()

    notes = []  # độ tin cậy của kết luận
    if len(df) < 30:
        notes.append(f"Chỉ có {len(df)} comment: kết luận mang tính tham khảo, nên crawl thêm.")
    if n_blank:
        notes.append(f"Bỏ {n_blank} dòng không có chữ (review chỉ chấm sao).")
    if n_dup:
        notes.append(f"{n_dup} comment trùng nội dung, vẫn giữ (kiểm tra lại file crawl nếu con số này lớn).")
    # Dữ liệu train chỉ gồm comment >= 4 từ (clean_data.py --min-words 4), comment ngắn hơn kém tin cậy
    n_short = int(texts.map(lambda t: len(normalize_absa(extract_template(t)[0]).split()) < 4).sum())
    if n_short:
        notes.append(f"{n_short} comment dưới 4 từ: loại này không có trong dữ liệu train, kết quả kém tin cậy hơn.")
    n_low = int((M.where(P != "none") < LOW_MARGIN).sum().sum())
    if n_low:
        notes.append(f"{n_low} ô (comment × aspect) model phân vân (margin < {LOW_MARGIN}), đánh dấu ⚠ trong bảng.")
    notes.append("Model hay bỏ sót lời chê nhẹ nằm trong câu khen (bước 5), nên tỉ lệ khen thực tế có thể thấp hơn vài điểm %.")
    if all(f"{a}_c3" in df.columns for a in ASPECTS):
        T = df[[f"{a}_c3" for a in ASPECTS]].set_axis(ASPECTS, axis=1)
        notes.append(f"File có sẵn nhãn: tỉ lệ ô khớp nhãn thật {(P.values == T.values).mean():.1%}.")
    return (overview(df, V, ov, notes, source), file_chart(P), complaints(texts, P, S),
            comment_table(texts, P, M, S, V), rates, best, str(pred_path))


def describe_source(df: pd.DataFrame) -> str:
    """Tên sản phẩm (cột product_name nếu có) và số dòng của file, để biết đang xem đúng file nào
    (file crawl của mọi sản phẩm đều tên reviews.csv)."""
    parts = []
    if "product_name" in df.columns:
        names = df["product_name"].dropna().astype(str).unique()
        if len(names) == 1:
            parts.append(f"Sản phẩm: <b>{html.escape(names[0])}</b>")
        elif len(names) > 1:
            parts.append(f"<b>{len(names)} sản phẩm</b> trong file")
    parts.append(f"File có {len(df)} dòng dữ liệu (không tính dòng tiêu đề)")
    return " · ".join(parts)


def stars(df: pd.DataFrame) -> str:
    """Phân bố số sao nếu file crawl có cột rating (1-5); để đối chiếu với kết luận của model."""
    if "rating" not in df.columns:
        return ""
    r = pd.to_numeric(df["rating"], errors="coerce").dropna()
    r = r[r.between(1, 5)].astype(int)
    if r.empty:
        return ""
    counts = r.value_counts().sort_index(ascending=False)
    return f"{r.mean():.1f}★ trung bình (" + ", ".join(f"{k}★: {v}" for k, v in counts.items()) + ")"


def verdict_card(source: str, kicker: str, label: str, tiles: list[tuple[str, str]], bar_title: str,
                 counts: dict[str, int], names: dict[str, str], items: list[str], how: list[str], notes: list[str]) -> str:
    """Thẻ kết luận tổng dùng chung cho tab Một comment và tab File CSV."""
    # Màu chấm theo nhãn của OVERALL trong 06_predict.py
    level = {"Tích cực": "pos", "Khá tích cực, có phàn nàn đáng chú ý": "pos", "Trái chiều": "mixed",
             "Tiêu cực": "non_pos"}.get(label, "none")
    n = sum(counts.values()) or 1
    seg = "".join(f"<span class='ov-seg {k}' style='flex:{c}' title='{names[k]}: {c}'></span>"
                  for k, c in counts.items() if c)
    legend = "".join(f"<span><i class='dot {k}'></i>{names[k]} <b>{c}</b> ({c / n:.0%})</span>" for k, c in counts.items())
    return (
        "<div class='ov'>" + (f"<div class='ov-src'>{source}</div>" if source else "")
        + f"<div class='ov-top'><div><div class='ov-k'>{kicker}</div>"
        f"<div class='ov-verdict'><i class='dot big {level}'></i>{html.escape(label)}</div></div>"
        + "".join(f"<div class='ov-tile'><div class='ov-k'>{k}</div><div class='ov-v'>{v}</div></div>" for k, v in tiles)
        + "</div>"
        f"<div class='ov-k' style='margin-top:14px'>{bar_title}</div><div class='ov-bar'>{seg}</div>"
        f"<div class='ov-legend'>{legend}</div>"
        f"<ul class='ov-list'>{''.join(items)}</ul>"
        "<details class='ov-how'><summary>Cách tính và độ tin cậy</summary><ul>"
        "<li><b>Điểm của aspect</b> = điểm khen − điểm không khen của SVM (không phải thang 1–10); &gt; 0 là khen.</li>"
        + "".join(f"<li>{h}</li>" for h in how)
        + "".join(f"<li>{html.escape(t)}</li>" for t in notes)
        + "</ul></details></div>")


def overview(df: pd.DataFrame, V: pd.DataFrame, ov: dict, notes: list[str], source: str) -> str:
    """Thẻ kết luận tổng cho cả file."""
    counts = {k: int((V["verdict"] == k).sum()) for k in VERDICT}

    def fmt(aspects):
        return ", ".join(f"{a} ({ov['rate'][a]:.0%} khen / {int(ov['mentions'][a])} lượt)" for a in aspects)

    best = fmt(ov["best"]) or "chưa đủ dữ liệu"
    worst = fmt(ov["worst"]) or "không có aspect nào bị chê (trong số aspect đủ lượt nhắc)"
    unmentioned = [a for a in ASPECTS if ov["mentions"][a] == 0]
    score = V["score"].mean()
    tiles = [("Comment phân tích", f"{len(V)}"),
             ("Chỉ số khen", f"{ov['share']:.0%}" if ov["n_mentions"] else "-"),
             ("Lượt nhắc aspect", f"{ov['n_pos']} khen / {ov['n_mentions']}"),
             ("Điểm TB / comment", f"{score:+.2f}" if pd.notna(score) else "-")]
    star = stars(df)
    items = [f"<li><b>Mạnh nhất:</b> {best}</li>", f"<li><b>Cần lưu ý:</b> {worst}</li>"]
    if unmentioned:
        items.append(f"<li><b>Không ai nhắc:</b> {', '.join(unmentioned)}</li>")
    if star:
        items.append(f"<li><b>Số sao trên Shopee:</b> {star}</li>")
    how = ["<b>Điểm comment</b> = trung bình điểm các aspect được nhắc. <b>Kết luận comment</b>: mọi aspect được nhắc đều khen → "
           "Tích cực; đều không khen → Không tích cực; có cả hai → Trái chiều; không nhắc aspect nào → Không xác định.",
           "<b>Chỉ số khen</b> = lượt khen / lượt nhắc aspect. Kết luận file: ≥ 80% tích cực · 60–79% khá tích cực · "
           "40–59% trái chiều · &lt; 40% tiêu cực (ngưỡng quy ước của demo, tab Một comment dùng cùng ngưỡng).",
           f"Điểm mạnh / cần lưu ý chỉ xét aspect có ít nhất {predict_mod.MIN_MENTIONS} lượt nhắc."]
    return verdict_card(source, "Kết luận cho file", ov["label"], tiles, "Kết luận từng comment", counts, VERDICT,
                        items, how, notes)


def comment_table(texts: pd.Series, P: pd.DataFrame, M: pd.DataFrame, S: pd.DataFrame, V: pd.DataFrame,
                  limit: int = 500) -> str:
    """Bảng từng comment: điểm của 8 aspect, điểm comment, kết luận."""
    # Điểm và Kết luận đặt ngay sau Comment để không bị đẩy ra ngoài khung khi bảng rộng
    head = ("<tr><th>#</th><th>Comment</th><th class='num'>Điểm</th><th>Kết luận</th>"
            + "".join(f"<th class='num'>{a}</th>" for a in ASPECTS) + "</tr>")
    rows = []
    for i in texts.index[:limit]:
        cells = []
        for a in ASPECTS:
            c = P.at[i, a]
            if c == "none":
                cells.append("<td class='num muted'>·</td>")
            else:
                flag = " ⚠" if M.at[i, a] < LOW_MARGIN else ""
                cells.append(f"<td class='num' title='{VI[c]}'><i class='dot {c}'></i>{S.at[i, a]:+.2f}{flag}</td>")
        sc, vd = V.at[i, "score"], V.at[i, "verdict"]
        score = f"<td class='num'><b>{sc:+.2f}</b></td>" if pd.notna(sc) else "<td class='num muted'>-</td>"
        text = html.escape(texts[i])
        rows.append(f"<tr><td class='muted'>{i + 1}</td><td class='txt'><div class='clamp' title='{text}'>{text}</div></td>"
                    f"{score}<td class='vd'><i class='dot {vd}'></i>{VERDICT[vd]}</td>{''.join(cells)}</tr>")
    shown = min(limit, len(texts))
    count = (f"<b>{len(texts)} comment</b>" if shown == len(texts)
             else f"<b>{shown}/{len(texts)} comment đầu</b> (xem đủ trong file tải về)")
    return (f"<p class='absa-meta'>{count} · cuộn trong khung để xem hết, rê chuột vào comment để đọc đủ. "
            "<i class='dot pos'></i>khen · <i class='dot non_pos'></i>không khen · <span class='muted'>·</span> không nhắc. "
            "Số trong ô = điểm khen − không khen; ⚠ = model phân vân. Cột Điểm = trung bình các aspect được nhắc.</p>"
            f"<div class='ct-wrap'><table class='ct'>{head}{''.join(rows)}</table></div>")


def file_chart(P: pd.DataFrame) -> str:
    """Bar diverging theo số comment: không khen bên trái, khen bên phải; cột phải là % khen trong số có nhắc."""
    pos, neg = (P == "pos").sum(), (P == "non_pos").sum()
    top = max(int(pos.max()), int(neg.max()), 1)
    step = 5 if top <= 25 else 10 if top <= 100 else 50
    scale = -(-top // (2 * step)) * 2 * step  # làm tròn lên bội số của 2*step để vạch giữa là số nguyên
    rows = []
    for a in ASPECTS:
        p, n = int(pos[a]), int(neg[a])
        if p + n == 0:
            rows.append(f"<div class='ac-grid ac-row none' tabindex='0'><div class='ac-name'>{a}</div>"
                        "<div class='ac-track'><span class='ac-none'>không ai nhắc</span></div><div class='ac-val'>-</div>"
                        f"<div class='ac-tip'><b>{a}</b> · không comment nào nhắc tới</div></div>")
            continue
        wp, wn = p / scale * 50, n / scale * 50
        bars = ""
        if p:
            bars += (f"<div class='ac-bar pos' style='width:{wp:.2f}%'></div>"
                     f"<span class='ac-count' style='left:calc(50% + {wp:.2f}% + 4px)'>{p}</span>")
        if n:
            bars += (f"<div class='ac-bar non_pos' style='width:{wn:.2f}%'></div>"
                     f"<span class='ac-count' style='right:calc(50% + {wn:.2f}% + 4px)'>{n}</span>")
        share = p / (p + n)
        tip = f"<b>{a}</b> · nhắc trong {p + n} comment<br>Khen: {p} · không khen: {n}<br>Tỉ lệ khen: {share:.0%}"
        small = " ⚠" if p + n < 10 else ""
        rows.append(f"<div class='ac-grid ac-row' tabindex='0'><div class='ac-name'>{a}</div>"
                    f"<div class='ac-track'>{bars}</div><div class='ac-val'>{share:.0%}{small}</div>"
                    f"<div class='ac-tip'>{tip}{'<br>Dưới 10 lượt nhắc: tỉ lệ chưa đáng tin' if small else ''}</div></div>")
    ticks = "".join(f"<span style='left:{(t / scale + 1) * 50:g}%'>{abs(t):g}</span>"
                    for t in (-scale, -scale / 2, 0, scale / 2, scale))
    label = "; ".join(f"{a} khen {int(pos[a])}, không khen {int(neg[a])}" for a in ASPECTS)
    return (f"<div class='absa-chart' role='img' aria-label='Số comment khen và không khen theo aspect: {html.escape(label)}'>"
            "<div class='ac-head'><div><div class='ac-title'>Số comment khen / không khen theo aspect</div>"
            "<div class='ac-sub'>Cột phải: % khen trong số comment có nhắc · ⚠ dưới 10 lượt nhắc · rê chuột để xem chi tiết</div></div>"
            "<div class='ac-legend'><span><i class='pos'></i>khen</span><span><i class='non_pos'></i>không khen</span></div></div>"
            + "".join(rows)
            + f"<div class='ac-grid ac-axis'><div></div><div><div class='ac-axis-track'>{ticks}</div>"
            "<div class='ac-axis-names'><span>← không khen</span><span>khen →</span></div></div><div></div></div></div>")


def complaints(texts: pd.Series, P: pd.DataFrame, S: pd.DataFrame, k: int = 5) -> str:
    """Mỗi aspect: tối đa k comment bị đoán 'không khen', xếp theo điểm khen − không khen thấp nhất (model chắc nhất)."""
    parts = []
    for a in ASPECTS:
        idx = S[a][P[a] == "non_pos"].sort_values().index
        if len(idx) == 0:
            continue
        items = "".join(f"<li><span class='ac-score'>{S.at[i, a]:+.2f}</span> {html.escape(texts[i])}</li>" for i in idx[:k])
        more = f" (hiện {k} câu đầu)" if len(idx) > k else ""
        parts.append(f"<details class='ac-complaint'><summary><b>{a}</b>: {len(idx)} comment{more}</summary><ul>{items}</ul></details>")
    if not parts:
        return "<p class='absa-meta'>Model không thấy comment nào chê.</p>"
    return ("<p class='absa-meta'>Xếp theo điểm khen − không khen thấp nhất trước. Đây là dự đoán của model, có thể sai "
            "(model hay bỏ sót lời chê nhẹ nằm trong câu khen).</p>" + "".join(parts))


def build() -> gr.Blocks:
    with gr.Blocks(title="ABSA bàn phím cơ", analytics_enabled=False) as demo:
        gr.Markdown(
            "# ABSA review bàn phím cơ\n"
            f"Model `{META['experiment']}` ({META['params']}), CV Macro-F1 {META['cv_mean_macro_f1']:.3f}, "
            f"train {META['train_rows']} comment ngày {META['trained_on']}.  \n"
            "Lớp: **khen** (điểm 7–10) · **không khen** (1–6) · **-** (không nhắc). "
            "**Margin**: điểm lớp được chọn trừ lớp đứng thứ hai, gần 0 là model phân vân. "
            "**Từ ảnh hưởng**: chỉ tính phần word TF-IDF.")
        with gr.Tab("Một comment"):
            with gr.Row():
                with gr.Column(scale=2):
                    box = gr.Textbox(label="Comment", lines=4, placeholder="ví dụ: gõ êm, led đẹp nhưng giao hơi chậm")
                    btn = gr.Button("Dự đoán", variant="primary")
                    gr.Examples(EXAMPLES, inputs=box, label="Ví dụ (tự viết)")
                with gr.Column(scale=3):
                    result = gr.HTML()
                    with gr.Accordion("Bảng chi tiết", open=False):
                        details = gr.HTML()
            btn.click(analyze, box, [result, details])
            box.submit(analyze, box, [result, details])
        with gr.Tab("File CSV"):
            with gr.Row():
                file = gr.File(label="File CSV (UTF-8)", file_types=[".csv"])
                with gr.Column():
                    # allow_custom_value: gõ tên cột được cả khi danh sách cột chưa kịp cập nhật
                    col = gr.Dropdown(label="Cột comment", choices=[], allow_custom_value=True)
                    group = gr.Dropdown(label="Nhóm theo (tuỳ chọn)", choices=[], allow_custom_value=True)
                    run = gr.Button("Dự đoán cả file", variant="primary")
            overview_box = gr.HTML()
            with gr.Row():
                with gr.Column(scale=3):
                    fchart = gr.HTML()
                with gr.Column(scale=2):
                    gr.Markdown("**Comment bị chê theo aspect**")
                    fcomplaints = gr.HTML()
            gr.Markdown("### Từng comment")
            ctable = gr.HTML()
            with gr.Accordion("Bảng theo nhóm (chỉ có dòng \"Tất cả\" nếu không chọn cột nhóm)", open=False):
                gr.Markdown("**Tỉ lệ khen theo aspect**: % khen trong số comment có nhắc aspect, (số comment có nhắc).")
                rates = gr.Dataframe(interactive=False)
                gr.Markdown("**Được khen / bị chê nhiều nhất** (theo số comment)")
                best = gr.Dataframe(interactive=False)
            download = gr.File(label="Dự đoán từng dòng (CSV)")
            file.change(on_upload, file, [col, group])
            run.click(analyze_file, [file, col, group], [overview_box, fchart, fcomplaints, ctable, rates, best, download])
    return demo


def main():
    ap = argparse.ArgumentParser(description="Giao diện web Gradio cho model ABSA.")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="tạo link công khai tạm thời (ai có link đều dùng được)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    build().launch(server_name="127.0.0.1", server_port=args.port, share=args.share,
                   css=CSS, theme=gr.themes.Soft(), allowed_paths=[str(OUT_DIR)])


if __name__ == "__main__":
    main()
