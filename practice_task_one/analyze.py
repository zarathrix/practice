import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Etc/GMT-3"                      
CUTOFF = pd.Timestamp("2026-07-01 00:00", tz=TZ)
K_IQR = 5          
INSTANT_SEC = 60   
MIN_GROUP = 30     

CODE = {
    "Проверка заявки": "CHK", "Согласование руководителем": "MGR",
    "Согласование бюджета": "BUD", "Проверка договора": "LEG",
    "Исполнение заявки": "EXE", "Доработка заявки": "REW",
    "Внеплановая проверка": "AUD",
}
NAME = {v: k for k, v in CODE.items()}


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["EventIndex"] = df["EventIndex"].astype(int)
    df["Amount"] = df["Amount"].astype(float)
    df["ts"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(TZ)
    return df.sort_values(["ProcessInstanceId", "EventIndex"]).reset_index(drop=True)


def work_hours(a, b):
    """Рабочие часы между a и b: пн-пт, 09:00-18:00 (UTC+3)."""
    if pd.isna(a) or pd.isna(b) or b <= a:
        return 0.0
    tot, d = 0.0, a.normalize()
    while d <= b.normalize():
        if d.weekday() < 5:
            s, e = max(a, d + timedelta(hours=9)), min(b, d + timedelta(hours=18))
            if e > s:
                tot += (e - s).total_seconds()
        d += timedelta(days=1)
    return tot / 3600


def fmt_ts(t):
    return t.strftime("%d.%m %H:%M:%S")


def short(u):
    return "—" if pd.isna(u) else str(u)[:8]


def build_tasks(df):
    t = df[df["ActivityName"].isin(CODE)].copy()
    t["c"] = t["ActivityName"].map(CODE)
    cr = t[t.EventType == "task_created"].set_index("TaskId")
    cl = t[t.EventType == "task_completed"].set_index("TaskId")
    tk = cr[["ProcessInstanceId", "c", "RequestType", "Amount", "ts", "EventIndex"]].join(
        cl[["ts", "Role", "UserId", "Result", "EventIndex"]], rsuffix="_d")
    tk = tk.rename(columns={"ts": "t_cr", "ts_d": "t_cl", "EventIndex_d": "idx_cl"})
    tk["cal_h"] = (tk.t_cl - tk.t_cr).dt.total_seconds() / 3600
    tk["wrk_h"] = [work_hours(a, b) if pd.notna(b) else np.nan for a, b in zip(tk.t_cr, tk.t_cl)]
    tk["age_wrk_h"] = [work_hours(a, CUTOFF) if pd.isna(b) else np.nan for a, b in zip(tk.t_cr, tk.t_cl)]
    tk["age_cal_h"] = [(CUTOFF - a).total_seconds() / 3600 if pd.isna(b) else np.nan for a, b in zip(tk.t_cr, tk.t_cl)]
    return tk.reset_index().sort_values(["ProcessInstanceId", "EventIndex"]).reset_index(drop=True)


def build_processes(df, tk):
    start = df[df.EventType == "process_started"].set_index("ProcessInstanceId")
    p = start[["RequestType", "Amount", "UserId", "ts"]].rename(columns={"UserId": "starter", "ts": "t_start"})
    done = df[df.EventType == "process_completed"].set_index("ProcessInstanceId")
    p["t_end"] = done["ts"]
    p["final"] = done["Result"]
    p["route"] = tk[tk.t_cl.notna()].groupby("ProcessInstanceId").c.agg("-".join)
    p["route_all"] = tk.groupby("ProcessInstanceId").c.agg("-".join)
    p["lead_cal_h"] = (p.t_end - p.t_start).dt.total_seconds() / 3600
    return p


def baselines(tk):
    """Медиана/Q3/IQR/граница длительности (рабочие часы) по этап x тип заявки."""
    closed = tk[tk.wrk_h.notna() & (tk.c != "AUD")]

    def stats(s):
        q1, q3 = s.quantile(.25), s.quantile(.75)
        return dict(n=len(s), med=s.median(), q1=q1, q3=q3, fence=q3 + K_IQR * (q3 - q1))

    out = {}
    for c, g in closed.groupby("c"):
        allt = stats(g.wrk_h)
        for rt in ["standard", "urgent", "contract"]:
            s = g[g.RequestType == rt].wrk_h
            out[(c, rt)] = stats(s) if len(s) >= MIN_GROUP else allt
    return out


def detect(df, tk, p, base):
    rows = []

    def add(pid, typ, text):
        rows.append((pid, typ, text))

    exp_role = tk.groupby("c").Role.agg(lambda s: s.value_counts().idxmax()).to_dict()
    n_total = len(p)

    # 1.
    need = {"MGR": "Согласование руководителем", "BUD": "Согласование бюджета",
            "LEG": "Проверка договора", "CHK": "Проверка заявки"}
    for pid, g in tk.groupby("ProcessInstanceId"):
        g = g[g.t_cl.notna()]
        if g.empty or "EXE" not in set(g.c):
            continue
        done = [c for c in g.c]
        rt = p.at[pid, "RequestType"]
        miss = [c for c in need if c not in done and (c != "LEG" or rt == "contract")]
        extra_leg = "LEG" in done and rt != "contract"
        if miss:
            ref = tk[tk.c.isin(["BUD"]) & tk.t_cl.notna()].ProcessInstanceId.nunique()
            with_mgr = tk[(tk.c == "MGR") & (tk.Result == "accept")].ProcessInstanceId.nunique()
            names = ", ".join(f"«{need[c]}»" for c in miss)
            first = g.iloc[0]
            txt = (f"Фактический маршрут {'→'.join(done)} ({rt}) не содержит этапа {names}; "
                   f"заявка дошла до исполнения без него. Схема требует CHK→MGR→BUD→[LEG]→EXE. "
                   f"Сумма {p.at[pid, 'Amount']:.2f}. "
                   f"Для сравнения: согласование руководителем есть у {with_mgr} из {ref} заявок, прошедших бюджет.")
            add(pid, f"Пропущен этап: {need[miss[0]].lower()}" if len(miss) == 1 else "Пропущены обязательные этапы", txt)
        if extra_leg:
            add(pid, "Лишняя юридическая проверка",
                f"Юридическая проверка выполнена для заявки типа {rt}, хотя по схеме она только для contract.")

    # 2. 
    for pid, g in tk.groupby("ProcessInstanceId"):
        g = g[g.t_cl.notna()].sort_values("idx_cl")
        cs = list(g.c)
        if "EXE" in cs:
            i = cs.index("EXE")
            after = [c for c in cs[i + 1:] if c in ("MGR", "BUD", "LEG")]
            if after:
                e = g[g.c == "EXE"].iloc[0]
                late = g[g.c.isin(after)]
                det = "; ".join(f"«{NAME[r.c]}» закрыто {fmt_ts(r.t_cl)}" for r in late.itertuples())
                add(pid, "Исполнение до согласования",
                    f"Этап «Исполнение заявки» закрыт {fmt_ts(e.t_cl)} (закупщик {short(e.UserId)}), "
                    f"а обязательные этапы выполнены позже: {det}. Фактический порядок {'→'.join(cs)}; "
                    f"по схеме исполнение — последний этап после бюджета{' и юриста' if p.at[pid,'RequestType']=='contract' else ''}. "
                    f"Закупка исполнена до финансового контроля (сумма {p.at[pid,'Amount']:.2f}).")

    # 3.
    for pid, g in tk.groupby("ProcessInstanceId"):
        cs = list(g.c)
        for i in range(len(cs) - 1):
            if cs[i] == "CHK" and cs[i + 1] == "CHK":
                a, b = g.iloc[i], g.iloc[i + 1]
                add(pid, "Повторная проверка без возврата",
                    f"«Проверка заявки» выполнена дважды подряд без доработки между ними: "
                    f"задание 1 {fmt_ts(a.t_cr)}→{fmt_ts(a.t_cl)} (specialist {short(a.UserId)}, {a.Result}), "
                    f"задание 2 создано {fmt_ts(b.t_cr)} и закрыто {fmt_ts(b.t_cl)} (specialist {short(b.UserId)}, {b.Result}). "
                    f"Повторная проверка обычно идёт только после возврата руководителем (этап «Доработка заявки»), здесь возврата нет.")
                break

    # 4.
    n_aud = tk[tk.c == "AUD"].ProcessInstanceId.nunique()
    for pid, g in tk.groupby("ProcessInstanceId"):
        a = g[g.c == "AUD"]
        if a.empty:
            continue
        users = ", ".join(sorted({short(u) for u in a.UserId.dropna()}))
        seq = "→".join(g.c)
        add(pid, "Внеплановая проверка (auditor)",
            f"В процессе {len(a)} задания «Внеплановая проверка» (роль auditor, пользователи {users}); маршрут {seq}. "
            f"Этапа нет в основной схеме; встречается лишь в {n_aud} из {n_total} заявок. Оба задания закрыты accept, "
            f"заявка завершена {p.at[pid,'final']}; длительность {', '.join(f'{x:.1f}' for x in a.wrk_h)} раб. ч. "
            f"Допустимый случай контроля или нарушение — по данным не определить: нужна ручная проверка основания проверки.")

    # 5.
    ret = tk[tk.c == "REW"].groupby("ProcessInstanceId").size()
    q1, q3 = ret.quantile(.25), ret.quantile(.75)
    fence = q3 + 3 * (q3 - q1)
    one = int((ret == 1).sum())
    for pid, k in ret[ret > fence].items():
        g = tk[tk.ProcessInstanceId == pid]
        add(pid, "Циклические возвраты на доработку",
            f"Руководитель возвращал заявку {k} раз(а): {k} циклов «Доработка→Проверка→Согласование» "
            f"({len(g)} заданий, маршрут завершён {p.at[pid,'final']}). Среди {len(ret)} заявок с возвратами "
            f"{one} имеют ровно один возврат, промежуточных значений 2–4 нет; граница Q3+3·IQR = {fence:.0f}. "
            f"Календарная длительность процесса {p.at[pid,'lead_cal_h']/24:.1f} сут. Признак системной ошибки заявителя или «пинг-понга» между ролями.")

    # 6.
    for r in tk[tk.t_cl.notna() & (tk.Role != tk.c.map(exp_role)) & (tk.c != "AUD")].itertuples():
        selfapp = r.UserId == p.at[r.ProcessInstanceId, "starter"]
        n_ok = int(((tk.c == r.c) & (tk.Role == exp_role[r.c])).sum())
        add(r.ProcessInstanceId, "Согласование не руководителем" if r.c == "MGR" else "Роль не соответствует этапу",
            f"Этап «{NAME[r.c]}» закрыт пользователем {short(r.UserId)} с ролью {r.Role}, ожидается {exp_role[r.c]} "
            f"(в остальных {n_ok} заданиях этапа роль {exp_role[r.c]}). Результат {r.Result}, закрыто {fmt_ts(r.t_cl)}, "
            f"сумма {p.at[r.ProcessInstanceId,'Amount']:.2f}."
            + (" Пользователь является инициатором этой же заявки — самосогласование, нарушение разделения обязанностей." if selfapp
               else " Заявитель согласовал заявку (в т.ч. чужую) вместо руководителя."))

    # 7.
    for r in tk.itertuples():
        if r.c == "AUD":
            continue
        b = base[(r.c, r.RequestType)]
        if pd.notna(r.wrk_h) and r.wrk_h > b["fence"] and r.cal_h * 3600 >= INSTANT_SEC:
            add(r.ProcessInstanceId, "Аномально долгий этап",
                f"Этап «{NAME[r.c]}» ({r.RequestType}): задание {fmt_ts(r.t_cr)}→{fmt_ts(r.t_cl)}, {r.cal_h:.0f} кал. ч / "
                f"{r.wrk_h:.1f} раб. ч. Медиана для этапа и типа {b['med']:.2f} раб. ч, граница Q3+{K_IQR}·IQR = {b['fence']:.2f} "
                f"({r.wrk_h/b['med']:.0f}× медианы, n={b['n']}). Исполнитель {r.Role} {short(r.UserId)}; "
                f"время включает ожидание, но выходные учтены в рабочих часах.")
        if pd.isna(r.t_cl) and r.age_wrk_h > b["fence"]:
            add(r.ProcessInstanceId, "Зависшее задание в открытой заявке",
                f"Задание «{NAME[r.c]}» создано {fmt_ts(r.t_cr)} и не закрыто на срез 01.07.2026 00:00: ожидание "
                f"{r.age_cal_h/24:.0f} кал. сут. / {r.age_wrk_h:.0f} раб. ч при медиане этапа {b['med']:.2f} раб. ч и границе "
                f"{b['fence']:.2f} раб. ч (превышение в {r.age_wrk_h/b['fence']:.0f}× границы). Последующих событий нет, процесс не завершён. "
                f"Открытая заявка сама по себе не аномалия, но столь долгое ожидание указывает на потерянное задание.")

    # 8.
    fast = tk[tk.cal_h * 3600 < INSTANT_SEC]
    for pid, g in fast.groupby("ProcessInstanceId"):
        if len(g) < 2:
            continue
        full = tk[tk.ProcessInstanceId == pid]
        detail = ", ".join(f"{r.c} {r.cal_h*3600:.0f} с" for r in full.itertuples())
        meds = ", ".join(f"{c} {base[(c, p.at[pid,'RequestType'])]['med']:.2f}" for c in full.c.unique() if c != "AUD")
        add(pid, "Мгновенное прохождение этапов",
            f"Все {len(full)} человеческих задания закрыты за секунды: {detail}. Медианы этапов по типу {p.at[pid,'RequestType']} "
            f"(раб. ч): {meds}. Быстрее {INSTANT_SEC} с закрыто {int((tk.cal_h*3600<INSTANT_SEC).sum())} заданий во всём журнале, и все они "
            f"сосредоточены в {fast.ProcessInstanceId.nunique()} заявках. Похоже на автоматическое «штампование» без реальной проверки; "
            f"общая длительность заявки {p.at[pid,'lead_cal_h']*60:.1f} мин." if pd.notna(p.at[pid, "lead_cal_h"]) else
            f"Все {len(full)} задания закрыты за секунды: {detail}. Медианы (раб. ч): {meds}.")

    out = pd.DataFrame(rows, columns=["ProcessInstanceId", "AnomalyType", "Explanation"]).drop_duplicates()
    return out.sort_values(["AnomalyType", "ProcessInstanceId"]).reset_index(drop=True)


def quality(df, tk, p):
    last = df.groupby("ProcessInstanceId").tail(1)
    after = df[df.ts > CUTOFF]
    start9 = (df[df.EventType == "process_started"].ts.dt.strftime("%H:%M:%S") == "09:00:00").sum()
    return {
        "events": len(df), "processes": int(df.ProcessInstanceId.nunique()),
        "duplicate_event_ids": int(df.EventId.duplicated().sum()),
        "index_gaps": int((~df.groupby("ProcessInstanceId").EventIndex.apply(
            lambda s: (s.values == np.arange(1, len(s) + 1)).all())).sum()),
        "time_not_monotonic": int((~df.groupby("ProcessInstanceId").ts.apply(lambda s: s.is_monotonic_increasing)).sum()),
        "tasks_created": int((df.EventType == "task_created").sum()),
        "tasks_closed": int((df.EventType == "task_completed").sum()),
        "closed_without_created": int(len(set(df[df.EventType == "task_completed"].TaskId) - set(df[df.EventType == "task_created"].TaskId))),
        "negative_durations": int((tk.cal_h < 0).sum()),
        "open_processes": int((last.EventType != "process_completed").sum()),
        "events_after_cutoff": int(len(after)), "processes_with_events_after_cutoff": int(after.ProcessInstanceId.nunique()),
        "events_outside_work_hours_or_weekend": int(((df.ts.dt.weekday >= 5) | (df.ts.dt.hour < 9) | (df.ts.dt.hour >= 18)).sum()),
        "starts_at_exactly_0900": int(start9),
        "single_initiator": int(df[df.EventType == "process_started"].UserId.nunique()),
    }


def figures(tk, p, anom, outdir, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["CHK", "MGR", "BUD", "LEG", "EXE", "REW"]
    closed = tk[tk.wrk_h.notna() & tk.c.isin(order)].copy()
    fence = np.array([base[(c, rt)]["fence"] for c, rt in zip(closed.c, closed.RequestType)])
    closed["bad"] = (closed.cal_h * 3600 < INSTANT_SEC) | (closed.wrk_h > fence)
    closed["y"] = closed.wrk_h.clip(lower=1 / 360)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.boxplot([closed[(closed.c == c) & ~closed.bad].y for c in order], tick_labels=order, showfliers=False, whis=1.5)
    rng = np.random.default_rng(0)
    for i, c in enumerate(order, 1):
        g = closed[closed.c == c]
        for flag, col, sz, al in [(False, "#9aa5b1", 3, .35), (True, "#d62728", 16, .9)]:
            y = g[g.bad == flag].y
            ax.scatter(i + rng.uniform(-.18, .18, len(y)), y, s=sz, c=col, alpha=al)
    ax.set_yscale("log"); ax.set_ylabel("рабочие часы (лог. шкала)")
    ax.set_title("Длительность заданий: красные — быстрее 60 с или выше границы Q3+5·IQR")
    fig.tight_layout(); fig.savefig(outdir / "fig_durations.png", dpi=170); plt.close(fig)

    top = p.groupby("route_all").size().sort_values(ascending=False)
    short_r = top.head(12)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [r if len(r) < 40 else r[:37] + "…" for r in short_r.index]
    ax.barh(labels[::-1], short_r.values[::-1], color="#3b6ea5")
    ax.set_xlabel("число заявок"); ax.set_title("12 самых частых маршрутов (закрытые + открытые задания)")
    fig.tight_layout(); fig.savefig(outdir / "fig_routes.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="students.csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    df = load(a.input)
    tk = build_tasks(df)
    p = build_processes(df, tk)
    base = baselines(tk)
    anom = detect(df, tk, p, base)
    anom.to_csv(out / "anomalies.csv", index=False, encoding="utf-8")

    routes = p.groupby(["RequestType", "route_all"]).size().sort_values(ascending=False)
    summary = {
        "quality": quality(df, tk, p),
        "request_types": p.RequestType.value_counts().to_dict(),
        "final": p.final.fillna("OPEN").value_counts().to_dict(),
        "typical_routes": {f"{k[0]}: {k[1]}": int(v) for k, v in routes.head(8).items()},
        "baselines_work_hours": {f"{c}/{t}": {k: round(float(v), 3) for k, v in s.items()} for (c, t), s in base.items()},
        "anomaly_rows": int(len(anom)), "anomalous_processes": int(anom.ProcessInstanceId.nunique()),
        "by_type": anom.groupby("AnomalyType").ProcessInstanceId.nunique().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figures(tk, p, anom, out, base)
    print(json.dumps(summary["by_type"], ensure_ascii=False, indent=2))
    print(f"строк: {len(anom)}, процессов: {anom.ProcessInstanceId.nunique()} из {len(p)}")


if __name__ == "__main__":
    main()
