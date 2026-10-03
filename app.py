"""SME Utility Optimizer - digital-twin prototype (compressors, boilers, electrical, VFD PdM)."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

import os
import json
import joblib
from pathlib import Path


st.set_page_config(page_title="SME Utility Optimizer", page_icon="⚡", layout="wide")

FUELS = {"PNG": (1200, 0.0561), "Coal": (350, 0.0946), "Furnace oil": (1100, 0.0774)}  # ₹/GJ, tCO2/GJ (editable defaults)
CAP, P_LOAD, UNL = 6.5, 40.0, 0.30   # compressor m3/min, kW at full load, unloaded power fraction
BCAP, DH = 2.0, 2.4                  # boiler capacity (tph), MJ per kg steam
ZLIM = 19.1                          # z-score at which VFD health index reaches 20
inr = lambda x: f"₹{x:,.0f}"


# ------------------------------------------------------------------
# AI ENERGY FORECASTING MODEL
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "energy_forecasting_xgboost.pkl"
FEATURE_PATH = BASE_DIR / "model_features.json"


@st.cache_resource
def load_energy_model():

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)

    return model, features


energy_model, energy_features = load_energy_model()


def flag(ok, msg, fail=st.error):
    """Show a success/failure banner. Returns None so Streamlit magic does not print an object."""
    (st.success if ok else fail)(msg)
    return None

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.title("⚡ Plant settings")
tariff = sb.number_input("Electricity tariff (₹/kWh)", 4.0, 15.0, 8.0, 0.5)
kva_chg = sb.number_input("kVA demand charge (₹/kVA/month)", 0, 1000, 450, 25)
ef_grid = sb.number_input("Grid factor (tCO₂/MWh) - check latest CEA", 0.5, 1.0, 0.71, 0.01)
fuel = sb.selectbox("Boiler fuel", list(FUELS))
days = sb.slider("Operating days / year", 200, 360, 300)
capex = sb.number_input("Installed cost (₹)", 50000, 2000000, 500000, 10000)
dt_cost = sb.number_input("Downtime cost (₹/h)", 0, 200000, 20000, 1000)
sb.subheader("Levers")
band = sb.slider("Compressor set-point reduction (bar)", 0.0, 1.5, 0.7, 0.1)
leak_cut = sb.slider("Air-leak reduction (%)", 0, 80, 30, 5)
rot_h = sb.slider("Lead-rotation interval (h, 0 = fixed lead)", 0, 24, 4)
o2_o = sb.slider("Boiler O₂ target (%)", 2.0, 5.0, 3.5, 0.5)
day = sb.slider("Replay day (VFD / busbar history)", 20, 60, 60)
fuel_cost, fuel_ef = FUELS[fuel]


# ------------------------------------------------------------------ simulators
@st.cache_data
def sim_comp(band, leak_cut, rot_h, ndays=7):
    """3 x 37 kW screw compressors. Baseline: 3 on in shift, 1 off-shift, load/unload sharing, fixed lead.
    Optimised: minimum machines, base-load at 100 %, VFD trim, hours-based lead rotation, lower set-point."""
    r = np.random.default_rng(1)
    h = np.arange(1440) / 60
    shift = (h >= 6) & (h < 22)
    prod = np.where(shift, 11 + 3 * np.sin((h - 6) * np.pi / 8) + r.normal(0, .8, 1440), 0)
    prod = np.convolve(prod, np.ones(15) / 15, "same").clip(0)
    db, do_ = prod + 1.6, prod + 1.6 * (1 - leak_cut / 100)      # 1.6 m3/min baseline leakage
    hb, ho = np.zeros(3), np.zeros(3)
    pb, po, act = np.zeros(1440), np.zeros(1440), np.zeros((1440, 3))
    n2, prev, order, starts = 1, set(), [0, 1, 2], 0
    for d in range(ndays):
        for m in range(1440):
            n = 3 if shift[m] else 1
            f = min(db[m] / (n * CAP), 1)
            p_b = n * P_LOAD * (f + UNL * (1 - f))
            hb[:n] += 1 / 60
            while n2 < 3 and do_[m] > n2 * CAP: n2 += 1
            while n2 > 1 and do_[m] < (n2 - 1) * CAP - 0.8: n2 -= 1   # hysteresis
            if rot_h and (d * 1440 + m) % (rot_h * 60) == 0:
                order = list(np.argsort(ho, kind="stable"))          # lowest run-hours leads
            active = set(order[:n2]); ns = len(active - prev); prev = active
            rem = min(max(do_[m] - (n2 - 1) * CAP, 0), CAP)
            p_o = ((n2 - 1) * P_LOAD + P_LOAD * (0.12 + 0.88 * rem / CAP)) * (1 - 0.065 * band)
            for k in active: ho[k] += 1 / 60
            if d == ndays - 1:
                pb[m], po[m] = p_b, p_o + ns * 0.5 * 60             # 0.5 kWh penalty per start
                starts += ns
                act[m, list(active)] = 1
    return dict(pb=pb, po=po, act=act, hb=hb, ho=ho, starts=starts, prod=prod.sum(), peak=do_.max())


def eff_fn(load, tst, o2):
    """Indirect-method efficiency (%), simplified: stack loss + fixed losses + radiation (rises at low load)."""
    ea = o2 / (21 - o2) * 100
    return 100 - (tst - 30) * (0.05 + 0.0004 * ea) - 4.5 - 0.6 / np.maximum(load, 0.1)


@st.cache_data
def sim_boil(o2_o):
    r = np.random.default_rng(2)
    m = np.arange(1440); h = m / 60
    steam = (np.where((h >= 6) & (h < 22), 1.84 + 0.56 * np.sin((h - 6) * np.pi / 8), 0.32) + r.normal(0, .03, 1440)).clip(0.12)
    qs = steam * 1000 * DH / 60                                    # useful MJ/min
    qb = qs / (eff_fn(steam / 2 / BCAP, 200 + 30 * m / 1440, 5.5) / 100)   # equal split, soot fouling, O2 5.5 %
    a = np.minimum(steam, BCAP); b = steam - a                     # fill boiler A first
    qa = a * 1000 * DH / 60 / (eff_fn(a / BCAP, 195, o2_o) / 100)
    qbo = np.where(b > 0, b * 1000 * DH / 60 / (eff_fn(b / BCAP, 195, o2_o) / 100), 0)
    qo = qa + qbo
    return dict(gb=qb.sum() / 1000, go=qo.sum() / 1000, tons=steam.sum() / 60, peak=steam.max(),
                eff=pd.DataFrame({"Baseline": qs / qb * 100, "Optimised": qs / qo * 100},
                                 index=pd.date_range("2026-01-01", periods=1440, freq="min")))


FEEDERS = {"Lighting & HVAC": ("R", 45), "Welding": ("R", 80), "Furnace fan": ("R", 60),
           "Control panel": ("Y", 30), "Chiller aux": ("Y", 25), "Packing line": ("B", 40), "Tool room": ("B", 35)}


def phase_balance():
    def cur(assign):
        i = {"R": 150., "Y": 150., "B": 150.}
        for p, a in assign.values(): i[p] += a
        return i
    unb = lambda i: max(abs(v - np.mean(list(i.values()))) for v in i.values()) / np.mean(list(i.values())) * 100
    before = cur(FEEDERS)
    new, load = {}, {"R": 150., "Y": 150., "B": 150.}
    for n, (_, a) in sorted(FEEDERS.items(), key=lambda x: -x[1][1]):   # greedy longest-first
        p = min(load, key=load.get); load[p] += a; new[n] = (p, a)
    after = cur(new)
    moves = [(n, FEEDERS[n][0], new[n][0], FEEDERS[n][1]) for n in FEEDERS if FEEDERS[n][0] != new[n][0]]
    avg = np.mean(list(after.values())); p_kw = 1.732 * 0.415 * avg * 0.86
    lossf = lambda i: 0.025 * p_kw * sum(v ** 2 for v in i.values()) / (3 * avg ** 2)
    return dict(before=before, after=after, ub=unb(before), ua=unb(after), moves=moves,
                dkw=lossf(before) - lossf(after), p_kw=p_kw)


@st.cache_data
def sim_busbar():
    r = np.random.default_rng(3); d = np.arange(1, 61)
    t = pd.DataFrame({p: 55 + r.normal(0, 1.2, 60) for p in "RYB"}, index=d)
    t["Y"] += 0.015 * np.clip(d - 15, 0, None) ** 2                # loosening joint on Y phase
    t["ΔT Y vs R,B"] = t["Y"] - t[["R", "B"]].mean(axis=1)
    return t


@st.cache_data
def sim_vfd():
    r = np.random.default_rng(7); rows = []
    for v in range(1, 5):
        for d in range(1, 61):
            load = r.uniform(55, 85)
            rip, hs, cur = 2.5 + .03 * load + r.normal(0, .12), 35 + .35 * load + r.normal(0, .8), r.normal(0, .6)
            if v == 3 and d > 25: rip += .12 * (d - 25)            # DC-bus capacitor ageing
            if v == 2 and d > 32: hs += .5 * (d - 32)              # clogged heatsink / fan
            if v == 4 and d > 40: cur += .15 * (d - 40)            # drifting mechanical load
            rows.append((f"VFD-{v}", d, load, rip, hs, cur))
    df = pd.DataFrame(rows, columns=["vfd", "day", "load", "ripple", "heatsink", "cur_res"])
    tr = df[df.day <= 20]                                          # healthy training window
    pr, ph = np.polyfit(tr.load, tr.ripple, 1), np.polyfit(tr.load, tr.heatsink, 1)
    df["rip_res"] = df.ripple - np.polyval(pr, df.load)
    df["hs_res"] = df.heatsink - np.polyval(ph, df.load)
    X = df[["rip_res", "hs_res", "cur_res"]]
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(X[df.day <= 20])
    df["anomaly"] = iso.predict(X) == -1
    sd = X[df.day <= 20].std()
    df["zr"], df["zh"], df["zc"] = (df.rip_res.abs() / sd.rip_res, df.hs_res.abs() / sd.hs_res, df.cur_res.abs() / sd.cur_res)
    df["health"] = (100 * np.exp(-np.clip(df[["zr", "zh", "zc"]].max(axis=1) - 3, 0, None) / 10)).round(0)
    return df

# ------------------------------------------------------------------
# AI ENERGY INTELLIGENCE
# ------------------------------------------------------------------

def prepare_energy_data(df):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"],dayfirst=True)

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["day_of_week_num"] = (
        df["date"].dt.dayofweek
    )
    df["day_of_month"] = (
        df["date"].dt.day
    )
    df["month"] = (
        df["date"].dt.month
    )
    df["is_weekend"] = (
        df["day_of_week_num"] >= 5
    ).astype(int)

    df["quarter_hour"] = (
        df["hour"] * 4 +
        df["minute"] // 15
    )
    df["week_status_encoded"] = (
        df["WeekStatus"]
        .map({
            "Weekday": 0,
            "Weekend": 1
        })
    )

    # --------------------------------------------------------------
    # Load type
    # --------------------------------------------------------------

    df["load_type_encoded"] = (
        df["Load_Type"]
        .map({
            "Light_Load": 0,
            "Medium_Load": 1,
            "Maximum_Load": 2
        })
    )

    # --------------------------------------------------------------
    # Historical energy
    # --------------------------------------------------------------

    df["usage_lag_1"] = (
        df["Usage_kWh"].shift(1)
    )

    df["usage_lag_2"] = (
        df["Usage_kWh"].shift(2)
    )

    df["usage_lag_4"] = (
        df["Usage_kWh"].shift(4)
    )

    df["usage_lag_8"] = (
        df["Usage_kWh"].shift(8)
    )

    df["usage_lag_96"] = (
        df["Usage_kWh"].shift(96)
    )

    df["usage_lag_672"] = (
        df["Usage_kWh"].shift(672)
    )

    # --------------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------------

    df["usage_roll_mean_1h"] = (
        df["Usage_kWh"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    df["usage_roll_mean_3h"] = (
        df["Usage_kWh"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    df["usage_roll_mean_24h"] = (
        df["Usage_kWh"]
        .shift(1)
        .rolling(96)
        .mean()
    )

    df["usage_roll_std_1h"] = (
        df["Usage_kWh"]
        .shift(1)
        .rolling(4)
        .std()
    )

    df["usage_roll_std_24h"] = (
        df["Usage_kWh"]
        .shift(1)
        .rolling(96)
        .std()
    )

    # --------------------------------------------------------------
    # Keep only rows where all model features exist
    # --------------------------------------------------------------

    df = df.dropna(
        subset=energy_features
    ).reset_index(drop=True)

    return df


def analyze_energy(df):

    data = prepare_energy_data(df)

    # AI predicted/expected energy
    data["AI_Expected_kWh"] = (
        energy_model.predict(
            data[energy_features]
        )
    )

    # Difference between actual and expected
    data["Excess_kWh"] = (
        data["Usage_kWh"] -
        data["AI_Expected_kWh"]
    )

    # Percentage deviation
    data["Deviation_%"] = (
        data["Excess_kWh"] /
        data["AI_Expected_kWh"].replace(0, np.nan)
    ) * 100

    # --------------------------------------------------------------
    # Statistical anomaly score
    # --------------------------------------------------------------

    error_mean = data["Excess_kWh"].mean()

    error_std = data["Excess_kWh"].std()

    if error_std > 0:

        data["Z_score"] = (
            data["Excess_kWh"] -
            error_mean
        ) / error_std

    else:

        data["Z_score"] = 0

    data["Anomaly"] = (
        data["Z_score"] > 3
    )

    # Potential excess energy
    data["Potential_Excess_kWh"] = np.where(
        data["Anomaly"],
        np.maximum(
            data["Excess_kWh"],
            0
        ),
        0
    )

    return data
    



# ------------------------------------------------------------------ run everything
cs, cs0 = sim_comp(band, leak_cut, rot_h), sim_comp(0.0, 0, rot_h)
bo, eb = sim_boil(o2_o), sim_busbar()
pbal, vf = phase_balance(), sim_vfd()
tidx = pd.date_range("2026-01-01", periods=1440, freq="min")

e_base, e_seq, e_opt = cs["pb"].sum() / 60, cs0["po"].sum() / 60, cs["po"].sum() / 60
comp_seq_kwh, comp_extra_kwh = (e_base - e_seq) * days, (e_seq - e_opt) * days
fuel_gj = (bo["gb"] - bo["go"]) * days
bal_kwh = pbal["dkw"] * days * 16
kva_b, kva_a = pbal["p_kw"] / 0.86, pbal["p_kw"] / 0.98
pf_inr = (kva_b - kva_a) * kva_chg * 12
vnow = vf[vf.day <= day]
flagged = vnow[vnow.day == day].query("health < 70").vfd.tolist()
pdm_inr = dt_cost * 12 if flagged else 0

actions = pd.DataFrame([
    ("Compressors", "Base-load + VFD trim, stop surplus machines, rotate lead by run-hours", comp_seq_kwh * tariff, comp_seq_kwh),
    ("Compressors", f"Lower set-point {band} bar + fix {leak_cut}% of leaks", comp_extra_kwh * tariff, comp_extra_kwh),
    ("Boilers", f"Trim O₂ to {o2_o}%, soot-blow, single-boiler dispatch", fuel_gj * fuel_cost, 0),
    ("Electrical", "Re-phase " + ", ".join(m[0] for m in pbal["moves"]), bal_kwh * tariff, bal_kwh),
    ("Electrical", "Fit/tune APFC to reach PF 0.98", pf_inr, 0),
    ("VFD PdM", "Replace ageing/clogged drives before failure: " + (", ".join(flagged) or "none flagged"), pdm_inr, 0),
], columns=["Module", "Action", "₹ / year", "kWh / year"]).sort_values("₹ / year", ascending=False)
if eb["ΔT Y vs R,B"].loc[day] > 12:
    actions.loc[len(actions)] = ["Busbar", "Re-torque / replace Y-phase joint (fire-risk alert)", 0, 0]

tot_inr = actions["₹ / year"].sum()
mwh = (comp_seq_kwh + comp_extra_kwh + bal_kwh) / 1000
tco2 = mwh * ef_grid + fuel_gj * fuel_ef
payback = capex / tot_inr * 12 if tot_inr else float("inf")

# ------------------------------------------------------------------ UI
st.title("⚡ SME Utility Optimizer - digital twin")
st.caption(
    "Digital twin + AI energy intelligence | "
    "Synthetic utility scenarios + real SME steel-industry data"
)
tabs = st.tabs(["Overview", "AI Energy Intelligence", "Compressors", "Boilers", "Electrical", "VFD health", "Architecture & data", "Business case"])

with tabs[0]:
    c = st.columns(4)
    c[0].metric("Annual saving", inr(tot_inr))
    c[1].metric("Electricity saved", f"{mwh:,.1f} MWh")
    c[2].metric("Fuel saved", f"{fuel_gj:,.0f} GJ")
    c[3].metric("CO₂ avoided", f"{tco2:,.1f} tCO₂/yr")
    c = st.columns(2)
    c[0].metric("Payback", f"{payback:.1f} months")
    c[1].metric("Compressor SEC (kWh / 1000 m³)", f"{e_opt / cs['prod'] * 1000:.1f}",
                f"{(e_opt - e_base) / e_base * 100:.1f}% vs {e_base / cs['prod'] * 1000:.1f} baseline", delta_color="inverse")
    st.subheader("Prioritised action list")
    st.dataframe(actions.style.format({"₹ / year": "₹{:,.0f}", "kWh / year": "{:,.0f}"}), hide_index=True, use_container_width=True)
    st.subheader("Quality & throughput guard-rails")
    setpoint = 7.5 - band
    flag(cs["peak"] <= 3 * CAP, f"Air demand served: peak {cs['peak']:.1f} m³/min vs {3 * CAP:.1f} installed")
    flag(bo["peak"] <= 2 * BCAP, f"Steam demand served: peak {bo['peak']:.1f} tph vs {2 * BCAP:.0f} tph installed")
    flag(setpoint >= 6.3, f"New pressure set-point {setpoint:.1f} bar (process minimum 6.0 bar + 0.3 margin)", st.warning)

# ================================================================
# AI ENERGY INTELLIGENCE
# ================================================================

with tabs[1]:
    st.subheader(
        "🤖 AI Energy Intelligence"
    )
    st.caption(
        "Real SME steel-industry data + trained XGBoost energy baseline"
    )
    # ------------------------------------------------------------
    # Upload real plant data
    # ------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload SME plant energy data",
        type=["csv"],
        help="Upload Steel_industry_data.csv or compatible 15-minute plant energy data."
    )
    if uploaded_file is None:
        st.info(
            "Upload Steel_industry_data.csv to activate "
            "the AI Energy Intelligence module."
        )
        st.markdown("""
        ### What this module does
        **1.** Learns the plant's expected energy behavior
        **2.** Predicts expected 15-minute energy consumption
        **3.** Compares actual vs AI baseline
        **4.** Detects statistically unusual energy consumption
        **5.** Uses load type and power-quality measurements
        to help operators investigate the event
        **6.** Estimates potential energy and cost opportunity
        """)
    else:
        try:
            raw_energy_df = pd.read_csv(
                uploaded_file
            )
            required_columns = [
                "date",
                "Usage_kWh",
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
                "Lagging_Current_Power_Factor",
                "Leading_Current_Power_Factor",
                "WeekStatus",
                "Load_Type"
            ]
            missing_columns = [
                col
                for col in required_columns
                if col not in raw_energy_df.columns
            ]
            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )
                st.stop()
            # ----------------------------------------------------
            # Run model
            # ----------------------------------------------------
            with st.spinner(
                "Running AI energy analysis..."
            ):
                energy_results = analyze_energy(
                    raw_energy_df
                )
            # ----------------------------------------------------
            # KPI calculations
            # ----------------------------------------------------
            actual_energy = (
                energy_results["Usage_kWh"].sum()
            )
            expected_energy = (
                energy_results["AI_Expected_kWh"].sum()
            )
            potential_excess = (
                energy_results[
                    "Potential_Excess_kWh"
                ].sum()
            )
            anomaly_count = int(
                energy_results["Anomaly"].sum()
            )
            # ----------------------------------------------------
            # KPI CARDS
            # ----------------------------------------------------
            st.subheader(
                "Energy Performance"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Actual Energy",
                f"{actual_energy:,.0f} kWh"
            )
            c2.metric(
                "AI Baseline",
                f"{expected_energy:,.0f} kWh"
            )
            c3.metric(
                "Potential Excess",
                f"{potential_excess:,.0f} kWh"
            )
            c4.metric(
                "Anomalous Intervals",
                f"{anomaly_count:,}"
            )
            # ----------------------------------------------------
            # ACTUAL VS EXPECTED
            # ----------------------------------------------------
            st.subheader(
                "Actual vs AI Expected Energy"
            )
            chart = (
                energy_results[
                    [
                        "date",
                        "Usage_kWh",
                        "AI_Expected_kWh"
                    ]
                ]
                .set_index("date")
                .rename(columns={
                    "Usage_kWh": "Actual",
                    "AI_Expected_kWh": "AI Expected"
                })
            )
            st.line_chart(
                chart
            )
            st.caption(
                "The AI baseline represents the expected "
                "15-minute energy consumption based on "
                "historical operating patterns."
            )
            # ----------------------------------------------------
            # DEVIATION
            # ----------------------------------------------------
            st.subheader(
                "Energy Deviation from Baseline"
            )
            deviation_chart = (
                energy_results[
                    [
                        "date",
                        "Deviation_%"
                    ]
                ]
                .set_index("date")
            )
            st.line_chart(
                deviation_chart
            )
            # ----------------------------------------------------
            # ANOMALIES
            # ----------------------------------------------------
            st.subheader(
                "⚠️ Highest Energy Anomalies"
            )
            anomalies = (
                energy_results[
                    energy_results["Anomaly"]
                ]
                .sort_values(
                    "Z_score",
                    ascending=False
                )
            )
            if len(anomalies) == 0:
                st.success(
                    "No statistically significant "
                    "high-energy anomalies detected."
                )
            else:
                display_columns = [
                    "date",
                    "Usage_kWh",
                    "AI_Expected_kWh",
                    "Excess_kWh",
                    "Deviation_%",
                    "Load_Type",
                    "Lagging_Current_Power_Factor",
                    "Lagging_Current_Reactive",
                    "Z_score"
                ]
                st.dataframe(
                    anomalies[
                        display_columns
                    ]
                    .head(20)
                    .style.format({
                        "Usage_kWh": "{:.2f}",
                        "AI_Expected_kWh": "{:.2f}",
                        "Excess_kWh": "{:.2f}",
                        "Deviation_%": "{:.1f}%",
                        "Lagging_Current_Power_Factor": "{:.1f}",
                        "Lagging_Current_Reactive": "{:.2f}",
                        "Z_score": "{:.2f}"
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                # ------------------------------------------------
                # INTERPRETATION
                # ------------------------------------------------
                st.subheader(
                    "🧠 Operator Interpretation"
                )
                worst = anomalies.iloc[0]
                actual = worst["Usage_kWh"]
                expected = worst["AI_Expected_kWh"]
                deviation = worst["Deviation_%"]
                load = worst["Load_Type"]
                pf = worst[
                    "Lagging_Current_Power_Factor"
                ]
                reactive = worst[
                    "Lagging_Current_Reactive.Power_kVarh"
                ]
                st.warning(
                    f"""
                    **High-energy event detected**
                    **Time:** {worst["date"]}
                    **Actual consumption:** {actual:.2f} kWh
                    **AI expected consumption:** {expected:.2f} kWh
                    **Deviation:** +{deviation:.1f}%
                    **Operating regime:** {load}
                    **Lagging power factor:** {pf:.1f}
                    **Lagging reactive power:** {reactive:.2f}
                    ### Suggested investigation
                    • Check which equipment/processes were operating
                    during this interval.
                    • Verify whether the high consumption was caused
                    by a legitimate production requirement.
                    • If low power factor coincides with the event,
                    investigate inductive loads, motor loading and
                    power-factor compensation.
                    • Check whether flexible loads can be shifted
                    away from high-demand periods.
                    **This is an investigation opportunity, not proof
                    that all excess energy is avoidable.**
                    """
                )
            # ----------------------------------------------------
            # POWER QUALITY
            # ----------------------------------------------------
            st.subheader(
                "Electrical Context"
            )
            q1, q2, q3 = st.columns(3)
            q1.metric(
                "Average Lagging PF",
                f"{energy_results['Lagging_Current_Power_Factor'].mean():.1f}"
            )
            q2.metric(
                "Average Reactive Power",
                f"{energy_results['Lagging_Current_Reactive.Power_kVarh'].mean():.2f}"
            )
            q3.metric(
                "Maximum Load Intervals",
                f"{(energy_results['Load_Type'] == 'Maximum_Load').sum():,}"
            )
            st.caption(
                "Power-factor and reactive-power values are "
                "used as diagnostic context. They should not "
                "be interpreted as proof of avoidable process energy."
            )
            # ----------------------------------------------------
            # POTENTIAL COST IMPACT
            # ----------------------------------------------------
            potential_cost = (
                potential_excess *
                tariff
            )
            st.subheader(
                "💰 Potential Energy Opportunity"
            )
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Potential excess energy",
                f"{potential_excess:,.1f} kWh"
            )
            c2.metric(
                "Electricity tariff",
                f"₹{tariff:.2f}/kWh"
            )
            c3.metric(
                "Potential cost impact",
                inr(potential_cost)
            )
            st.caption(
                "Potential opportunity = energy above the "
                "statistical AI baseline during flagged intervals. "
                "It is not guaranteed savings."
            )
        except Exception as e:
            st.error(f"AI Energy analysis failed: {e}")


with tabs[2]:
    st.subheader("Load / unload waste → sequencing, trim and run-hour equalisation")
    c = st.columns(3)
    c[0].metric("Energy / day", f"{e_opt:,.0f} kWh", f"{e_opt - e_base:,.0f} kWh", delta_color="inverse")
    c[1].metric("Run-hour gap after 7 days", f"{cs['ho'].max() - cs['ho'].min():.0f} h", f"{(cs['ho'].max() - cs['ho'].min()) - (cs['hb'].max() - cs['hb'].min()):.0f} h vs fixed baseline", delta_color="inverse")
    c[2].metric("Starts / day (rotation cost)", cs["starts"])
    st.line_chart(pd.DataFrame({"Baseline kW": cs["pb"], "Optimised kW": cs["po"]}, index=tidx))
    st.caption("Machines running (optimised)")
    st.area_chart(pd.DataFrame(cs["act"], index=tidx, columns=["C1", "C2", "C3"]))
    st.caption("Cumulative run-hours after 7 days")
    st.bar_chart(pd.DataFrame({"Baseline (fixed lead)": cs["hb"], "Optimised (rotation)": cs["ho"]}, index=["C1", "C2", "C3"]))
    st.info("Trade-off: short rotation intervals equalise hours but add starts (each costs energy). Move the rotation slider to see both effects.")

with tabs[3]:
    st.subheader("Indirect-method efficiency and load-sharing")
    c = st.columns(3)
    c[0].metric("Fuel / day", f"{bo['go']:,.0f} GJ", f"{bo['go'] - bo['gb']:,.0f} GJ", delta_color="inverse")
    c[1].metric("Fuel per tonne steam", f"{bo['go'] / bo['tons']:.2f} GJ/t", f"{(bo['go'] - bo['gb']) / bo['gb'] * 100:.1f}%", delta_color="inverse")
    c[2].metric("Mean efficiency", f"{bo['eff']['Optimised'].mean():.1f}%", f"{bo['eff']['Optimised'].mean() - bo['eff']['Baseline'].mean():+.1f} pts")
    st.line_chart(bo["eff"])
    st.warning("Baseline alert: stack temperature rising ~30 °C over the day (soot/scaling) and O₂ at 5.5 % (excess air ≈ 35 %). Both boilers share load at ~50 % where radiation loss is high.")

with tabs[4]:
    st.subheader("Phase balance, power factor and busbar joints")
    c = st.columns(3)
    c[0].metric("Current unbalance", f"{pbal['ua']:.1f}%", f"{pbal['ua'] - pbal['ub']:.1f} pts", delta_color="inverse")
    c[1].metric("Distribution loss saved", f"{pbal['dkw']:.2f} kW")
    c[2].metric("kVA demand", f"{kva_a:.0f} kVA", f"{kva_a - kva_b:.0f} kVA", delta_color="inverse")
    st.bar_chart(pd.DataFrame({"Before": pbal["before"], "After": pbal["after"]}))
    st.write("**Feeder moves:** " + "; ".join(f"{n}: {a}→{b} ({i} A)" for n, a, b, i in pbal["moves"]))
    st.line_chart(eb.loc[:day])
    dT = eb["ΔT Y vs R,B"].loc[day]
    flag(dT <= 12, f"Y-phase joint is {dT:.1f} °C above the other phases" + (" - loose-joint alert" if dT > 12 else ""))

with tabs[5]:
    st.subheader(f"Isolation-forest anomaly + health index (day {day})")
    ZN = {"zr": "DC-bus capacitor ageing", "zh": "Heatsink / fan clogging", "zc": "Mechanical load / bearing drift"}
    rows = []
    for name, g in vnow.groupby("vfd"):
        zs = g.tail(3)[["zr", "zh", "zc"]].mean(); drv = zs.idxmax(); zn = zs.max()
        slope = np.polyfit(g.tail(10).day, g.tail(10)[drv], 1)[0]
        rul = "Act now" if zn >= ZLIM else (f"{(ZLIM - zn) / slope:.0f} days" if slope > .05 else "Stable")
        rows.append((name, g.health.iloc[-1], bool(g.anomaly.tail(3).all()), ZN[drv] if zn > 3 else "Normal", rul))
    st.dataframe(pd.DataFrame(rows, columns=["Drive", "Health (0-100)", "Alert (3-day anomaly)", "Likely cause", "Est. remaining life"]),
                 hide_index=True, use_container_width=True)
    sel = st.selectbox("Drive", sorted(vf.vfd.unique()), index=2)
    st.line_chart(vnow[vnow.vfd == sel].set_index("day")[["rip_res", "hs_res", "cur_res"]])
    st.caption("Residuals vs load-normalised healthy behaviour (first 20 days): DC-bus ripple %, heatsink °C, current A. Data comes from standard VFD Modbus registers - no new sensors.")

with tabs[6]:
    st.subheader("System architecture")
    st.graphviz_chart("""
digraph G {

    rankdir=LR;

    node [
        shape=box,
        style="rounded,filled",
        fillcolor="#eef3fb",
        fontname="Helvetica"
    ];

    subgraph cluster_f {
        label="Field layer";

        "CT clamps / 3-ph meters";
        "Air pressure + flow";
        "Flue-gas O2 + stack temp";
        "NTC/IR busbar joints";
        "VFD Modbus registers";
    }

    subgraph cluster_e {
        label="Edge gateway (RPi / ESP32 + Node-RED)";

        "Modbus / MQTT collector";
        "Local buffer + safety rules";
    }

    subgraph cluster_c {
        label="Analytics (on-prem or cloud)";

        "MQTT broker";
        "TimescaleDB";

        "M1 Compressor optimiser";
        "M2 Boiler efficiency";
        "M3 Phase / busbar / PF";
        "M4 VFD predictive maintenance";

        "M5 AI Energy Baseline";
        "Energy anomaly detection";
        "Operator recommendations";
    }

    subgraph cluster_a {
        label="Applications";

        "Streamlit dashboard";
        "ERP: Tally / SAP B1 (CSV, REST)";
        "GHG Protocol carbon report";
        "WhatsApp / SMS alerts";
    }

    "CT clamps / 3-ph meters"
        -> "Modbus / MQTT collector";

    "Air pressure + flow"
        -> "Modbus / MQTT collector";

    "Flue-gas O2 + stack temp"
        -> "Modbus / MQTT collector";

    "NTC/IR busbar joints"
        -> "Modbus / MQTT collector";

    "VFD Modbus registers"
        -> "Modbus / MQTT collector";

    "Modbus / MQTT collector"
        -> "Local buffer + safety rules"
        -> "MQTT broker"
        -> "TimescaleDB";


    "TimescaleDB"
        -> "M1 Compressor optimiser";

    "TimescaleDB"
        -> "M2 Boiler efficiency";

    "TimescaleDB"
        -> "M3 Phase / busbar / PF";

    "TimescaleDB"
        -> "M4 VFD predictive maintenance";

    "TimescaleDB"
        -> "M5 AI Energy Baseline";


    "M1 Compressor optimiser"
        -> "Streamlit dashboard";

    "M2 Boiler efficiency"
        -> "Streamlit dashboard";

    "M3 Phase / busbar / PF"
        -> "Streamlit dashboard";

    "M4 VFD predictive maintenance"
        -> "Streamlit dashboard";


    "M5 AI Energy Baseline"
        -> "Energy anomaly detection";

    "Energy anomaly detection"
        -> "Operator recommendations";

    "Operator recommendations"
        -> "Streamlit dashboard";


    "Streamlit dashboard"
        -> "WhatsApp / SMS alerts";

    "TimescaleDB"
        -> "ERP: Tally / SAP B1 (CSV, REST)";

    "TimescaleDB"
        -> "GHG Protocol carbon report";


    "M1 Compressor optimiser"
        -> "Local buffer + safety rules"
        [
            style=dashed,
            label="set-point advice (operator-approved)"
        ];
}
""")
    
    st.subheader("Data model")
    st.code("""CREATE TABLE asset   (asset_id serial PRIMARY KEY, plant_id int, type text, tag text, rated_kw real);
CREATE TABLE reading (ts timestamptz, asset_id int, metric text, value double precision);  -- hypertable on ts
CREATE TABLE event   (ts timestamptz, asset_id int, severity text, code text, action text, est_saving_inr real);
CREATE TABLE emission(month date, plant_id int, scope int, source text, qty real, unit text, tco2 real);""", language="sql")

with tabs[7]:
    st.subheader("Business case (from the sliders)")
    c = st.columns(3)
    c[0].metric("Installed cost", inr(capex)); c[1].metric("Annual saving", inr(tot_inr)); c[2].metric("Payback", f"{payback:.1f} months")
    fee = st.number_input("Subscription (₹ / asset / month, 10 assets)", 0, 5000, 1000, 100)
    st.write(f"With a subscription model the plant pays ₹0 upfront and keeps **{inr(tot_inr - fee * 10 * 12)}/yr** net; vendor recurring revenue is {inr(fee * 120)}/yr per plant.")
    st.markdown("""
**Segments:** foundries, forging, textiles, plastics, food processing (Pune, Coimbatore, Rajkot, Ludhiana, Ahmedabad).  
**Scale-up:** pilot 10 plants via an industry association → ESCO shared-savings + discom/BEE/MSME scheme tie-ins → multi-plant benchmarking and buyer-facing carbon reports.  
**Caveat:** all data here is simulated; validate constants (SEC curves, tariffs, emission factors, sensor costs) against real plant measurements before quoting figures.""")
