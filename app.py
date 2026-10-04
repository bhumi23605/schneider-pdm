import json
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import IsolationForest

# Optional packages
try:
    import joblib
except ImportError:
    joblib = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# PAGE CONFIG — ONLY ONCE
# ============================================================

st.set_page_config(
    page_title="CarbonTwin - SME Decarbonization",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

# DEMO VALUES.
# Replace with the emission factors selected for your final
# reporting methodology.

GRID_EF = 0.70          # kg CO2e / kWh
DIESEL_EF = 2.68        # kg CO2e / litre
NATURAL_GAS_EF = 2.00   # kg CO2e / SCM

FUELS = {
    "PNG": (1200, 0.0561),
    "Coal": (350, 0.0946),
    "Furnace oil": (1100, 0.0774)
}

CAP = 6.5
P_LOAD = 40.0
UNL = 0.30

BCAP = 2.0
DH = 2.4

ZLIM = 19.1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def inr(value):
    return f"₹{value:,.0f}"


def flag(ok, msg):
    if ok:
        st.success(msg)
    else:
        st.error(msg)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ SME Configuration")

company_name = st.sidebar.text_input(
    "Company Name",
    "ABC Steel Components"
)

industry = st.sidebar.selectbox(
    "Industry",
    [
        "Steel Manufacturing",
        "Fabrication",
        "Automotive Components",
        "Food Processing",
        "Textile",
        "Other"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("💰 Financial Assumptions")

tariff = st.sidebar.number_input(
    "Electricity tariff (₹/kWh)",
    min_value=1.0,
    max_value=30.0,
    value=8.0,
    step=0.5
)

electricity_ef = st.sidebar.number_input(
    "Grid emission factor (kg CO₂e/kWh)",
    min_value=0.1,
    max_value=2.0,
    value=GRID_EF,
    step=0.01
)

st.sidebar.markdown("---")

st.sidebar.subheader("🔥 Utility Parameters")

fuel = st.sidebar.selectbox(
    "Boiler fuel",
    list(FUELS)
)

operating_days = st.sidebar.slider(
    "Operating days/year",
    200,
    360,
    300
)

capex = st.sidebar.number_input(
    "Potential investment (₹)",
    min_value=50000,
    max_value=5000000,
    value=500000,
    step=10000
)

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload SME energy CSV",
    type=["csv"]
)


# ============================================================
# LOAD DEMO / USER DATA
# ============================================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(f"Could not read CSV: {e}")
        st.stop()

else:

    np.random.seed(42)

    months = pd.date_range(
        start="2026-01-01",
        periods=12,
        freq="MS"
    )

    df = pd.DataFrame({
        "date": months,

        "electricity_kwh":
            np.random.randint(
                115000,
                135000,
                12
            ),

        "diesel_litres":
            np.random.randint(
                7000,
                9000,
                12
            ),

        "natural_gas_scm":
            np.random.randint(
                14000,
                17000,
                12
            ),

        "production_tonnes":
            np.random.randint(
                450,
                550,
                12
            )
    })


# ============================================================
# CARBON DATA VALIDATION
# ============================================================

carbon_columns = [
    "electricity_kwh",
    "diesel_litres",
    "natural_gas_scm",
    "production_tonnes"
]

missing = [
    c for c in carbon_columns
    if c not in df.columns
]

# We don't stop the whole application if the uploaded dataset
# isn't a CarbonTwin dataset.

carbon_data_available = len(missing) == 0


if carbon_data_available:

    for col in carbon_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=carbon_columns
    )


# ============================================================
# CARBON CALCULATION
# ============================================================

if carbon_data_available:

    df["scope1_diesel_kg"] = (
        df["diesel_litres"] *
        DIESEL_EF
    )

    df["scope1_gas_kg"] = (
        df["natural_gas_scm"] *
        NATURAL_GAS_EF
    )

    df["scope2_electricity_kg"] = (
        df["electricity_kwh"] *
        electricity_ef
    )

    df["scope1_kg"] = (
        df["scope1_diesel_kg"] +
        df["scope1_gas_kg"]
    )

    df["scope2_kg"] = (
        df["scope2_electricity_kg"]
    )

    df["total_kg"] = (
        df["scope1_kg"] +
        df["scope2_kg"]
    )

    df["scope1_tco2e"] = (
        df["scope1_kg"] / 1000
    )

    df["scope2_tco2e"] = (
        df["scope2_kg"] / 1000
    )

    df["total_tco2e"] = (
        df["total_kg"] / 1000
    )

    total_scope1 = df["scope1_tco2e"].sum()

    total_scope2 = df["scope2_tco2e"].sum()

    total_emissions = (
        total_scope1 +
        total_scope2
    )

    total_production = (
        df["production_tonnes"].sum()
    )

    emission_intensity = (
        total_emissions /
        total_production
        if total_production > 0
        else 0
    )

    baseline_electricity = (
        df["electricity_kwh"].sum()
    )

    baseline_diesel = (
        df["diesel_litres"].sum()
    )

    baseline_gas = (
        df["natural_gas_scm"].sum()
    )

else:

    total_scope1 = 0
    total_scope2 = 0
    total_emissions = 0
    total_production = 0
    emission_intensity = 0
    baseline_electricity = 0
    baseline_diesel = 0
    baseline_gas = 0


# ============================================================
# HEADER
# ============================================================

st.title("🌱 CarbonTwin")

st.subheader(
    "SME Carbon Digital Twin & GHG Reporting Platform"
)

st.caption(
    "Energy intelligence → carbon accounting → "
    "decarbonization simulation → buyer-ready reporting"
)


# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs([
    "🏭 Carbon Footprint",
    "🔮 Digital Twin",
    "🤖 Energy Intelligence",
    "⚙️ Utility Optimizer",
    "📄 GHG Reporting"
])


# ============================================================
# TAB 1 — CARBON FOOTPRINT
# ============================================================

with tabs[0]:

    st.header("🏭 SME Carbon Footprint")

    if not carbon_data_available:

        st.warning(
            "Uploaded file does not contain the CarbonTwin "
            "columns required for carbon accounting."
        )

        st.code(
            """
electricity_kwh
diesel_litres
natural_gas_scm
production_tonnes
            """
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total GHG",
            f"{total_emissions:,.0f} tCO₂e"
        )

        c2.metric(
            "Scope 1",
            f"{total_scope1:,.0f} tCO₂e"
        )

        c3.metric(
            "Scope 2",
            f"{total_scope2:,.0f} tCO₂e"
        )

        c4.metric(
            "Carbon Intensity",
            f"{emission_intensity:.3f} tCO₂e/t"
        )

        st.subheader("🔥 Emission Hotspots")

        diesel_emissions = (
            df["scope1_diesel_kg"].sum() / 1000
        )

        gas_emissions = (
            df["scope1_gas_kg"].sum() / 1000
        )

        electricity_emissions = (
            df["scope2_electricity_kg"].sum() / 1000
        )

        breakdown = pd.DataFrame({
            "Source": [
                "Grid Electricity",
                "Diesel",
                "Natural Gas"
            ],

            "Emissions": [
                electricity_emissions,
                diesel_emissions,
                gas_emissions
            ]
        })

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                breakdown,
                names="Source",
                values="Emissions",
                title="Carbon Footprint by Source"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                breakdown,
                x="Source",
                y="Emissions",
                title="Emission Hotspots",
                labels={
                    "Emissions": "tCO₂e"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if "date" in df.columns:

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            fig = px.line(
                df,
                x="date",
                y="total_tco2e",
                markers=True,
                title="Monthly Carbon Footprint"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# TAB 2 — DIGITAL TWIN
# ============================================================

with tabs[1]:

    st.header("🔮 Decarbonization Digital Twin")

    if not carbon_data_available:

        st.warning(
            "Carbon data is required for the digital twin."
        )

    else:

        st.write(
            """
            Change the intervention assumptions below and
            simulate what happens to the SME's carbon footprint.
            """
        )

        c1, c2 = st.columns(2)

        with c1:

            solar_percentage = st.slider(
                "☀️ Solar adoption (%)",
                0,
                100,
                20
            )

            efficiency_percentage = st.slider(
                "⚡ Energy efficiency improvement (%)",
                0,
                30,
                10
            )

        with c2:

            fuel_switch_percentage = st.slider(
                "🔋 Diesel → Electric (%)",
                0,
                100,
                0
            )

            renewable_percentage = st.slider(
                "🌱 Renewable electricity contract (%)",
                0,
                100,
                0
            )

        # ----------------------------------------------------
        # Scenario calculation
        # ----------------------------------------------------

        # Efficiency reduces electricity demand first.

        efficient_electricity = (
            baseline_electricity *
            (1 - efficiency_percentage / 100)
        )

        # Solar supplies a percentage of remaining demand.

        solar_energy = (
            efficient_electricity *
            solar_percentage / 100
        )

        grid_after_solar = (
            efficient_electricity -
            solar_energy
        )

        # Renewable electricity contract

        renewable_energy = (
            grid_after_solar *
            renewable_percentage / 100
        )

        grid_energy = (
            grid_after_solar -
            renewable_energy
        )

        # Diesel switching

        diesel_reduced = (
            baseline_diesel *
            fuel_switch_percentage / 100
        )

        scenario_diesel = (
            baseline_diesel -
            diesel_reduced
        )

        scenario_scope1 = (
            scenario_diesel * DIESEL_EF +
            baseline_gas * NATURAL_GAS_EF
        ) / 1000

        scenario_scope2 = (
            grid_energy *
            electricity_ef
        ) / 1000

        scenario_total = (
            scenario_scope1 +
            scenario_scope2
        )

        reduction = (
            (total_emissions - scenario_total)
            / total_emissions * 100
            if total_emissions > 0
            else 0
        )

        electricity_saved = (
            baseline_electricity -
            efficient_electricity
        )

        # Simple financial estimate

        electricity_saving = (
            electricity_saved *
            tariff
        )

        fuel_saving = (
            diesel_reduced *
            90
        )

        annual_saving = (
            electricity_saving +
            fuel_saving
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.subheader("📊 Scenario Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Baseline",
            f"{total_emissions:,.0f} tCO₂e"
        )

        c2.metric(
            "Scenario",
            f"{scenario_total:,.0f} tCO₂e",
            delta=f"-{total_emissions - scenario_total:,.0f}"
        )

        c3.metric(
            "CO₂ Reduction",
            f"{reduction:.1f}%"
        )

        c4.metric(
            "Estimated Saving",
            f"₹{annual_saving / 100000:.2f} L"
        )

        comparison = pd.DataFrame({
            "Scenario": [
                "Baseline",
                "Decarbonized"
            ],

            "Emissions": [
                total_emissions,
                scenario_total
            ]
        })

        fig = px.bar(
            comparison,
            x="Scenario",
            y="Emissions",
            title="Baseline vs Decarbonization Scenario",
            labels={
                "Emissions": "tCO₂e"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "energy_forecasting_xgboost.pkl"
)

FEATURE_PATH = (
    BASE_DIR /
    "model_features.json"
)


@st.cache_resource
def load_energy_model():

    if (
        joblib is None or
        not MODEL_PATH.exists() or
        not FEATURE_PATH.exists()
    ):

        return None, None

    try:

        model = joblib.load(
            MODEL_PATH
        )

        with open(
            FEATURE_PATH,
            "r"
        ) as f:

            features = json.load(f)

        return model, features

    except Exception:

        return None, None


energy_model, energy_features = (
    load_energy_model()
)


# ============================================================
# ENERGY FEATURE ENGINEERING
# ============================================================

def prepare_energy_data(
    data,
    features
):

    data = data.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    data["hour"] = (
        data["date"].dt.hour
    )

    data["minute"] = (
        data["date"].dt.minute
    )

    data["day_of_week_num"] = (
        data["date"].dt.dayofweek
    )

    data["day_of_month"] = (
        data["date"].dt.day
    )

    data["month"] = (
        data["date"].dt.month
    )

    data["is_weekend"] = (
        data["day_of_week_num"] >= 5
    ).astype(int)

    data["quarter_hour"] = (
        data["hour"] * 4 +
        data["minute"] // 15
    )

    data["week_status_encoded"] = (
        data["WeekStatus"]
        .map({
            "Weekday": 0,
            "Weekend": 1
        })
    )

    data["load_type_encoded"] = (
        data["Load_Type"]
        .map({
            "Light_Load": 0,
            "Medium_Load": 1,
            "Maximum_Load": 2
        })
    )

    # Historical features

    for lag in [1, 2, 4, 8, 96, 672]:

        data[
            f"usage_lag_{lag}"
        ] = (
            data["Usage_kWh"]
            .shift(lag)
        )

    # Rolling features

    data["usage_roll_mean_1h"] = (
        data["Usage_kWh"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    data["usage_roll_mean_3h"] = (
        data["Usage_kWh"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    data["usage_roll_mean_24h"] = (
        data["Usage_kWh"]
        .shift(1)
        .rolling(96)
        .mean()
    )

    data["usage_roll_std_1h"] = (
        data["Usage_kWh"]
        .shift(1)
        .rolling(4)
        .std()
    )

    data["usage_roll_std_24h"] = (
        data["Usage_kWh"]
        .shift(1)
        .rolling(96)
        .std()
    )

    available_features = [
        f for f in features
        if f in data.columns
    ]

    if len(available_features) != len(features):

        missing = [
            f for f in features
            if f not in data.columns
        ]

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
        )

    data = data.dropna(
        subset=features
    ).reset_index(drop=True)

    return data


# ============================================================
# ENERGY ANALYSIS
# ============================================================

def analyze_energy(data):

    if energy_model is None:

        raise FileNotFoundError(
            "XGBoost model files not found."
        )

    prepared = prepare_energy_data(
        data,
        energy_features
    )

    prepared[
        "AI_Expected_kWh"
    ] = energy_model.predict(
        prepared[energy_features]
    )

    prepared["Excess_kWh"] = (
        prepared["Usage_kWh"] -
        prepared["AI_Expected_kWh"]
    )

    prepared["Deviation_%"] = (
        prepared["Excess_kWh"] /
        prepared["AI_Expected_kWh"].replace(
            0,
            np.nan
        )
    ) * 100

    mean_error = (
        prepared["Excess_kWh"].mean()
    )

    std_error = (
        prepared["Excess_kWh"].std()
    )

    if std_error > 0:

        prepared["Z_score"] = (
            prepared["Excess_kWh"] -
            mean_error
        ) / std_error

    else:

        prepared["Z_score"] = 0

    prepared["Anomaly"] = (
        prepared["Z_score"] > 3
    )

    prepared[
        "Potential_Excess_kWh"
    ] = np.where(
        prepared["Anomaly"],
        np.maximum(
            prepared["Excess_kWh"],
            0
        ),
        0
    )

    return prepared


# ============================================================
# TAB 3 — ENERGY INTELLIGENCE
# ============================================================

with tabs[2]:

    st.header("🤖 AI Energy Intelligence")

    st.caption(
        "AI baseline + anomaly detection for SME energy consumption"
    )

    energy_file = st.file_uploader(
        "Upload 15-minute plant energy data",
        type=["csv"],
        key="energy_upload"
    )

    if energy_model is None:

        st.warning(
            "XGBoost model not found. "
            "Place these files beside app.py to activate "
            "the AI forecasting module:"
        )

        st.code(
            """
energy_forecasting_xgboost.pkl
model_features.json
            """
        )

        st.info(
            "The CarbonTwin and Digital Twin modules "
            "continue to work without the AI model."
        )

    elif energy_file is None:

        st.info(
            "Upload your Steel_industry_data.csv "
            "to run AI energy analysis."
        )

    else:

        try:

            energy_df = pd.read_csv(
                energy_file
            )

            required_energy_columns = [
                "date",
                "Usage_kWh",
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
                "Lagging_Current_Power_Factor",
                "Leading_Current_Power_Factor",
                "WeekStatus",
                "Load_Type"
            ]

            missing_energy = [
                c for c in required_energy_columns
                if c not in energy_df.columns
            ]

            if missing_energy:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing_energy)
                )

            else:

                with st.spinner(
                    "Running AI energy analysis..."
                ):

                    results = analyze_energy(
                        energy_df
                    )

                actual = (
                    results["Usage_kWh"].sum()
                )

                expected = (
                    results["AI_Expected_kWh"].sum()
                )

                excess = (
                    results[
                        "Potential_Excess_kWh"
                    ].sum()
                )

                anomaly_count = int(
                    results["Anomaly"].sum()
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Actual Energy",
                    f"{actual:,.0f} kWh"
                )

                c2.metric(
                    "AI Baseline",
                    f"{expected:,.0f} kWh"
                )

                c3.metric(
                    "Potential Excess",
                    f"{excess:,.0f} kWh"
                )

                c4.metric(
                    "Anomalous Intervals",
                    f"{anomaly_count:,}"
                )

                st.subheader(
                    "Actual vs AI Expected"
                )

                chart = (
                    results[
                        [
                            "date",
                            "Usage_kWh",
                            "AI_Expected_kWh"
                        ]
                    ]
                    .set_index("date")
                    .rename(
                        columns={
                            "Usage_kWh": "Actual",
                            "AI_Expected_kWh": "Expected"
                        }
                    )
                )

                st.line_chart(chart)

                st.subheader(
                    "⚠️ Energy Anomalies"
                )

                anomalies = (
                    results[
                        results["Anomaly"]
                    ]
                    .sort_values(
                        "Z_score",
                        ascending=False
                    )
                )

                if anomalies.empty:

                    st.success(
                        "No statistically significant "
                        "high-energy anomalies detected."
                    )

                else:

                    # IMPORTANT:
                    # Correct column name used here.
                    display_columns = [
                        "date",
                        "Usage_kWh",
                        "AI_Expected_kWh",
                        "Excess_kWh",
                        "Deviation_%",
                        "Load_Type",
                        "Lagging_Current_Power_Factor",
                        "Lagging_Current_Reactive.Power_kVarh",
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
                            "Lagging_Current_Power_Factor":
                                "{:.3f}",
                            "Lagging_Current_Reactive.Power_kVarh":
                                "{:.2f}",
                            "Z_score": "{:.2f}"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

                    worst = anomalies.iloc[0]

                    st.subheader(
                        "🧠 Operator Interpretation"
                    )

                    st.warning(
                        f"""
High-energy event detected.

Time: {worst["date"]}

Actual consumption:
{worst["Usage_kWh"]:.2f} kWh

AI expected:
{worst["AI_Expected_kWh"]:.2f} kWh

Deviation:
+{worst["Deviation_%"]:.1f}%

Operating regime:
{worst["Load_Type"]}

Power factor:
{worst["Lagging_Current_Power_Factor"]:.3f}

Investigate whether the event was caused by:
• unusual production demand
• inefficient equipment operation
• motor loading
• low power factor
• flexible loads operating during the interval

This is an investigation opportunity, not proof
that all excess energy is avoidable.
"""
                    )

                potential_cost = (
                    excess * tariff
                )

                st.subheader(
                    "💰 Potential Energy Opportunity"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Potential excess",
                    f"{excess:,.0f} kWh"
                )

                c2.metric(
                    "Electricity tariff",
                    f"₹{tariff:.2f}/kWh"
                )

                c3.metric(
                    "Potential cost",
                    inr(potential_cost)
                )

        except Exception as e:

            st.error(
                f"Energy analysis failed: {e}"
            )


# ============================================================
# SIMPLE UTILITY DIGITAL TWIN
# ============================================================

def compressor_simulation(
    band,
    leak_cut,
    rotation_hours
):

    rng = np.random.default_rng(1)

    h = np.arange(1440) / 60

    shift = (
        (h >= 6) &
        (h < 22)
    )

    production = np.where(
        shift,
        11 +
        3 * np.sin(
            (h - 6) * np.pi / 8
        ) +
        rng.normal(
            0,
            0.8,
            1440
        ),
        0
    )

    production = np.convolve(
        production,
        np.ones(15) / 15,
        "same"
    ).clip(0)

    baseline_demand = (
        production + 1.6
    )

    optimized_demand = (
        production +
        1.6 * (
            1 -
            leak_cut / 100
        )
    )

    baseline_power = np.zeros(1440)
    optimized_power = np.zeros(1440)

    for m in range(1440):

        n = (
            3 if shift[m]
            else 1
        )

        load_fraction = min(
            baseline_demand[m] /
            (n * CAP),
            1
        )

        baseline_power[m] = (
            n *
            P_LOAD *
            (
                load_fraction +
                UNL *
                (1 - load_fraction)
            )
        )

        machines = max(
            1,
            int(
                np.ceil(
                    optimized_demand[m] /
                    CAP
                )
            )
        )

        machines = min(
            machines,
            3
        )

        remaining = min(
            max(
                optimized_demand[m] -
                (machines - 1) * CAP,
                0
            ),
            CAP
        )

        optimized_power[m] = (
            (
                (machines - 1) *
                P_LOAD
            )
            +
            P_LOAD *
            (
                0.12 +
                0.88 *
                remaining /
                CAP
            )
        )

        optimized_power[m] *= (
            1 -
            0.065 * band
        )

    return {
        "baseline": baseline_power,
        "optimized": optimized_power,
        "production": production
    }


# ============================================================
# TAB 4 — UTILITY OPTIMIZER
# ============================================================

with tabs[3]:

    st.header(
        "⚙️ Industrial Utility Optimizer"
    )

    st.caption(
        "Digital-twin scenarios for compressors and industrial utilities"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        pressure_reduction = st.slider(
            "Compressor set-point reduction",
            0.0,
            1.5,
            0.7,
            0.1
        )

    with c2:

        leak_reduction = st.slider(
            "Air leak reduction (%)",
            0,
            80,
            30,
            5
        )

    with c3:

        rotation_hours = st.slider(
            "Lead rotation interval",
            0,
            24,
            4
        )

    sim = compressor_simulation(
        pressure_reduction,
        leak_reduction,
        rotation_hours
    )

    baseline_kwh_day = (
        sim["baseline"].sum() / 60
    )

    optimized_kwh_day = (
        sim["optimized"].sum() / 60
    )

    annual_saving_kwh = (
        baseline_kwh_day -
        optimized_kwh_day
    ) * operating_days

    annual_saving_inr = (
        annual_saving_kwh *
        tariff
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Baseline energy/day",
        f"{baseline_kwh_day:,.0f} kWh"
    )

    c2.metric(
        "Optimized energy/day",
        f"{optimized_kwh_day:,.0f} kWh"
    )

    c3.metric(
        "Annual saving",
        inr(annual_saving_inr)
    )

    chart = pd.DataFrame({
        "Baseline kW":
            sim["baseline"],

        "Optimized kW":
            sim["optimized"]
    })

    st.line_chart(chart)

    st.subheader(
        "Recommended Actions"
    )

    actions = pd.DataFrame({
        "Action": [
            "Reduce compressor pressure",
            "Repair compressed-air leaks",
            "Optimize compressor sequencing"
        ],

        "Potential benefit": [
            "Lower compressor power",
            "Lower air demand",
            "Avoid unloaded operation"
        ]
    })

    st.dataframe(
        actions,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf():

    if not REPORTLAB_AVAILABLE:

        return None

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            company_name,
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "GHG Emissions Report — FY 2026–27",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            f"Industry: {industry}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    report_data = [
        ["Metric", "Value"],

        [
            "Scope 1",
            f"{total_scope1:.2f} tCO₂e"
        ],

        [
            "Scope 2",
            f"{total_scope2:.2f} tCO₂e"
        ],

        [
            "Total GHG",
            f"{total_emissions:.2f} tCO₂e"
        ],

        [
            "Production",
            f"{total_production:.2f} tonnes"
        ],

        [
            "Emission Intensity",
            f"{emission_intensity:.4f} tCO₂e/tonne"
        ]
    ]

    table = Table(
        report_data,
        colWidths=[250, 200]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.darkgreen
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Energy Consumption",
            styles["Heading2"]
        )
    )

    energy_data = [
        ["Source", "Annual Consumption"],

        [
            "Electricity",
            f"{baseline_electricity:,.0f} kWh"
        ],

        [
            "Diesel",
            f"{baseline_diesel:,.0f} L"
        ],

        [
            "Natural Gas",
            f"{baseline_gas:,.0f} SCM"
        ]
    ]

    energy_table = Table(
        energy_data,
        colWidths=[250, 200]
    )

    energy_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.darkgreen
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(
        energy_table
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# TAB 5 — GHG REPORTING
# ============================================================

with tabs[4]:

    st.header(
        "📄 GHG Reporting"
    )

    st.write(
        """
        Generate a simplified GHG inventory and
        buyer sustainability disclosure.
        """
    )

    if not carbon_data_available:

        st.warning(
            "Carbon accounting data is unavailable."
        )

    else:

        st.subheader(
            "GHG Inventory"
        )

        report_df = pd.DataFrame({
            "Category": [
                "Scope 1",
                "Scope 2",
                "Total"
            ],

            "Emissions (tCO₂e)": [
                total_scope1,
                total_scope2,
                total_emissions
            ]
        })

        st.dataframe(
            report_df.style.format({
                "Emissions (tCO₂e)":
                    "{:,.2f}"
            }),
            hide_index=True,
            use_container_width=True
        )

        st.subheader(
            "🤝 Buyer Sustainability Disclosure"
        )

        st.markdown(
            f"""
## {company_name}

**Industry:** {industry}

### GHG Emissions

| Category | Emissions |
|---|---:|
| Scope 1 | {total_scope1:,.2f} tCO₂e |
| Scope 2 | {total_scope2:,.2f} tCO₂e |
| **Total** | **{total_emissions:,.2f} tCO₂e** |

### Energy Consumption

- Electricity: {baseline_electricity:,.0f} kWh
- Diesel: {baseline_diesel:,.0f} L
- Natural Gas: {baseline_gas:,.0f} SCM

### Carbon Intensity

**{emission_intensity:.4f} tCO₂e / tonne**

### Reporting Boundary

Manufacturing operations included in the uploaded dataset.

> Prototype disclosure — emission factors, boundaries and
> reporting assumptions should be verified before formal use.
"""
        )

        if REPORTLAB_AVAILABLE:

            pdf = generate_pdf()

            st.download_button(
                "📥 Download GHG Report",
                data=pdf,
                file_name="SME_GHG_Report.pdf",
                mime="application/pdf"
            )

        else:

            st.warning(
                "Install reportlab to enable PDF download:"
            )

            st.code(
                "pip install reportlab"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CarbonTwin — SME Carbon Digital Twin & GHG Reporting Prototype"
)
