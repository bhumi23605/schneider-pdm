import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CarbonTwin | SME Decarbonization",
    page_icon="🌱",
    layout="wide",
)


# ============================================================
# DEFAULT CONSTANTS
# ============================================================

DEFAULT_GRID_EF = 0.70       # kg CO2e / kWh
DEFAULT_TARIFF = 8.0         # ₹ / kWh


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_factory_csv(file_bytes):

    df = pd.read_csv(
        pd.io.common.BytesIO(file_bytes)
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    required_columns = {
        "date",
        "Usage_kWh"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Date
    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce"
    )

    # Electricity
    df["Usage_kWh"] = pd.to_numeric(
        df["Usage_kWh"],
        errors="coerce"
    ).fillna(0)

    # Numeric columns
    numeric_columns = [
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Remove invalid dates
    df = df.dropna(
        subset=["date"]
    )

    # Sort
    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return df


@st.cache_data
def load_benchmark(file_bytes):

    excel_file = pd.ExcelFile(
        pd.io.common.BytesIO(file_bytes)
    )

    if "Energy Use Emissions" in excel_file.sheet_names:

        benchmark = pd.read_excel(
            pd.io.common.BytesIO(file_bytes),
            sheet_name="Energy Use Emissions",
            header=1
        )

    else:

        benchmark = pd.read_excel(
            pd.io.common.BytesIO(file_bytes),
            sheet_name=excel_file.sheet_names[0]
        )

    benchmark.columns = [
        str(c).strip()
        for c in benchmark.columns
    ]

    return benchmark


def format_money(value):

    return f"₹{value:,.0f}"


def format_tco2(value):

    return f"{value:,.1f} tCO₂e"


# ============================================================
# HEADER
# ============================================================

st.title("🎯 CarbonTwin")

st.subheader(
    "SME Decarbonization Intelligence Platform"
)

st.caption(
    "Factory energy → Carbon footprint → Energy intelligence → "
    "Digital twin → Decarbonization opportunities"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Factory Configuration"
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


state = st.sidebar.selectbox(
    "Factory State",
    [
        "Maharashtra",
        "Gujarat",
        "Karnataka",
        "Tamil Nadu",
        "Odisha",
        "Jharkhand",
        "Chhattisgarh",
        "West Bengal",
        "Other"
    ]
)


grid_ef = st.sidebar.number_input(
    "Grid emission factor (kg CO₂e/kWh)",
    min_value=0.0,
    value=DEFAULT_GRID_EF,
    step=0.01
)


electricity_tariff = st.sidebar.number_input(
    "Electricity tariff (₹/kWh)",
    min_value=0.0,
    value=DEFAULT_TARIFF,
    step=0.5
)


st.sidebar.markdown("---")


st.sidebar.header(
    "📂 Data"
)


factory_file = st.sidebar.file_uploader(
    "Upload factory energy CSV",
    type=["csv"],
    help="Upload Steel_industry_data.csv"
)


benchmark_file = st.sidebar.file_uploader(
    "Upload industry benchmark",
    type=["xlsx", "xls", "csv"],
    help="Upload the Industrial Energy Use & Emissions database"
)


# ============================================================
# LOAD FACTORY DATA
# ============================================================

if factory_file is None:

    st.info(
        "👈 Upload your Steel_industry_data.csv "
        "from the sidebar to start."
    )

    st.markdown(
        "### Supported factory dataset"
    )

    st.code(
        """date
Usage_kWh
Lagging_Current_Reactive.Power_kVarh
Leading_Current_Reactive_Power_kVarh
CO2(tCO2)
Lagging_Current_Power_Factor
Leading_Current_Power_Factor
NSM
WeekStatus
Day_of_week
Load_Type"""
    )

    st.stop()


try:

    df = load_factory_csv(
        factory_file.getvalue()
    )

except Exception as e:

    st.error(
        f"Could not read the factory CSV: {e}"
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded factory CSV contains "
        "no valid records."
    )

    st.stop()


# ============================================================
# FACTORY DATA METRICS
# ============================================================

start_date = df["date"].min()

end_date = df["date"].max()


days_covered = max(
    (
        end_date - start_date
    ).total_seconds() / 86400 + 1,
    1
)


years_covered = (
    days_covered / 365.0
)


total_electricity = max(
    df["Usage_kWh"].sum(),
    0
)


annual_electricity = (
    total_electricity /
    years_covered
)


# Scope 2 footprint
scope2_tco2e = (
    annual_electricity *
    grid_ef /
    1000
)


# Electricity cost
annual_electricity_cost = (
    annual_electricity *
    electricity_tariff
)


# Dataset-reported CO2
reported_co2 = None


if "CO2(tCO2)" in df.columns:

    reported_co2 = df[
        "CO2(tCO2)"
    ].sum(
        skipna=True
    )


# Sidebar status
st.sidebar.success(
    f"✓ {len(df):,} records loaded"
)


st.sidebar.caption(
    f"Coverage: "
    f"{start_date:%d %b %Y} → "
    f"{end_date:%d %b %Y}"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏭 Carbon Footprint",
        "🔮 Digital Twin",
        "🤖 Energy Intelligence",
        "♻️ Waste Heat",
        "📊 Industry Benchmark"
    ]
)


# ============================================================
# TAB 1
# CARBON FOOTPRINT
# ============================================================

with tab1:

    st.header(
        "🌍 Factory Carbon Footprint"
    )

    st.caption(
        "The uploaded steel dataset contains electricity "
        "measurements. It does not contain diesel, natural-gas "
        "or production data, so CarbonTwin does not fabricate "
        "those values."
    )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Annual Electricity",
            f"{annual_electricity:,.0f} kWh"
        )


    with c2:

        st.metric(
            "Scope 2 Footprint",
            format_tco2(scope2_tco2e)
        )


    with c3:

        st.metric(
            "Annual Electricity Cost",
            format_money(
                annual_electricity_cost
            )
        )


    with c4:

        st.metric(
            "Data Coverage",
            f"{days_covered:.0f} days"
        )


    # --------------------------------------------------------
    # SOURCE CO2 INFORMATION
    # --------------------------------------------------------

    if reported_co2 is not None:

        st.info(
            f"The uploaded CSV also contains a "
            f"`CO2(tCO2)` field with approximately "
            f"**{reported_co2:,.2f} tCO₂** in the source data. "
            f"CarbonTwin keeps this source value separate from "
            f"the configurable grid-emission-factor calculation."
        )


    st.markdown("---")


    # --------------------------------------------------------
    # MONTHLY ENERGY
    # --------------------------------------------------------

    monthly = (
        df
        .set_index("date")
        .resample("ME")["Usage_kWh"]
        .sum()
        .reset_index()
    )


    monthly["CO2_tCO2e"] = (
        monthly["Usage_kWh"] *
        grid_ef /
        1000
    )


    monthly["Cost_INR"] = (
        monthly["Usage_kWh"] *
        electricity_tariff
    )


    col1, col2 = st.columns(2)


    with col1:

        fig = px.line(
            monthly,
            x="date",
            y="Usage_kWh",
            markers=True,
            title="Monthly Electricity Consumption"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Electricity (kWh)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    with col2:

        fig = px.bar(
            monthly,
            x="date",
            y="CO2_tCO2e",
            title="Monthly Scope 2 Emissions"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="tCO₂e"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    # LOAD TYPE
    if "Load_Type" in df.columns:

        st.subheader(
            "⚡ Energy Consumption by Load Type"
        )
        load_energy = (
            df
            .groupby(
                "Load_Type",
                dropna=False
            )["Usage_kWh"]
            .sum()
            .reset_index()
        )

        load_energy["Share_%"] = (
            load_energy["Usage_kWh"] /
            load_energy["Usage_kWh"].sum() *
            100
        )
        fig = px.bar(
            load_energy,
            x="Load_Type",
            y="Usage_kWh",
            text=(
                load_energy["Share_%"]
                .round(1)
                .astype(str)
                + "%"
            ),
            title="Electricity Consumption by Load Type"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        hotspot = load_energy.loc[
            load_energy["Usage_kWh"].idxmax()
        ]
        st.warning(
            f"🔥 **Largest energy hotspot:** "
            f"{hotspot['Load_Type']} "
            f"({hotspot['Share_%']:.1f}% "
            f"of electricity consumption)."
        )


    # --------------------------------------------------------
    # POWER FACTOR
    # --------------------------------------------------------

    st.subheader(
        "⚡ Power Quality"
    )
    p1, p2 = st.columns(2)
    if "Lagging_Current_Power_Factor" in df.columns:
        lagging_pf = (
            df[
                "Lagging_Current_Power_Factor"
            ].mean()
        )
        p1.metric(
            "Average Lagging Power Factor",
            f"{lagging_pf:.2f}"
        )
    else:
        p1.metric(
            "Average Lagging Power Factor",
            "N/A"
        )
    if "Leading_Current_Power_Factor" in df.columns:
        leading_pf = (
            df[
                "Leading_Current_Power_Factor"
            ].mean()
        )
        p2.metric(
            "Average Leading Power Factor",
            f"{leading_pf:.2f}"
        )
    else:
        p2.metric(
            "Average Leading Power Factor",
            "N/A"
        )
    # DECARBONIZATION

    st.markdown("---")
    st.header(
        "🎯 Decarbonization Opportunities"
    )
    efficiency_pct = st.slider(
        "Efficiency improvement assumption (%)",
        0,
        30,
        10,
        key="main_efficiency"
    )
    renewable_pct = st.slider(
        "Renewable electricity share (%)",
        0,
        100,
        30,
        key="main_renewable"
    )
    # Efficiency
    efficiency_kwh = (
        annual_electricity *
        efficiency_pct /
        100
    )
    efficiency_co2 = (
        efficiency_kwh *
        grid_ef /
        1000
    )
    efficiency_saving = (
        efficiency_kwh *
        electricity_tariff
    )
    remaining_after_efficiency = (
        annual_electricity -
        efficiency_kwh
    )
    renewable_kwh = (
        remaining_after_efficiency *
        renewable_pct /
        100
    )
    future_grid_energy = (
        remaining_after_efficiency -
        renewable_kwh
    )
    future_scope2 = (
        future_grid_energy *
        grid_ef /
        1000
    )
    total_reduction = (
        scope2_tco2e -
        future_scope2
    )
    reduction_pct = (
        total_reduction /
        scope2_tco2e *
        100
        if scope2_tco2e > 0

        else 0
    )
    o1, o2, o3 = st.columns(3)
    with o1:
        st.metric(
            "Efficiency Saving",
            f"{efficiency_kwh:,.0f} kWh/year"
        )
    with o2:
        st.metric(
            "Renewable Electricity",
            f"{renewable_kwh:,.0f} kWh/year"
        )
    with o3:

        st.metric(
            "Potential CO₂ Reduction",
            format_tco2(total_reduction)
        )

    # DECARBONIZATION SCENARIO
    scenario = pd.DataFrame(
        {
            "Scenario": [
                "Current",
                "After Efficiency",
                "Efficiency + Renewable"
            ],
            "Emissions": [
                scope2_tco2e,

                remaining_after_efficiency
                * grid_ef
                / 1000,

                future_scope2
            ]
        }
    )
    fig = px.bar(
        scenario,
        x="Scenario",
        y="Emissions",
        text="Emissions",
        title="CarbonTwin Decarbonization Pathway"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric(
            "Current Footprint",
            format_tco2(scope2_tco2e)
        )
    with s2:
        st.metric(
            "Scenario Footprint",
            format_tco2(future_scope2)
        )
    with s3:
        st.metric(
            "Potential Reduction",
            f"{reduction_pct:.1f}%"
        )
    st.success(
        f"""
        💡 **Management takeaway**
        Under the selected assumptions, the factory could reduce
        electricity-related emissions by approximately
        **{total_reduction:,.1f} tCO₂e/year**
        """
    )

# TAB 2
# DIGITAL TWIN
with tab2:
    st.header(
        "🔮 Factory Energy Digital Twin"
    )
    st.write(
        "Explore how efficiency improvements and renewable "
        "electricity change the factory's energy demand "
        "and Scope 2 footprint."
    )
    efficiency = st.slider(
        "Efficiency improvement (%)",
        0,
        40,
        10,
        key="twin_efficiency"
    )
    renewable = st.slider(
        "Renewable electricity share (%)",
        0,
        100,
        30,
        key="twin_renewable"
    )
    efficient_energy = (
        annual_electricity *
        (
            1 -
            efficiency /
            100
        )
    )
    renewable_energy = (
        efficient_energy *
        renewable /
        100
    )
    grid_energy = (
        efficient_energy -
        renewable_energy
    )
    baseline_emissions = (
        annual_electricity *
        grid_ef /
        1000
    )
    scenario_emissions = (
        grid_energy *
        grid_ef /
        1000
    )
    avoided_emissions = (
        baseline_emissions -
        scenario_emissions
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Baseline",
            format_tco2(
                baseline_emissions
            )
        )
    with c2:
        st.metric(
            "Scenario",
            format_tco2(
                scenario_emissions
            )
        )
    with c3:
        st.metric(
            "CO₂ Avoided",
            format_tco2(
                avoided_emissions
            )
        )
    twin = pd.DataFrame(
        {
            "Scenario": [
                "Current Factory",
                "After Efficiency",
                "Efficiency + Renewable"
            ],
            "Grid Electricity (kWh)": [
                annual_electricity,
                efficient_energy,
                grid_energy
            ]
        }
    )
    fig = px.bar(
        twin,
        x="Scenario",
        y="Grid Electricity (kWh)",
        text="Grid Electricity (kWh)",
        title="Digital Twin Energy Scenario"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    avoided_cost = (
        annual_electricity -
        grid_energy
    ) * electricity_tariff
    st.metric(
        "Potential Annual Electricity-Cost Avoidance",
        format_money(
            avoided_cost
        )
    )
    st.info(
        "The digital twin is a what-if simulation. "
        "The selected percentages are scenarios and do not "
        "claim that these improvements have already been achieved."
    )

# ENERGY INTELLIGENCE

with tab3:

    st.header(
        "🤖 Energy Intelligence"
    )
    st.write(
        "Unsupervised anomaly detection identifies operating "
        "periods whose energy profile differs significantly "
        "from normal behavior."
    )
    # SELECT ML FEATURES

    features = [
        "Usage_kWh"
    ]
    additional_features = [
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor"
    ]
    for col in additional_features:
        if col in df.columns:
            features.append(col)
    model_df = (
        df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )
    if len(model_df) >= 100:
        model = IsolationForest(
            n_estimators=150,
            contamination=0.02,
            random_state=42,
            n_jobs=-1
        )
        labels = model.fit_predict(
            model_df
        )
        anomaly_scores = (
            model.decision_function(
                model_df
            )
        )
        result = df.loc[
            model_df.index
        ].copy()
        result["Anomaly_Score"] = (
            anomaly_scores
        )
        result["Anomaly"] = (
            labels == -1
        )
        anomalies = result[
            result["Anomaly"]
        ].copy()
        # KPIs
    
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Records Analyzed",
                f"{len(model_df):,}"
            )
        with c2:
            st.metric(
                "Potential Anomalies",
                f"{len(anomalies):,}"
            )
        with c3:
            st.metric(
                "Anomalous Energy",
                f"{anomalies['Usage_kWh'].sum():,.0f} kWh"
            )
        # DAILY ENERGY
        daily = (
            result
            .set_index("date")
            .resample("D")
            .agg(
                Energy_kWh=(
                    "Usage_kWh",
                    "sum"
                ),
                Anomalies=(
                    "Anomaly",
                    "sum"
                )
            )
            .reset_index()
        )
        fig = px.line(
            daily,
            x="date",
            y="Energy_kWh",
            title="Daily Electricity Consumption"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        # ANOMALIES

        if not anomalies.empty:
            st.subheader(
                "🚨 Highest-Priority Energy Anomalies"
            )
            display_columns = [
                "date",
                "Usage_kWh",
                "Anomaly_Score"
            ]
            if "Load_Type" in anomalies.columns:
                display_columns.append(
                    "Load_Type"
                )
            anomaly_table = (
                anomalies[
                    display_columns
                ]
                .sort_values(
                    "Anomaly_Score"
                )
                .head(25)
            )
            st.dataframe(
                anomaly_table,
                use_container_width=True,
                hide_index=True
            )
            st.warning(
                """
                Investigate these periods for equipment overload,
                unexpected process changes, idle running,
                maintenance issues or abnormal operating conditions.
                """
            )
    else:
        st.warning(
            "Not enough valid records for anomaly detection."
        )
    # LOAD ANALYSIS

    if "Load_Type" in df.columns:
        st.markdown("---")
        st.subheader(
            "🏭 Operational Load Analysis"
        )
        load_stats = (
            df
            .groupby(
                "Load_Type",
                dropna=False
            )
            .agg(
                Energy_kWh=(
                    "Usage_kWh",
                    "sum"
                ),
                Average_kWh=(
                    "Usage_kWh",
                    "mean"
                ),
                Peak_kWh=(
                    "Usage_kWh",
                    "max"
                )
            )
            .reset_index()
        )
        load_stats = (
            load_stats
            .sort_values(
                "Energy_kWh",
                ascending=False
            )
        )
        st.dataframe(
            load_stats,
            use_container_width=True,
            hide_index=True
        )

# WASTE HEAT

with tab4:

    st.header(
        "♻️ Waste Heat Recovery Opportunity"
    )
    st.write(
        """
        Estimate the potential value of recovering thermal energy
        from hot process equipment, exhaust streams, compressors
        or other industrial sources.
        """
    )
    st.info(
        """
        Your uploaded electricity dataset does not contain
        waste-heat temperature or mass-flow measurements.

        Therefore this module uses engineering assumptions entered
        by the user instead of pretending that waste heat was
        measured from the CSV.
        """
    )
    st.subheader(
        "🌡️ Waste Heat Parameters"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        waste_heat_kw = st.number_input(
            "Recoverable waste heat (kW)",
            min_value=0.0,
            value=20.0,
            step=1.0
        )
    with c2:
        operating_hours = st.number_input(
            "Operating hours/year",
            min_value=1,
            value=4000,
            step=100
        )
    with c3:
        teg_efficiency = st.slider(
            "TEG conversion efficiency (%)",
            0.1,
            10.0,
            3.0,
            0.1
        )
    # CALCULATIONS
    
    heat_energy_kwh = (
        waste_heat_kw *
        operating_hours
    )
    recovered_electricity = (
        heat_energy_kwh *
        teg_efficiency /
        100
    )
    avoided_co2 = (
        recovered_electricity *
        grid_ef /
        1000
    )
    avoided_cost = (
        recovered_electricity *
        electricity_tariff
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric(
            "Recoverable Thermal Energy",
            f"{heat_energy_kwh:,.0f} kWh/year"
        )
    with r2:
        st.metric(
            "Potential Auxiliary Electricity",
            f"{recovered_electricity:,.0f} kWh/year"
        )
    with r3:
        st.metric(
            "Potential CO₂ Avoidance",
            format_tco2(
                avoided_co2
            )
        )
    st.metric(
        "Potential Electricity-Cost Avoidance",
        format_money(
            avoided_cost
        )
    )
    # TECHNOLOGIES

    st.markdown(
        "### 🔧 Recovery Options"
    )
    technology = pd.DataFrame(
        {
            "Technology": [
                "Heat Exchanger",
                "Economizer",
                "Waste Heat Boiler",
                "Thermoelectric Generator"
            ],
            "Best Use": [
                "Preheat air, water or process streams",
                "Preheat boiler/feedwater",
                "Generate useful steam",
                "Small auxiliary electrical loads"
            ]
        }
    )
    st.dataframe(
        technology,
        use_container_width=True,
        hide_index=True
    )
    st.warning(
        """
        For most industrial applications, direct thermal recovery
        through heat exchangers or economizers is generally more
        effective than using a Peltier/TEG solely to generate electricity.
        """
    )
# INDUSTRY BENCHMARK

with tab5:

    st.header(
        "📊 Industry Benchmark"
    )
    if benchmark_file is None:
        st.info(
            """
            Upload the Industrial Energy Use & Emissions XLSX
            to enable the industry reference layer.
            """
        )
    else:
        try:
            benchmark = load_benchmark(
                benchmark_file.getvalue()
            )
        except Exception as e:
            st.error(
                f"Could not read the benchmark file: {e}"
            )

            st.stop()
        required_benchmark_columns = {
            "Industry",
            "Fuel Type",
            "State"
        }
        missing = (
            required_benchmark_columns -
            set(benchmark.columns)
        )
        if missing:
            st.error(
                "Benchmark file is missing: "
                +
                ", ".join(
                    sorted(missing)
                )
            )
        else:
            # INDUSTRY MAPPING

            industry_map = {

                "Steel Manufacturing":
                    "Iron and Steel",
                "Fabrication":
                    "Iron and Steel",
                "Automotive Components":
                    "Transport Equipment",
                "Food Processing":
                    "Food Products",
                "Textile":
                    "Textiles"
            }
            selected_industry = (
                industry_map.get(
                    industry,
                    "Iron and Steel"
                )
            )
            benchmark_filtered = (
                benchmark[
                    benchmark["Industry"]
                    .astype(str)
                    .str.strip()
                    .str.casefold()
                    ==
                    selected_industry.casefold()
                ]
                .copy()
            )
            if benchmark_filtered.empty:
                st.warning(
                    f"No benchmark records found for "
                    f"{selected_industry}."
                )

            else:
                # DETECT AVAILABLE REFERENCE YEARS
                year_columns = [
                    c
                    for c in benchmark_filtered.columns

                    if (
                        str(c).strip().startswith("20")
                        and
                        " - " in str(c)
                    )
                ]
                if not year_columns:
                    st.warning(
                        "No year columns were detected "
                        "in the benchmark file."
                    )
                else:
                    latest_reference_period = sorted(
                        year_columns,
                        key=str
                    )[-1]
                    benchmark_filtered[
                        latest_reference_period
                    ] = pd.to_numeric(
                        benchmark_filtered[
                            latest_reference_period
                        ],
                        errors="coerce"
                    ).fillna(0)
                    # FUEL PROFILE

                    fuel_summary = (
                        benchmark_filtered
                        .groupby(
                            "Fuel Type"
                        )[
                            latest_reference_period
                        ]
                        .sum()
                        .reset_index()
                    )
                    fuel_summary = fuel_summary[
                        fuel_summary[
                            latest_reference_period
                        ] > 0
                    ].copy()
                    if not fuel_summary.empty:
                        fuel_summary["Share_%"] = (
                            fuel_summary[
                                latest_reference_period
                            ]
                            /
                            fuel_summary[
                                latest_reference_period
                            ].sum()
                            *
                            100
                        )
                        fig = px.bar(
                            fuel_summary,
                            x="Fuel Type",
                            y="Share_%",
                            text=(fuel_summary["Share_%"].round(1).astype(str)+ "%")
                            title=(
                                "Industry Fuel / "
                                "Emissions Profile"
                            )
                        )
                        fig.update_layout(
                            yaxis_title="Share (%)",
                            xaxis_title="Fuel"
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )
                    # SELECTED STATE

                    selected_state_data = (
                        benchmark_filtered[
                            benchmark_filtered[
                                "State"
                            ]
                            .astype(str)
                            .str.strip()
                            .str.casefold()
                            ==
                            state.casefold()
                        ]
                    )
                    if not selected_state_data.empty:

                        state_value = (
                            selected_state_data[
                                latest_reference_period
                            ]
                            .sum()
                        )


                        st.metric(
                            "Industry Reference Value "
                            "for Selected State",
                            f"{state_value:,.0f}"
                        )
                    # STATE HOTSPOTS
                    st.subheader(
                        "🗺️ Industry State Hotspots"
                    )
                    state_summary = (
                        benchmark_filtered
                        .groupby(
                            "State"
                        )[
                            latest_reference_period
                        ]
                        .sum()
                        .reset_index()
                        .sort_values(
                            latest_reference_period,
                            ascending=False
                        )
                        .head(10)
                    )


                    fig = px.bar(
                        state_summary,
                        x=latest_reference_period,
                        y="State",
                        orientation="h",
                        title="Industry State Hotspots"
                    )


                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


                    # ------------------------------------------------
                    # IMPORTANT DISCLOSURE
                    # ------------------------------------------------

                    st.caption(
                        """
                        Industry benchmark data is used only as a
                        reference layer for understanding fuel mix,
                        regional hotspots and industry context.

                        It is not treated as the factory's current
                        footprint and is not mixed into the factory
                        carbon calculation.
                        """
                    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(
    """
    CarbonTwin | SME Decarbonization Intelligence Platform

    Factory data drives the footprint →
    Energy intelligence →
    Digital twin →
    Decarbonization scenarios →
    Industry reference
    """
)
