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
    type=["csv"],
    help="Current factory energy and production data"
)

benchmark_file = st.sidebar.file_uploader(
    "Upload Industry Benchmark (optional)",
    type=["xlsx", "xls", "csv"],
    help="Industrial energy/emissions benchmark used only for comparison"
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
# INDUSTRY BENCHMARK LOADER
# ============================================================

benchmark_df = None
benchmark_available = False

if benchmark_file is not None:

    try:

        if benchmark_file.name.lower().endswith(".csv"):

            benchmark_df = pd.read_csv(
                benchmark_file
            )

        else:

            xls = pd.ExcelFile(
                benchmark_file
            )

            if "Energy Use Emissions" in xls.sheet_names:

                benchmark_df = pd.read_excel(
                    benchmark_file,
                    sheet_name="Energy Use Emissions",
                    header=1
                )

            else:

                benchmark_df = pd.read_excel(
                    benchmark_file,
                    header=1
                )

        benchmark_df.columns = [
            str(c).strip()
            for c in benchmark_df.columns
        ]

        required_benchmark_columns = [
            "Industry",
            "Fuel Type",
            "State"
        ]

        if all(
            c in benchmark_df.columns
            for c in required_benchmark_columns
        ):

            benchmark_available = True

    except Exception as e:

        st.sidebar.warning(
            f"Benchmark file could not be loaded: {e}"
        )


# ============================================================
# INDUSTRY NAME MAPPING
# ============================================================

benchmark_industry_map = {

    "Steel Manufacturing": "Iron and Steel",

    "Fabrication": "Iron and Steel",

    "Automotive Components": "Transport Equipment",

    "Food Processing": "Food Products",

    "Textile": "Textiles",

    "Other": None
}


benchmark_industry = benchmark_industry_map.get(
    industry
)


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

# ============================================================
# TAB 1 — DECARBONIZATION OPPORTUNITY ASSESSMENT
# ============================================================

with tabs[0]:

    st.header(
        "🎯 Decarbonization Opportunity Assessment"
    )

    st.caption(
        "Turn current factory energy consumption into "
        "measurable carbon-reduction opportunities."
    )

    # ========================================================
    # NO CURRENT DATA
    # ========================================================

    if not carbon_data_available:

        st.warning(
            "Upload your current SME energy CSV to calculate "
            "the factory carbon footprint."
        )

        st.markdown(
            """
            ### Required columns

            Your current factory dataset should contain:

            ```text
            electricity_kwh
            diesel_litres
            natural_gas_scm
            production_tonnes
            ```

            These represent **current factory activity data**.
            """
        )

    else:

        # ====================================================
        # CURRENT FACTORY BASELINE
        # ====================================================

        st.subheader("🏭 Current Factory Baseline")

        # Annualized values
        data_months = len(df)

        if data_months > 0 and data_months < 12:

            annual_factor = 12 / data_months

        else:

            annual_factor = 1

        annual_emissions = (
            total_emissions * annual_factor
        )

        annual_scope1 = (
            total_scope1 * annual_factor
        )

        annual_scope2 = (
            total_scope2 * annual_factor
        )

        annual_production = (
            total_production * annual_factor
        )

        annual_electricity = (
            baseline_electricity *
            annual_factor
        )

        annual_diesel = (
            baseline_diesel *
            annual_factor
        )

        annual_gas = (
            baseline_gas *
            annual_factor
        )

        annual_intensity = (

            annual_emissions /
            annual_production

            if annual_production > 0

            else 0
        )


        # ====================================================
        # KPI CARDS
        # ====================================================

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Current GHG Footprint",
                f"{annual_emissions:,.0f} tCO₂e/year"
            )

        with c2:

            st.metric(
                "Scope 1",
                f"{annual_scope1:,.0f} tCO₂e/year"
            )

        with c3:

            st.metric(
                "Scope 2",
                f"{annual_scope2:,.0f} tCO₂e/year"
            )

        with c4:

            st.metric(
                "Carbon Intensity",
                f"{annual_intensity:.3f} tCO₂e/t"
            )


        # ====================================================
        # SOURCE BREAKDOWN
        # ====================================================

        st.divider()

        st.subheader(
            "🔥 Where is the Carbon Coming From?"
        )

        current_breakdown = pd.DataFrame({

            "Source": [
                "Grid Electricity",
                "Diesel",
                "Natural Gas"
            ],

            "Emissions": [

                annual_electricity *
                electricity_ef / 1000,

                annual_diesel *
                DIESEL_EF / 1000,

                annual_gas *
                NATURAL_GAS_EF / 1000
            ]

        })

        current_breakdown["Share"] = (

            current_breakdown["Emissions"] /
            current_breakdown["Emissions"].sum() *
            100

        )


        col1, col2 = st.columns(2)


        with col1:

            fig = px.bar(
                current_breakdown,
                x="Source",
                y="Emissions",
                text="Share",
                title="Current Carbon Sources",
                labels={
                    "Emissions": "tCO₂e/year"
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            fig = px.pie(
                current_breakdown,
                names="Source",
                values="Emissions",
                hole=0.45,
                title="Current Emission Mix"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ====================================================
        # IDENTIFY CURRENT HOTSPOT
        # ====================================================

        hotspot = (
            current_breakdown
            .sort_values(
                "Emissions",
                ascending=False
            )
            .iloc[0]
        )

        hotspot_source = hotspot["Source"]
        hotspot_share = hotspot["Share"]

        st.warning(
            f"""
            **Primary carbon hotspot: {hotspot_source}**

            It currently contributes approximately
            **{hotspot_share:.1f}%** of the factory's calculated
            footprint.

            This should be investigated before prioritizing
            decarbonization investments.
            """
        )


        # ====================================================
        # DECARBONIZATION OPPORTUNITY ENGINE
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Top Decarbonization Opportunities"
        )

        st.caption(
            "Potential values are scenario estimates based on "
            "the current factory baseline. They are not claimed "
            "as measured savings."
        )


        # -----------------------------------------------
        # USER ASSUMPTIONS
        # -----------------------------------------------

        with st.expander(
            "⚙️ Adjust opportunity assumptions"
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                solar_reduction = st.slider(
                    "Renewable electricity (%)",
                    0,
                    100,
                    20,
                    5,
                    key="carbon_solar"
                )

            with col2:

                efficiency_reduction = st.slider(
                    "Energy efficiency (%)",
                    0,
                    30,
                    10,
                    5,
                    key="carbon_efficiency"
                )

            with col3:

                diesel_switch = st.slider(
                    "Diesel electrification (%)",
                    0,
                    100,
                    20,
                    5,
                    key="carbon_diesel"
                )


        # ====================================================
        # OPPORTUNITY CALCULATIONS
        # ====================================================

        opportunities = []


        # -----------------------------------------------
        # 1. RENEWABLE ELECTRICITY
        # -----------------------------------------------

        electricity_co2 = (
            annual_electricity *
            electricity_ef / 1000
        )

        solar_co2_reduction = (
            electricity_co2 *
            solar_reduction /
            100
        )

        solar_energy_replaced = (
            annual_electricity *
            solar_reduction /
            100
        )

        solar_cost_saving = (
            solar_energy_replaced *
            tariff
        )

        # Illustrative investment assumption
        solar_capex = (
            solar_energy_replaced *
            65000 / 1000
        )

        solar_payback = (

            solar_capex /
            solar_cost_saving

            if solar_cost_saving > 0

            else 0
        )


        opportunities.append({

            "Opportunity":
                "Renewable electricity",

            "CO₂ Reduction":
                solar_co2_reduction,

            "Annual Saving":
                solar_cost_saving,

            "Investment":
                solar_capex,

            "Payback":
                solar_payback

        })


        # -----------------------------------------------
        # 2. ENERGY EFFICIENCY
        # -----------------------------------------------

        efficiency_energy_saved = (
            annual_electricity *
            efficiency_reduction /
            100
        )

        efficiency_co2_reduction = (
            efficiency_energy_saved *
            electricity_ef /
            1000
        )

        efficiency_cost_saving = (
            efficiency_energy_saved *
            tariff
        )

        efficiency_capex = (
            efficiency_cost_saving *
            2.0
        )

        efficiency_payback = (

            efficiency_capex /
            efficiency_cost_saving

            if efficiency_cost_saving > 0

            else 0
        )


        opportunities.append({

            "Opportunity":
                "Energy efficiency / VFD optimization",

            "CO₂ Reduction":
                efficiency_co2_reduction,

            "Annual Saving":
                efficiency_cost_saving,

            "Investment":
                efficiency_capex,

            "Payback":
                efficiency_payback

        })


        # -----------------------------------------------
        # 3. DIESEL ELECTRIFICATION
        # -----------------------------------------------

        diesel_co2 = (
            annual_diesel *
            DIESEL_EF /
            1000
        )

        diesel_co2_reduction = (
            diesel_co2 *
            diesel_switch /
            100
        )

        diesel_litres_replaced = (
            annual_diesel *
            diesel_switch /
            100
        )

        diesel_cost_saving = (
            diesel_litres_replaced *
            90
        )

        diesel_capex = (
            diesel_cost_saving *
            2.5
        )

        diesel_payback = (

            diesel_capex /
            diesel_cost_saving

            if diesel_cost_saving > 0

            else 0
        )


        opportunities.append({

            "Opportunity":
                "Diesel → electric conversion",

            "CO₂ Reduction":
                diesel_co2_reduction,

            "Annual Saving":
                diesel_cost_saving,

            "Investment":
                diesel_capex,

            "Payback":
                diesel_payback

        })


        # ====================================================
        # OPPORTUNITY TABLE
        # ====================================================

        opportunity_df = pd.DataFrame(
            opportunities
        )


        # Rank by CO2 reduction
        opportunity_df = (
            opportunity_df
            .sort_values(
                "CO₂ Reduction",
                ascending=False
            )
            .reset_index(drop=True)
        )


        opportunity_df.insert(
            0,
            "Priority",
            [
                "🔴 HIGH",
                "🟠 MEDIUM",
                "🟡 MEDIUM"
            ][:len(opportunity_df)]
        )


        display_opportunities = (
            opportunity_df.copy()
        )

        display_opportunities[
            "CO₂ Reduction"
        ] = display_opportunities[
            "CO₂ Reduction"
        ].map(
            lambda x: f"{x:,.1f} tCO₂e/year"
        )

        display_opportunities[
            "Annual Saving"
        ] = display_opportunities[
            "Annual Saving"
        ].map(
            lambda x: f"₹{x:,.0f}/year"
        )

        display_opportunities[
            "Investment"
        ] = display_opportunities[
            "Investment"
        ].map(
            lambda x: f"₹{x:,.0f}"
        )

        display_opportunities[
            "Payback"
        ] = display_opportunities[
            "Payback"
        ].map(
            lambda x: f"{x:.1f} years"
        )


        st.dataframe(
            display_opportunities,
            hide_index=True,
            use_container_width=True
        )


        # ====================================================
        # TOTAL OPPORTUNITY
        # ====================================================

        total_potential_reduction = (
            opportunity_df[
                "CO₂ Reduction"
            ].sum()
        )

        total_annual_saving = (
            opportunity_df[
                "Annual Saving"
            ].sum()
        )

        total_investment = (
            opportunity_df[
                "Investment"
            ].sum()
        )

        portfolio_payback = (

            total_investment /
            total_annual_saving

            if total_annual_saving > 0

            else 0
        )

        reduction_percentage = (

            total_potential_reduction /
            annual_emissions *
            100

            if annual_emissions > 0

            else 0
        )

        # Don't allow the dashboard to claim >100%
        reduction_percentage = min(
            reduction_percentage,
            100
        )


        # ====================================================
        # OPPORTUNITY SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📉 Decarbonization Potential"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Potential CO₂ Reduction",
                f"{total_potential_reduction:,.0f} "
                "tCO₂e/year"
            )

        with c2:

            st.metric(
                "Potential Reduction",
                f"{reduction_percentage:.1f}%"
            )

        with c3:

            st.metric(
                "Potential Annual Saving",
                f"₹{total_annual_saving / 100000:.2f} L"
            )

        with c4:

            st.metric(
                "Portfolio Payback",
                f"{portfolio_payback:.1f} years"
            )


        # ====================================================
        # CURRENT VS POTENTIAL
        # ====================================================

        scenario_emissions = max(
            annual_emissions -
            total_potential_reduction,
            0
        )

        comparison = pd.DataFrame({

            "State": [
                "Current",
                "After Opportunities"
            ],

            "Emissions": [
                annual_emissions,
                scenario_emissions
            ]

        })


        fig = px.bar(
            comparison,
            x="State",
            y="Emissions",
            text="Emissions",
            title="Current vs Potential Carbon Footprint",
            labels={
                "Emissions": "tCO₂e/year"
            }
        )

        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # INDUSTRY BENCHMARK
        # ====================================================

        if (
            benchmark_available
            and benchmark_industry is not None
        ):

            st.divider()

            st.subheader(
                "🏭 Industry Benchmark Intelligence"
            )

            benchmark_subset = (
                benchmark_df[
                    benchmark_df["Industry"]
                    .astype(str)
                    .str.strip()
                    ==
                    benchmark_industry
                ]
                .copy()
            )


            if not benchmark_subset.empty:

                # Get latest available numeric year
                benchmark_years = [

                    c for c in benchmark_subset.columns

                    if isinstance(c, str)
                    and " - " in c
                    and c[:4].isdigit()

                ]

                if benchmark_years:

                    benchmark_year = (
                        benchmark_years[-1]
                    )

                    benchmark_subset[
                        benchmark_year
                    ] = pd.to_numeric(
                        benchmark_subset[
                            benchmark_year
                        ],
                        errors="coerce"
                    )

                    benchmark_subset = (
                        benchmark_subset
                        .dropna(
                            subset=[
                                benchmark_year
                            ]
                        )
                    )


                    benchmark_fuel = (
                        benchmark_subset
                        .groupby(
                            "Fuel Type"
                        )[benchmark_year]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                    )


                    benchmark_share = (

                        benchmark_fuel /
                        benchmark_fuel.sum() *
                        100

                    )


                    benchmark_table = pd.DataFrame({

                        "Benchmark Source":
                            benchmark_share.index,

                        "Share":
                            benchmark_share.values

                    })


                    # --------------------------------------
                    # Don't show benchmark year
                    # --------------------------------------

                    st.info(
                        f"""
                        **Sector reference:** {industry}

                        The benchmark is used to understand the
                        dominant carbon/energy sources in this
                        industry. It is **not** treated as your
                        factory's current emissions.
                        """
                    )


                    col1, col2 = st.columns(2)


                    with col1:

                        fig = px.bar(
                            benchmark_table,
                            x="Benchmark Source",
                            y="Share",
                            title="Industry Fuel / Emission Profile",
                            labels={
                                "Share":
                                    "Benchmark share (%)"
                            }
                        )

                        fig.update_layout(
                            xaxis_tickangle=-30
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )


                    with col2:

                        dominant_benchmark = (
                            benchmark_table
                            .iloc[0]
                        )

                        st.metric(
                            "Largest Industry Carbon Lever",
                            str(
                                dominant_benchmark[
                                    "Benchmark Source"
                                ]
                            )
                        )

                        st.metric(
                            "Benchmark Share",
                            f"{dominant_benchmark['Share']:.1f}%"
                        )


                        st.markdown(
                            """
                            ### What this means

                            The benchmark helps identify where
                            decarbonization should be investigated
                            first.

                            It does **not** replace the factory's
                            measured energy and emissions data.
                            """
                        )


                    # --------------------------------------
                    # Benchmark-driven recommendation
                    # --------------------------------------

                    top_source = str(
                        benchmark_table.iloc[0][
                            "Benchmark Source"
                        ]
                    )

                    if (
                        "coal" in
                        top_source.lower()
                    ):

                        recommendation = (
                            "Thermal fuel/process efficiency "
                            "and fuel switching should be "
                            "investigated as high-priority "
                            "decarbonization pathways."
                        )

                    elif (
                        "electric" in
                        top_source.lower()
                    ):

                        recommendation = (
                            "Electrical efficiency, renewable "
                            "electricity and load optimization "
                            "should be investigated first."
                        )

                    elif (
                        "petroleum" in
                        top_source.lower()
                    ):

                        recommendation = (
                            "Fuel electrification and lower-carbon "
                            "fuel alternatives should be evaluated."
                        )

                    else:

                        recommendation = (
                            "Investigate process efficiency and "
                            "fuel switching around the dominant "
                            "energy source."
                        )


                    st.success(
                        f"💡 **Industry-level insight:** "
                        f"{recommendation}"
                    )


        # ====================================================
        # MANAGEMENT TAKEAWAY
        # ====================================================

        st.divider()

        st.subheader(
            "💼 Management Takeaway"
        )

        top_opportunity = (
            opportunity_df.iloc[0]
        )

        st.markdown(
            f"""
            ### Recommended first action

            **{top_opportunity["Opportunity"]}**

            Estimated potential:

            - **CO₂ reduction:** 
              {top_opportunity["CO₂ Reduction"]:,.1f} tCO₂e/year
            - **Annual financial benefit:** 
              ₹{top_opportunity["Annual Saving"]:,.0f}
            - **Estimated payback:** 
              {top_opportunity["Payback"]:.1f} years

            The opportunity ranking is based on the current factory
            baseline and the scenario assumptions selected above.
            Actual savings should be validated through an engineering
            assessment before investment.
            """
        )
