import streamlit as st
import pandas as pd
import io

# --------------------------
# Page config & styling
# --------------------------
st.set_page_config(
    page_title="Serverless FinOps Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Make tables a bit cleaner */
    .dataframe tbody tr th {display:none;}
    .dataframe thead tr th:first-child {display:none;}
    .dataframe tbody tr td:first-child {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Data loading
# --------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # Your CSV has each row quoted – clean that first
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip().strip('"') for line in f]

    clean_csv = io.StringIO("\n".join(lines))
    df_ = pd.read_csv(clean_csv)

    numeric_cols = [
        "InvocationsPerMonth",
        "AvgDurationMs",
        "MemoryMB",
        "ColdStartRate",
        "ProvisionedConcurrency",
        "GBSeconds",
        "DataTransferGB",
        "CostUSD",
    ]
    df_[numeric_cols] = df_[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df_


df = load_data("Serverless_Data.csv")

# --------------------------
# Sidebar filters
# --------------------------
st.sidebar.title("Filters")

# Environment filter
env_options = sorted(df["Environment"].dropna().unique())
selected_envs = st.sidebar.multiselect(
    "Environment",
    options=env_options,
    default=env_options,
)

# Cost range
min_cost, max_cost = float(df["CostUSD"].min()), float(df["CostUSD"].max())
cost_range = st.sidebar.slider(
    "Cost range (USD)",
    min_value=0.0,
    max_value=round(max_cost + 10, 1),
    value=(0.0, round(max_cost + 10, 1)),
)

# Invocation filter
min_inv, max_inv = int(df["InvocationsPerMonth"].min()), int(
    df["InvocationsPerMonth"].max()
)
inv_range = st.sidebar.slider(
    "Monthly invocations range",
    min_value=min_inv,
    max_value=max_inv,
    value=(min_inv, max_inv),
)

# Memory filter
min_mem, max_mem = int(df["MemoryMB"].min()), int(df["MemoryMB"].max())
mem_range = st.sidebar.slider(
    "Memory (MB) range",
    min_value=min_mem,
    max_value=max_mem,
    value=(min_mem, max_mem),
)

# Duration filter
min_dur, max_dur = int(df["AvgDurationMs"].min()), int(df["AvgDurationMs"].max())
dur_range = st.sidebar.slider(
    "Avg duration (ms) range",
    min_value=min_dur,
    max_value=max_dur,
    value=(min_dur, max_dur),
)

st.sidebar.markdown("---")



# Apply filters
filtered_df = df[
    (df["Environment"].isin(selected_envs))
    & (df["CostUSD"].between(cost_range[0], cost_range[1]))
    & (df["InvocationsPerMonth"].between(inv_range[0], inv_range[1]))
    & (df["MemoryMB"].between(mem_range[0], mem_range[1]))
    & (df["AvgDurationMs"].between(dur_range[0], dur_range[1]))
].copy()

# Safety to avoid divide-by-zero later
filtered_df = filtered_df.fillna(0)

# --------------------------
# Top-level KPIs
# --------------------------
st.title("Serverless FinOps Dashboard")

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

total_cost = filtered_df["CostUSD"].sum()
total_functions = len(filtered_df)
avg_cost = total_cost / total_functions if total_functions > 0 else 0
total_invocations = filtered_df["InvocationsPerMonth"].sum()

col_kpi1.metric("Total Cost (Filtered)", f"${total_cost:,.2f}")
col_kpi2.metric("Functions (Filtered)", total_functions)
col_kpi3.metric("Avg Cost per Function", f"${avg_cost:,.2f}")
col_kpi4.metric("Total Invocations (Filtered)", f"{total_invocations:,.0f}")

# --------------------------
# Tabs for Exercises 1–6
# --------------------------
(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
) = st.tabs(
    [
        "Top Cost Contributors",
        "Memory Right-Sizing",
        "Provisioned Concurrency",
        "Unused / Low-Value Workloads",
        "Cost Forecasting",
        "Containerization Candidates",
    ]
)

# --------------------------
# Exercise 1
# --------------------------
with tab1:
    st.subheader("Identify Top Cost Contributors")

    if filtered_df.empty:
        st.warning("No data after filters. Try relaxing the filters on the left.")
    else:
        df_sorted = filtered_df.sort_values("CostUSD", ascending=False).reset_index(
            drop=True
        )
        total_cost_filtered = df_sorted["CostUSD"].sum()
        df_sorted["CumCost"] = df_sorted["CostUSD"].cumsum()
        df_sorted["CumCostPct"] = df_sorted["CumCost"] / total_cost_filtered * 100

        top_80 = df_sorted[df_sorted["CumCostPct"] <= 80]

        st.markdown(
            f"- **Total Cost (Filtered):** `${total_cost_filtered:,.2f}`  \n"
            f"- **Functions contributing ~80% of cost:** `{len(top_80)}`"
        )

        st.markdown("**Top cost contributors (up to ~80% of spend):**")
        st.dataframe(
            top_80[
                ["FunctionName", "Environment", "CostUSD", "CumCostPct"]
            ].reset_index(drop=True)
        )

        st.markdown("---")
        st.markdown("#### Cost by Function (Top N)")

        top_n = st.slider("Show top N functions by cost", 5, 40, 20, key="ex1_topn")

        st.bar_chart(
            df_sorted.head(top_n).set_index("FunctionName")["CostUSD"],
            height=400,
        )

        st.markdown("---")
        st.markdown("#### Cost vs Invocation Frequency")

        st.caption(
            "This scatter helps highlight functions that are unusually expensive "
            "for their invocation volume."
        )

        scatter_df = filtered_df[
            ["FunctionName", "InvocationsPerMonth", "CostUSD"]
        ].copy()

        st.scatter_chart(
            scatter_df,
            x="InvocationsPerMonth",
            y="CostUSD",
            height=400,
        )

# --------------------------
# Exercise 2 – Memory Right-Sizing
# --------------------------
with tab2:
    st.subheader("Memory Right-Sizing")

    st.write(
        "We look for functions where **duration is low but memory is high**, "
        "indicating potential over-provisioning."
    )

    if filtered_df.empty:
        st.warning("No data after filters. Try relaxing the filters on the left.")
    else:
        st.markdown("##### Right-Sizing Parameters")

        col_thr1, col_thr2 = st.columns(2)
        with col_thr1:
            max_duration_ms = st.number_input(
                "Max 'low' duration (ms)",
                min_value=0,
                value=500,
                step=50,
                help="Functions faster than this are considered short-running.",
            )
        with col_thr2:
            min_memory_mb = st.number_input(
                "Min 'high' memory (MB)",
                min_value=128,
                value=1024,
                step=128,
                help="Functions with memory >= this are considered high-memory.",
            )

        candidates = filtered_df[
            (filtered_df["AvgDurationMs"] <= max_duration_ms)
            & (filtered_df["MemoryMB"] >= min_memory_mb)
        ].copy()

        if candidates.empty:
            st.info(
                "No clear right-sizing candidates with the current thresholds. "
                "Try adjusting duration/memory thresholds."
            )
        else:
            st.markdown("##### Suggested Memory Reduction")

            # Simple heuristic: cut memory in half, but not below 256 MB
            candidates["SuggestedMemoryMB"] = (
                (candidates["MemoryMB"] / 2).clip(lower=256).round().astype(int)
            )

            # Approximate cost impact: cost is roughly linear in memory for same duration & invocations
            candidates["EstimatedNewCostUSD"] = (
                candidates["CostUSD"] * candidates["SuggestedMemoryMB"] / candidates["MemoryMB"]
            )
            candidates["EstimatedSavingsUSD"] = (
                candidates["CostUSD"] - candidates["EstimatedNewCostUSD"]
            )

            show_cols = [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "AvgDurationMs",
                "MemoryMB",
                "SuggestedMemoryMB",
                "CostUSD",
                "EstimatedNewCostUSD",
                "EstimatedSavingsUSD",
            ]

            st.dataframe(
                candidates[show_cols]
                .sort_values("EstimatedSavingsUSD", ascending=False)
                .reset_index(drop=True)
            )

            total_savings = candidates["EstimatedSavingsUSD"].sum()
            st.success(
                f"Estimated potential savings from these right-sizing candidates: "
                f"**${total_savings:,.2f} per month** (approximate)."
            )

# --------------------------
# Exercise 3 – Provisioned Concurrency Optimization
# --------------------------
with tab3:
    st.subheader("Provisioned Concurrency Optimization")

    st.write(
        "We compare **cold start rate** vs **provisioned concurrency** to identify "
        "functions where PC might be overused."
    )

    pc_df = filtered_df[filtered_df["ProvisionedConcurrency"] > 0].copy()

    if pc_df.empty:
        st.info("No functions with Provisioned Concurrency > 0 in the filtered data.")
    else:
        cold_start_threshold = st.slider(
            "Cold start rate threshold (%) below which PC is considered excessive",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )

        # classify
        pc_df["PC_OptimizationFlag"] = pc_df.apply(
            lambda row: (
                "Candidate to Reduce/Remove PC"
                if row["ColdStartRate"] <= cold_start_threshold
                else "Likely Justified"
            ),
            axis=1,
        )

        st.markdown("##### Functions with Provisioned Concurrency")
        st.dataframe(
            pc_df[
                [
                    "FunctionName",
                    "Environment",
                    "InvocationsPerMonth",
                    "ColdStartRate",
                    "ProvisionedConcurrency",
                    "CostUSD",
                    "PC_OptimizationFlag",
                ]
            ]
            .sort_values("CostUSD", ascending=False)
            .reset_index(drop=True)
        )

        st.markdown("---")
        st.markdown("#### Cold Starts vs Provisioned Concurrency")

        st.caption(
            "Points in the lower-right region (low cold start rate, high PC) "
            "are strong candidates to reduce PC."
        )

        st.scatter_chart(
            pc_df,
            x="ProvisionedConcurrency",
            y="ColdStartRate",
            height=400,
        )

# --------------------------
# Exercise 4 – Detect Unused / Low-Value Workloads
# --------------------------
with tab4:
    st.subheader("Detect Unused or Low-Value Workloads")

    if filtered_df.empty:
        st.warning("No data after filters. Try relaxing the filters on the left.")
    else:
        total_inv = filtered_df["InvocationsPerMonth"].sum()
        filtered_df["InvocationSharePct"] = (
            filtered_df["InvocationsPerMonth"] / total_inv * 100
            if total_inv > 0
            else 0
        )

        st.markdown(
            "We look for any function with **< 1% of total invocations** but **non-trivial cost**."
        )

        min_cost_unused = st.number_input(
            "Minimum monthly cost threshold for 'low-value' candidates (USD)",
            min_value=0.0,
            value=10.0,
            step=5.0,
        )

        unused_candidates = filtered_df[
            (filtered_df["InvocationSharePct"] < 1.0)
            & (filtered_df["CostUSD"] >= min_cost_unused)
        ].copy()

        if unused_candidates.empty:
            st.info(
                "No low-invocation, high-cost candidates found with the current thresholds."
            )
        else:
            st.markdown("##### Potential Low-Value / Cleanup Candidates")
            st.dataframe(
                unused_candidates[
                    [
                        "FunctionName",
                        "Environment",
                        "InvocationsPerMonth",
                        "InvocationSharePct",
                        "CostUSD",
                    ]
                ]
                .sort_values("CostUSD", ascending=False)
                .reset_index(drop=True)
            )

            total_unused_cost = unused_candidates["CostUSD"].sum()
            st.warning(
                f"These functions represent **${total_unused_cost:,.2f} per month** "
                f"in potentially low-value workloads."
            )

# --------------------------
# Exercise 5 – Cost Forecasting
# --------------------------
with tab5:
    st.subheader("Cost Forecasting Model")

    st.write(
        "We use a simple model based on the activity formula:\n"
        "`Cost ≈ Invocations × Duration × Memory × PricingCoefficients`\n\n"
        "Here we estimate a per-function coefficient from the existing data, and "
        "then forecast cost for changed traffic, duration, and memory."
    )

    if filtered_df.empty:
        st.warning("No data after filters. Try relaxing the filters on the left.")
    else:
        st.markdown("##### Choose a baseline function")

        func_names = filtered_df["FunctionName"].unique().tolist()
        selected_func_name = st.selectbox(
            "Baseline function",
            options=func_names,
        )

        base_row = filtered_df[filtered_df["FunctionName"] == selected_func_name].iloc[0]

        st.markdown("**Baseline metrics (from dataset):**")
        colb1, colb2, colb3, colb4 = st.columns(4)
        colb1.metric("Invocations", f"{int(base_row['InvocationsPerMonth']):,}")
        colb2.metric("Avg Duration (ms)", f"{base_row['AvgDurationMs']:.0f}")
        colb3.metric("Memory (MB)", f"{base_row['MemoryMB']:.0f}")
        colb4.metric("Current Cost (USD)", f"${base_row['CostUSD']:.2f}")

        # compute GB-seconds for baseline
        base_inv = base_row["InvocationsPerMonth"]
        base_dur_s = base_row["AvgDurationMs"] / 1000
        base_mem_gb = base_row["MemoryMB"] / 1024

        base_gb_seconds = base_inv * base_dur_s * base_mem_gb
        if base_gb_seconds > 0:
            price_per_gb_second = base_row["CostUSD"] / base_gb_seconds
        else:
            price_per_gb_second = 0.00001667  # fallback typical Lambda price

        st.markdown("---")
        st.markdown("##### Adjust traffic and configuration")

        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            inv_multiplier = st.slider(
                "Invocation multiplier",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="1.0 means same traffic, 2.0 means double traffic, etc.",
            )
        with colf2:
            duration_multiplier = st.slider(
                "Duration multiplier",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Effect of code optimization or heavier logic.",
            )
        with colf3:
            memory_mb_new = st.number_input(
                "New memory (MB)",
                min_value=128,
                max_value=10240,
                value=int(base_row["MemoryMB"]),
                step=128,
            )

        # forecast
        new_inv = base_inv * inv_multiplier
        new_dur_s = base_dur_s * duration_multiplier
        new_mem_gb = memory_mb_new / 1024

        new_gb_seconds = new_inv * new_dur_s * new_mem_gb
        new_cost = new_gb_seconds * price_per_gb_second

        colr1, colr2 = st.columns(2)
        colr1.metric("Forecasted Cost (USD)", f"${new_cost:,.2f}")
        colr2.metric(
            "Cost Change vs Baseline",
            f"${(new_cost - base_row['CostUSD']):,.2f}",
        )

        st.caption(
            "Note: This is an approximate model based mainly on compute cost; "
            "request charges and data-transfer are not modeled in detail."
        )

# --------------------------
# Exercise 6 – Containerization Candidates
# --------------------------
with tab6:
    st.subheader("Spot Workloads for Containerization")

    st.write(
        "We identify long-running, high-memory, low-frequency functions that may be "
        "better suited to containers (e.g., ECS/Fargate) instead of pure serverless."
    )

    if filtered_df.empty:
        st.warning("No data after filters. Try relaxing the filters on the left.")
    else:
        st.markdown("##### Containerization Criteria")

        colc1, colc2, colc3 = st.columns(3)
        with colc1:
            dur_threshold = st.number_input(
                "Min duration for container candidate (ms)",
                min_value=0,
                value=3000,
                step=100,
            )
        with colc2:
            mem_threshold = st.number_input(
                "Min memory for container candidate (MB)",
                min_value=128,
                value=2048,
                step=128,
            )
        with colc3:
            max_inv_threshold = st.number_input(
                "Max monthly invocations for container candidate",
                min_value=0,
                value=5000,
                step=100,
            )

        cont_candidates = filtered_df[
            (filtered_df["AvgDurationMs"] >= dur_threshold)
            & (filtered_df["MemoryMB"] >= mem_threshold)
            & (filtered_df["InvocationsPerMonth"] <= max_inv_threshold)
        ].copy()

        if cont_candidates.empty:
            st.info(
                "No containerization candidates found with the current thresholds."
            )
        else:
            st.markdown("##### Potential Containerization Candidates")
            st.dataframe(
                cont_candidates[
                    [
                        "FunctionName",
                        "Environment",
                        "InvocationsPerMonth",
                        "AvgDurationMs",
                        "MemoryMB",
                        "CostUSD",
                    ]
                ]
                .sort_values("CostUSD", ascending=False)
                .reset_index(drop=True)
            )

            total_cont_cost = cont_candidates["CostUSD"].sum()
            st.warning(
                f"These candidates represent **${total_cont_cost:,.2f} per month** "
                f"in long-running, high-memory compute that may be cheaper on containers."
            )
