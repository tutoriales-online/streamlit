import base64
import time
from datetime import datetime, timedelta
from io import BytesIO

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Advanced Streamlit Features", page_icon="📊", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .card {
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "notes" not in st.session_state:
    st.session_state.notes = []
if "last_updated" not in st.session_state:
    st.session_state.last_updated = datetime.now()
if "data" not in st.session_state:
    # Generate fake sales data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    products = ["Product A", "Product B", "Product C", "Product D"]
    regions = ["North", "South", "East", "West"]

    data = []
    for date in dates:
        for product in products:
            for region in regions:
                sales = np.random.randint(10, 100)
                profit = sales * np.random.uniform(0.1, 0.3)
                data.append({"date": date, "product": product, "region": region, "sales": sales, "profit": profit})

    st.session_state.data = pd.DataFrame(data)

# Sidebar
with st.sidebar:
    st.title("Dashboard Controls")

    # Date filter
    st.subheader("Date Range")
    min_date = st.session_state.data["date"].min().date()
    max_date = st.session_state.data["date"].max().date()
    date_range = st.date_input("Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = st.session_state.data[
            (st.session_state.data["date"].dt.date >= start_date) & (st.session_state.data["date"].dt.date <= end_date)
        ]
    else:
        filtered_data = st.session_state.data.copy()

    # Filters
    st.subheader("Filters")
    selected_products = st.multiselect(
        "Select Products",
        options=st.session_state.data["product"].unique(),
        default=st.session_state.data["product"].unique(),
    )

    selected_regions = st.multiselect(
        "Select Regions",
        options=st.session_state.data["region"].unique(),
        default=st.session_state.data["region"].unique(),
    )

    # Apply filters
    filtered_data = filtered_data[
        (filtered_data["product"].isin(selected_products)) & (filtered_data["region"].isin(selected_regions))
    ]

    # Session state demo
    st.subheader("Session State Demo")
    st.write(f"Counter: {st.session_state.counter}")

    if st.button("Increment Counter"):
        st.session_state.counter += 1
        st.rerun()

    # Add notes
    st.subheader("Dashboard Notes")
    note = st.text_input("Add a note")
    if st.button("Save Note"):
        if note:
            st.session_state.notes.append({"text": note, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            st.success("Note added!")

# Main content
st.markdown("<h1 class='main-header'>Advanced Streamlit Dashboard</h1>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Detailed Analysis", "Data Explorer", "Advanced Features"])

with tab1:
    st.markdown("<h2 class='subheader'>Sales Overview</h2>", unsafe_allow_html=True)

    # KPI metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        total_sales = filtered_data["sales"].sum()
        st.markdown(f"<div class='metric-value'>${total_sales:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Sales</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        total_profit = filtered_data["profit"].sum()
        st.markdown(f"<div class='metric-value'>${total_profit:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Profit</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        profit_margin = (total_profit / total_sales) * 100
        st.markdown(f"<div class='metric-value'>{profit_margin:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Profit Margin</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        avg_daily_sales = filtered_data.groupby("date")["sales"].sum().mean()
        st.markdown(f"<div class='metric-value'>${avg_daily_sales:.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg. Daily Sales</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Time series chart
    st.subheader("Sales Trend")

    # Use caching for expensive operations
    @st.cache_data
    def prepare_time_series(data):
        daily_data = data.groupby("date").agg({"sales": "sum", "profit": "sum"}).reset_index()
        return daily_data

    daily_data = prepare_time_series(filtered_data)

    # Create interactive time series with Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_data["date"], y=daily_data["sales"], mode="lines", name="Sales", line=dict(color="#1E88E5", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_data["date"],
            y=daily_data["profit"],
            mode="lines",
            name="Profit",
            line=dict(color="#43A047", width=2),
        )
    )
    fig.update_layout(
        title="Daily Sales and Profit",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        legend=dict(orientation="h", y=1.1),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Product and region breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sales by Product")
        product_sales = filtered_data.groupby("product")["sales"].sum().reset_index()
        fig = px.pie(
            product_sales, values="sales", names="product", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sales by Region")
        region_sales = filtered_data.groupby("region")["sales"].sum().reset_index()
        fig = px.bar(
            region_sales, x="region", y="sales", color="region", color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h2 class='subheader'>Detailed Analysis</h2>", unsafe_allow_html=True)

    # Heat map of sales by product and region
    st.subheader("Sales Heatmap: Product vs Region")
    heatmap_data = filtered_data.groupby(["product", "region"])["sales"].sum().reset_index()
    heatmap_data_pivot = heatmap_data.pivot(index="product", columns="region", values="sales")

    fig = px.imshow(
        heatmap_data_pivot,
        labels=dict(x="Region", y="Product", color="Sales"),
        x=heatmap_data_pivot.columns,
        y=heatmap_data_pivot.index,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly analysis
    st.subheader("Monthly Performance")

    # Prepare monthly data
    filtered_data["month"] = filtered_data["date"].dt.strftime("%Y-%m")
    monthly_data = filtered_data.groupby("month").agg({"sales": "sum", "profit": "sum"}).reset_index()

    # Calculate growth rates
    monthly_data["sales_growth"] = monthly_data["sales"].pct_change() * 100
    monthly_data["profit_growth"] = monthly_data["profit"].pct_change() * 100

    # Create a multi-metric visualization
    fig = go.Figure()

    # Add bars for sales
    fig.add_trace(go.Bar(x=monthly_data["month"], y=monthly_data["sales"], name="Sales", marker_color="#1E88E5"))

    # Add line for profit
    fig.add_trace(
        go.Scatter(
            x=monthly_data["month"],
            y=monthly_data["profit"],
            name="Profit",
            mode="lines+markers",
            marker=dict(color="#FF8F00", size=8),
            line=dict(color="#FF8F00", width=2),
            yaxis="y2",
        )
    )

    # Layout with dual y-axes
    fig.update_layout(
        title="Monthly Sales and Profit",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Sales ($)", side="left", showgrid=False),
        yaxis2=dict(title="Profit ($)", side="right", overlaying="y", showgrid=False),
        legend=dict(orientation="h", y=1.1),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Growth rates
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sales Growth Rate (%)")

        sales_growth_chart = (
            alt.Chart(monthly_data)
            .mark_bar()
            .encode(
                x=alt.X("month:O", title="Month"),
                y=alt.Y("sales_growth:Q", title="Growth (%)"),
                color=alt.condition(
                    alt.datum.sales_growth > 0,
                    alt.value("#4CAF50"),  # positive - green
                    alt.value("#F44336"),  # negative - red
                ),
                tooltip=["month", "sales_growth"],
            )
            .properties(height=300)
        )

        st.altair_chart(sales_growth_chart, use_container_width=True)

    with col2:
        st.subheader("Profit Growth Rate (%)")

        profit_growth_chart = (
            alt.Chart(monthly_data)
            .mark_bar()
            .encode(
                x=alt.X("month:O", title="Month"),
                y=alt.Y("profit_growth:Q", title="Growth (%)"),
                color=alt.condition(
                    alt.datum.profit_growth > 0,
                    alt.value("#4CAF50"),  # positive - green
                    alt.value("#F44336"),  # negative - red
                ),
                tooltip=["month", "profit_growth"],
            )
            .properties(height=300)
        )

        st.altair_chart(profit_growth_chart, use_container_width=True)

with tab3:
    st.markdown("<h2 class='subheader'>Data Explorer</h2>", unsafe_allow_html=True)

    # Data table with search
    st.subheader("Raw Data")
    search_term = st.text_input("Search in data")

    if search_term:
        # Search across all columns
        mask = np.column_stack(
            [
                filtered_data[col].astype(str).str.contains(search_term, case=False, na=False)
                for col in filtered_data.columns
            ]
        )
        search_results = filtered_data[mask.any(axis=1)]
        st.dataframe(search_results, use_container_width=True)
    else:
        st.dataframe(filtered_data, use_container_width=True)

    # Download options
    st.subheader("Download Data")
    col1, col2 = st.columns(2)

    with col1:
        # CSV download
        def get_csv_download():
            csv = filtered_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sales_data.csv">Download CSV File</a>'
            return href

        st.markdown(get_csv_download(), unsafe_allow_html=True)

    with col2:
        # Excel download
        def get_excel_download():
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                filtered_data.to_excel(writer, sheet_name="Sales Data", index=False)
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="sales_data.xlsx">Download Excel File</a>'
            return href

        st.markdown(get_excel_download(), unsafe_allow_html=True)

    # Custom data aggregation
    st.subheader("Custom Data Aggregation")

    col1, col2, col3 = st.columns(3)

    with col1:
        group_by = st.selectbox("Group by", ["product", "region", "date"])

    with col2:
        metrics = st.multiselect("Select metrics", ["sales", "profit"], default=["sales", "profit"])

    with col3:
        agg_func = st.selectbox("Aggregation function", ["sum", "mean", "min", "max", "count"])

    if group_by and metrics and agg_func:
        # Handle date grouping specially
        if group_by == "date":
            time_unit = st.selectbox("Time unit", ["day", "week", "month", "quarter", "year"])

            if time_unit == "day":
                filtered_data["group_col"] = filtered_data["date"].dt.date
            elif time_unit == "week":
                filtered_data["group_col"] = filtered_data["date"].dt.isocalendar().week
            elif time_unit == "month":
                filtered_data["group_col"] = filtered_data["date"].dt.strftime("%Y-%m")
            elif time_unit == "quarter":
                filtered_data["group_col"] = filtered_data["date"].dt.to_period("Q").astype(str)
            else:  # year
                filtered_data["group_col"] = filtered_data["date"].dt.year

            agg_data = filtered_data.groupby("group_col")[metrics].agg(agg_func).reset_index()
            agg_data.columns = [time_unit if col == "group_col" else f"{agg_func}_{col}" for col in agg_data.columns]
        else:
            agg_data = filtered_data.groupby(group_by)[metrics].agg(agg_func).reset_index()
            agg_data.columns = [group_by if col == group_by else f"{agg_func}_{col}" for col in agg_data.columns]

        st.dataframe(agg_data, use_container_width=True)

        # Visualization of aggregated data
        st.subheader("Visualization of Aggregated Data")

        if len(metrics) > 0:
            metric_cols = [col for col in agg_data.columns if col != group_by and col != "group_col"]

            if group_by == "date":
                x_col = time_unit
            else:
                x_col = group_by

            fig = px.bar(
                agg_data, x=x_col, y=metric_cols, barmode="group", color_discrete_sequence=px.colors.qualitative.G10
            )

            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("<h2 class='subheader'>Advanced Features</h2>", unsafe_allow_html=True)

    # Progress bar and spinners
    st.subheader("Progress Bars and Spinners")

    if st.button("Run Process with Progress Bar"):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.success("Process completed!")

    # Spinner example
    if st.button("Run Process with Spinner"):
        with st.spinner("Processing data..."):
            time.sleep(2)  # Simulating work
        st.success("Done!")

    # Interactive forms
    st.subheader("Interactive Forms")

    with st.form("data_entry_form"):
        st.write("Enter new data point")
        col1, col2 = st.columns(2)

        with col1:
            new_product = st.selectbox("Product", st.session_state.data["product"].unique())
            new_region = st.selectbox("Region", st.session_state.data["region"].unique())

        with col2:
            new_date = st.date_input("Date", datetime.now())
            new_sales = st.number_input("Sales", min_value=0, value=50)
            new_profit = st.number_input("Profit", min_value=0.0, value=15.0)

        submit_button = st.form_submit_button("Add Data Point")

        if submit_button:
            new_data = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp(new_date),
                        "product": new_product,
                        "region": new_region,
                        "sales": new_sales,
                        "profit": new_profit,
                    }
                ]
            )

            st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
            st.success("Data point added successfully!")

    # Notes display
    st.subheader("Dashboard Notes")

    if st.session_state.notes:
        for i, note in enumerate(st.session_state.notes):
            with st.expander(f"Note {i+1} - {note['timestamp']}"):
                st.write(note["text"])
                if st.button(f"Delete Note {i+1}"):
                    st.session_state.notes.pop(i)
                    st.rerun()
    else:
        st.info("No notes yet. Add notes from the sidebar.")

    # File uploader
    st.subheader("File Uploader")
    uploaded_file = st.file_uploader("Upload a CSV file to append to the dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(upload_df.head())

            if st.button("Append Data"):
                # Validate columns
                required_cols = ["date", "product", "region", "sales", "profit"]
                if all(col in upload_df.columns for col in required_cols):
                    # Convert date column
                    upload_df["date"] = pd.to_datetime(upload_df["date"])

                    # Append to session state data
                    st.session_state.data = pd.concat([st.session_state.data, upload_df], ignore_index=True)
                    st.success(f"Successfully added {len(upload_df)} rows to the dataset!")
                else:
                    missing_cols = [col for col in required_cols if col not in upload_df.columns]
                    st.error(f"Upload failed. Missing columns: {', '.join(missing_cols)}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Expiring cache demo
    st.subheader("Cache Expiration Demo")

    # Function with expiring cache
    @st.cache_data(ttl=10)  # Cache expires after 10 seconds
    def get_random_data():
        return {"random_value": np.random.randint(1, 100), "timestamp": datetime.now().strftime("%H:%M:%S")}

    random_data = get_random_data()

    st.write(f"Random Value (refreshes every 10 seconds): {random_data['random_value']}")
    st.write(f"Generated at: {random_data['timestamp']}")
    st.write("Wait 10 seconds and refresh to see a new value.")

    # Last update info
    st.sidebar.markdown("---")
    st.sidebar.write(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.sidebar.button("Refresh Dashboard"):
        st.session_state.last_updated = datetime.now()
        st.rerun()
