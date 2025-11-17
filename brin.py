import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import numpy as np


data = pd.read_excel("Rekap Data BRIN NEW.xlsx")
# Pilih variabel independen (X) dan dependen (y)
X = data[[
    'Total Population',
    'Total Population Aged 0-4 Years',
    'Forest and Water Areas (1000 Hectare)',
    'Average Temperature',
    'Average Rainfall (mm)',
    'Total Days of Rain',
    'Average Humidity',
    'Average Solar Radiation',
    'Average Elevation'
]]

y = data['Total Malaria Cases']

# === 3. Split data 70:30 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === 4. Standardisasi fitur ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_test_s  = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

# === 5. Model SVR dengan parameter hasil PSO ===
svr_model = SVR(
    kernel="rbf",
    C=80,
    epsilon=0.01,
    gamma=0.1
)

svr_model.fit(X_train_s, y_train_s)
st.set_page_config(
    page_title="PREDIKSI PENYAKIT MALARIA",
    page_icon="📊",
)

st.markdown("""
    <style>
    .title {
        font-size: 30px;
        color: #000000;
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 20px;
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    </style>
    <h1 class="title">DASHBOARD MALARIA DISEASE</h1>
    """, unsafe_allow_html=True)

# Sidebar with option menu
with st.sidebar:
    selected = option_menu(
        menu_title="DASHBOARD",
        options=["Home", "Visualization", "Prediction"], 
    )

# CSS for gradient background
gradient_style = """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #76c893, #ffd700);
        height: 100vh;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #f0f0f0;  /* Optional: change sidebar background */
    }
    </style>
"""
st.markdown(gradient_style, unsafe_allow_html=True)

# If the user selects Home
# Home Page Content
if selected == "Home":
    st.markdown("""
        <style>
        .shimmer {
            font-size: 40px;  /* Membuat teks lebih besar */
            font-weight: 900;  /* Membuat teks lebih tebal */
            text-align: center;
            color: black;  /* Warna teks hitam solid */
            position: relative;
        }

        .shimmer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0) 100%);
            background-size: 200% 200%;
            animation: shimmer 4s infinite;
        }

        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        </style>

        <h1 class='shimmer'>Welcome to The Dashboard</h1>

        <marquee behavior="scroll" direction="left" scrollamount="10" style="font-size:24px; color:black;">
            This Dashboard Provides Malaria Disease Prediction
        </marquee>
    """, unsafe_allow_html=True)

    # Menambahkan gambar di bawah teks marquee tanpa efek
    st.image("logo.jpg", use_container_width=True)  
# Kondisi jika pengguna memilih Visualisasi
elif selected == "Visualization":
    graph_type = st.selectbox(
        "Pilih Jenis Grafik", 
        ["Box Plot", "Heatmap of Corellation", "Line Chart", "Stacked Bar Chart", "HeatMap"],
        key="graph_type"
    )
        # Applying the custom class to the title
    st.markdown('<h1 class="centered-title">Data Visualization</h1>', unsafe_allow_html=True)
    
 
    if graph_type == "Box Plot":
        # === 1️⃣ Konversi tanggal dan ambil tahun & bulan ===
        data["Year-Month"] = pd.to_datetime(data["Year-Month"], format="%d/%m/%Y", errors="coerce")
        data["Year"] = data["Year-Month"].dt.year
        data["Month"] = data["Year-Month"].dt.month

        # === 2️⃣ Tentukan kolom numerik ===
        exclude_cols = ["Province", "Year-Month", "Year", "Month", "Longitude", "Latitude"]
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

        fitur_boxplot = st.selectbox(
            "Select Feature for Box Plot",
            numeric_cols,
            key="boxplot_feature"
        )

        # === 3️⃣ Tambahkan opsi 'Indonesia' ===
        provinsi_list = ["Indonesia"] + sorted(data["Province"].dropna().unique().tolist())
        provinsi_pilihan = st.selectbox("Select Province", provinsi_list, key="boxplot_province")

        # === 4️⃣ Pilih Tahun ===
        tahun_list = sorted(data["Year"].dropna().unique())
        tahun_pilihan = st.selectbox("Select Year", tahun_list, key="boxplot_year")

        # === 5️⃣ Pilih Bulan (opsional, boleh kosong) ===
        bulan_dict = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }

        bulan_list = sorted(data["Month"].dropna().unique())
        bulan_pilihan = st.multiselect(
            "Select Month (leave empty to show all months)",
            bulan_list,
            format_func=lambda x: bulan_dict.get(x, str(x)),
            key="boxplot_month"
        )

        # === 6️⃣ Filter data ===
        if provinsi_pilihan == "Indonesia":
            # Semua provinsi dalam tahun (dan bulan jika dipilih)
            data_filtered = data[data["Year"] == tahun_pilihan]
            if bulan_pilihan:
                data_filtered = data_filtered[data_filtered["Month"].isin(bulan_pilihan)]
        else:
            # Provinsi spesifik
            data_filtered = data[data["Province"] == provinsi_pilihan]
            data_filtered = data_filtered[data_filtered["Year"] == tahun_pilihan]
            if bulan_pilihan:
                data_filtered = data_filtered[data_filtered["Month"].isin(bulan_pilihan)]

        # === 7️⃣ Tampilkan Box Plot ===
        if st.button("Show Box Plot", key="show_boxplot"):
            import matplotlib.pyplot as plt
            import seaborn as sns

            if data_filtered.empty:
                st.warning("No data available for the selected filters.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))

                if provinsi_pilihan == "Indonesia":
                    if bulan_pilihan:
                        # Distribusi antar provinsi untuk bulan tertentu
                        sns.boxplot(
                            x="Month",
                            y=fitur_boxplot,
                            data=data_filtered,
                            palette="coolwarm",
                            ax=ax
                        )
                        ax.set_xticklabels([bulan_dict[m] for m in sorted(data_filtered["Month"].unique())])
                        plt.title(
                            f"Distribution of {fitur_boxplot} Across Provinces — Selected Months {tahun_pilihan}",
                            fontsize=14
                        )
                    else:
                        # Distribusi antar provinsi sepanjang 12 bulan
                        sns.boxplot(
                            x="Month",
                            y=fitur_boxplot,
                            data=data_filtered,
                            palette="coolwarm",
                            ax=ax
                        )
                        ax.set_xticklabels([bulan_dict[m] for m in sorted(data_filtered["Month"].unique())])
                        plt.title(
                            f"Distribution of {fitur_boxplot} Across Provinces — All Months {tahun_pilihan}",
                            fontsize=14
                        )

                    plt.xlabel("Month", fontsize=12)
                    plt.ylabel(fitur_boxplot, fontsize=12)

                else:
                    # Distribusi antar bulan untuk provinsi tertentu
                    sns.boxplot(
                        x="Month",
                        y=fitur_boxplot,
                        data=data_filtered,
                        palette="Set2",
                        ax=ax
                    )
                    ax.set_xticklabels([bulan_dict[m] for m in sorted(data_filtered["Month"].unique())])
                    plt.title(f"Monthly Box Plot of {fitur_boxplot} — {provinsi_pilihan} ({tahun_pilihan})", fontsize=14)
                    plt.xlabel("Month", fontsize=12)
                    plt.ylabel(fitur_boxplot, fontsize=12)

                plt.grid(True, axis="y", linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)


    elif graph_type == "Heatmap of Corellation":
        if st.button("Tampilkan Heatmap Korelasi"):
            # === Kolom yang tidak dipakai ===
            exclude_cols = ["Provinsi", "Tahun", "Latitude", "Longitude"]

            # === Variabel yang digunakan di model SVR ===
            svr_vars = [
                "Total Malaria Cases",
                "Total Population", 
                "Forest and Water Areas (1000 Hectare)", 
                "Average Temperature", 
                "Average Rainfall (mm)", 
                "Total Days of Rain", 
                "Average Humidity", 
                "Average Solar Radiation", 
                "Average Elevation", 
                "Total Population Aged 0-4 Years"
            ]

            # === Filter data ===
            # Ambil kolom yang ada di data dan termasuk svr_vars, lalu buang yang dikecualikan
            used_cols = [col for col in svr_vars if col in data.columns and col not in exclude_cols]
            data_svr = data[used_cols].select_dtypes(include=["float64", "int64"])

            # === Hitung korelasi ===
            corr = data_svr.corr(method="pearson")  # bisa diganti spearman/kendall

            # === Plot heatmap ===
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm",
                cbar=True, square=True, ax=ax
            )
            plt.title("Heatmap of Correlation Variabel SVR")
            st.pyplot(fig)

    elif graph_type == "Line Chart":
        # === Daftar provinsi + opsi "Indonesia (Total)" ===
        provinsi_list = sorted(data["Province"].unique())
        provinsi_list = ["Indonesia (Total)"] + provinsi_list
        provinsi_pilihan = st.multiselect(
            "Pilih Provinsi", 
            provinsi_list, 
            key="line_prov"
        )

        # === Konversi kolom Year-Month ke datetime ===
        data["Year-Month"] = pd.to_datetime(data["Year-Month"], format="%d/%m/%Y", errors="coerce")

        # === Filter data sesuai pilihan provinsi ===
        if not provinsi_pilihan:
            data_filtered = data.copy()  # default semua provinsi
        else:
            if "Indonesia (Total)" in provinsi_pilihan:
                data_filtered = data.copy()
            else:
                data_filtered = data[data["Province"].isin(provinsi_pilihan)]

        if data_filtered.empty:
            st.warning("Data kosong untuk provinsi yang dipilih.")
        else:
            exclude_cols = ["Province", "Year-Month", "Latitude", "Longitude"]
            numeric_cols = [
                col for col in data_filtered.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]

            if len(numeric_cols) == 0:
                st.error("Dataset tidak memiliki variabel numerik untuk Line Chart.")
            else:
                var_line = st.selectbox("Pilih Variabel untuk Line Chart", numeric_cols, key="line_var")

                if st.button("Tampilkan Line Chart", key="show_line"):
                    import matplotlib.dates as mdates
                    import matplotlib.pyplot as plt
                    from functools import reduce

                    fig, ax = plt.subplots(figsize=(10, 6))

                    # --- Jika Indonesia (Total) dipilih ---
                    if "Indonesia (Total)" in provinsi_pilihan:
                        indo_data = (
                            data.groupby("Year-Month")[var_line]
                            .sum()
                            .sort_index()
                        )
                        ax.plot(
                            indo_data.index,
                            indo_data.values,
                            marker="o",
                            linewidth=3,
                            label="Indonesia (Total)",
                            color="black"
                        )

                    # --- Plot per provinsi yang dipilih ---
                    for prov in data_filtered["Province"].unique():
                        if prov not in provinsi_pilihan or prov == "Indonesia (Total)":
                            continue
                        line_data = (
                            data_filtered[data_filtered["Province"] == prov]
                            .groupby("Year-Month")[var_line]
                            .sum()
                            .sort_index()
                        )
                        ax.plot(line_data.index, line_data.values, marker="o", linewidth=2, label=prov)

                    # === Format tanggal dan tampilan ===
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    plt.xticks(rotation=45)
                    ax.set_title(f"Monthly Trend of {var_line}")
                    ax.set_xlabel("Month-Year")
                    ax.set_ylabel(var_line)
                    ax.legend(title="Province", bbox_to_anchor=(1.05, 1), loc="upper left")
                    ax.grid(True, linestyle='--', alpha=0.6)

                    # ✅ Tampilkan grafik
                    st.pyplot(fig)

                    # === 🔽 Tampilkan tabel nilai di bawah grafik ===
                    st.markdown("### 📊 Monthly Values Table")

                    if "Indonesia (Total)" in provinsi_pilihan:
                        # Gabungkan Indonesia (Total)
                        indo_table = (
                            data.groupby("Year-Month")[var_line]
                            .sum()
                            .reset_index()
                            .rename(columns={var_line: "Indonesia (Total)"})
                        )
                        # Gabungkan dengan provinsi lain yang dipilih
                        prov_tables = []
                        for prov in data_filtered["Province"].unique():
                            if prov not in provinsi_pilihan or prov == "Indonesia (Total)":
                                continue
                            tmp = (
                                data_filtered[data_filtered["Province"] == prov]
                                .groupby("Year-Month")[var_line]
                                .sum()
                                .reset_index()
                                .rename(columns={var_line: prov})
                            )
                            prov_tables.append(tmp)
                        # Gabungkan semua ke satu tabel
                        df_merge = reduce(lambda left, right: pd.merge(left, right, on="Year-Month", how="outer"), [indo_table] + prov_tables)
                    else:
                        # Hanya provinsi-provinsi terpilih
                        df_merge = (
                            data_filtered[data_filtered["Province"].isin(provinsi_pilihan)]
                            .pivot_table(
                                index="Year-Month",
                                columns="Province",
                                values=var_line,
                                aggfunc="sum"
                            )
                            .reset_index()
                        )

                    # Format tanggal jadi bulan-tahun
                    df_merge["Year-Month"] = df_merge["Year-Month"].dt.strftime("%b %Y")

                    # Format angka 2 desimal
                    st.dataframe(df_merge.style.format("{:,.2f}", subset=df_merge.columns[1:]))


    # === Stacked Bar Chart (Top 7) ===
    elif graph_type == "Stacked Bar Chart":
        exclude_cols = ["Province", "Year-Month", "Longitude", "Latitude"]
        numeric_cols = [
            col for col in data.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        if len(numeric_cols) == 0:
            st.error("No numeric variables available for Stacked Bar Chart!")
        else:
            fitur_stacked = st.selectbox(
                "Select Variable for Stacked Bar Chart",
                numeric_cols,
                key="stacked_var"
            )

            if st.button("Show Stacked Bar Chart", key="show_stacked"):
                # --- Pastikan kolom waktu dalam format datetime ---
                data["Year-Month"] = pd.to_datetime(data["Year-Month"], format="%d/%m/%Y", errors="coerce")
                data["Year"] = data["Year-Month"].dt.year  # ambil tahun

                # --- Ambil Top 7 Provinsi berdasarkan total fitur ---
                top7_prov = (
                    data.groupby("Province")[fitur_stacked]
                    .sum()
                    .nlargest(7)
                    .index
                )

                # --- Pivot tabel: Tahun x Provinsi ---
                pivot_df = (
                    data[data["Province"].isin(top7_prov)]
                    .pivot_table(
                        index="Year",
                        columns="Province",
                        values=fitur_stacked,
                        aggfunc="sum",
                        fill_value=0
                    )
                )

                # --- Urutkan kolom setiap tahun berdasarkan nilai dari kecil ke besar ---
                sorted_cols = (
                    pivot_df.apply(lambda x: x.sort_values(ascending=True).index, axis=1)
                    .iloc[-1]  # pakai urutan tahun terakhir sebagai acuan global
                )
                pivot_df = pivot_df[sorted_cols]

                # --- Buat Stacked Bar Chart ---
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.tab10.colors
                pivot_df.plot(
                    kind="bar",
                    stacked=True,
                    ax=ax,
                    color=colors[:len(top7_prov)]
                )

                plt.title(f"Top 7 Provinces by {fitur_stacked} (Stacked from Smallest to Largest per Year)")
                plt.xlabel("Year")
                plt.ylabel(fitur_stacked)
                plt.xticks(rotation=0)

                # --- Legend di luar ---
                ax.legend(
                    title="Province",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0
                )

                # --- Tampilkan grafik ---
                st.pyplot(fig)

                # --- Tampilkan tabel angka ---
                st.markdown("### 📊 Yearly Values (Top 7 Provinces, Sorted from Smallest to Largest)")
                styled_df = pivot_df.copy()
                styled_df.index = styled_df.index.astype(str)
                st.dataframe(styled_df.style.format("{:,.2f}", subset=styled_df.columns))




    # === HeatMap dengan Latitude & Longitude ===
    elif graph_type == "HeatMap":
        # ✅ Pastikan kolom koordinat ada
        if "Latitude" not in data.columns or "Longitude" not in data.columns:
            st.error("Kolom Latitude dan Longitude tidak ditemukan di dataset!")
        else:
            # Konversi tipe data koordinat ke numerik
            data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
            data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")

            # --- Pastikan kolom waktu dalam format datetime & ambil tahun ---
            data["Year-Month"] = pd.to_datetime(data["Year-Month"], format="%d/%m/%Y", errors="coerce")
            data["Year"] = data["Year-Month"].dt.year

            # --- Tentukan kolom numerik selain lat/lon dan identitas ---
            exclude_cols = ["Latitude", "Longitude", "Province", "Year-Month"]
            numeric_cols = [
                col for col in data.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]

            if len(numeric_cols) == 0:
                st.error("No numeric variables available for HeatMap visualization!")
            else:
                var_heatmap = st.selectbox("Select Variable for HeatMap", numeric_cols, key="heatmap_var")

                # === Filter Tahun ===
                tahun_list = sorted(data["Year"].dropna().unique())
                tahun_pilihan = st.selectbox("Select Year for HeatMap", tahun_list, key="heatmap_tahun")

                if st.button("Show HeatMap", key="show_heatmap"):
                    # --- Filter berdasarkan tahun ---
                    heat_data = data[data["Year"] == tahun_pilihan][
                        ["Latitude", "Longitude", "Province", var_heatmap]
                    ].dropna()

                    if heat_data.empty:
                        st.error(f"No data available for the year {tahun_pilihan}!")
                    else:
                        # --- Buat peta dasar ---
                        m = folium.Map(location=[-2.5, 118], zoom_start=5)

                        # --- Tambahkan HeatMap layer ---
                        HeatMap(
                            heat_data[["Latitude", "Longitude", var_heatmap]].values.tolist(),
                            radius=20,
                            blur=15,
                            max_zoom=6
                        ).add_to(m)

                        # --- Tambahkan label teks tanpa background tapi dengan efek shadow ---
                        for _, row in heat_data.iterrows():
                            folium.Marker(
                                location=[row["Latitude"], row["Longitude"]],
                                icon=folium.DivIcon(
                                    html=f"""
                                        <div style="
                                            font-size:10pt;
                                            font-weight:bold;
                                            color:#ffffff;
                                            text-shadow:
                                                -1px -1px 2px #000000,
                                                1px -1px 2px #000000,
                                                -1px 1px 2px #000000,
                                                1px 1px 2px #000000;
                                            text-align:center;
                                        ">
                                            {row["Province"]}<br>
                                            {row[var_heatmap]:,.0f}
                                        </div>
                                    """
                                )
                            ).add_to(m)

                        st.markdown(
                            f"### 🌍 HeatMap of Indonesia showing **{var_heatmap}** values by Province in **{tahun_pilihan}**"
                        )
                        st_folium(m, width=800, height=600, returned_objects=[])



# Kondisi jika pengguna memilih Prediksi
elif selected == "Prediction":

    import numpy as np
    from sklearn.metrics import mean_squared_error

    # ==========================
    # Precompute residual-based CI
    # ==========================
    # Prediksi training
    y_train_pred = svr_model.predict(X_train_s)
    y_train_pred_inv = scaler_y.inverse_transform(y_train_pred.reshape(-1,1)).ravel()
    y_train_true_inv = scaler_y.inverse_transform(y_train_s.reshape(-1,1)).ravel()

    # Residual training (error asli)
    residuals = y_train_true_inv - y_train_pred_inv

    # Standar deviasi residual (dipakai untuk CI)
    error_std = np.std(residuals)


    # ==========================
    # Style Section
    # ==========================
    st.markdown(
        """
        <style>
        .title-style {
            font-size: 32px;
            font-weight: bold;
            color: black;
            text-align: center;
            margin-bottom: 25px;
        }

        .stButton>button {
            background-color: grey;
            color: white;
            padding: 10px 50px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: grey;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="title-style">Prediction Section</h1>', unsafe_allow_html=True)

    # ==========================
    # Input Variables
    # ==========================
    st.subheader("Input Variables for SVR Prediction")

    total_population = st.number_input('Total Population', min_value=0, value=0)
    population_0_4 = st.number_input('Total Population Aged 0-4 Years', min_value=0, value=0)
    forest_water_area = st.number_input('Forest and Water Areas (1000 Hectare)', min_value=0.0, value=0.0)
    avg_temperature = st.number_input('Average Temperature', min_value=0.0, value=0.0)
    avg_rainfall = st.number_input('Average Rainfall (mm)', min_value=0.0, value=0.0)
    rainy_days = st.number_input('Total Days of Rain', min_value=0, value=0)
    avg_humidity = st.number_input('Average Humidity', min_value=0.0, value=0.0)
    avg_solar_radiation = st.number_input('Average Solar Radiation', min_value=0.0, value=0.0)
    avg_elevation = st.number_input('Average Elevation', min_value=0.0, value=0.0)

    new_data = pd.DataFrame({
        "Total Population": [total_population],
        "Total Population Aged 0-4 Years": [population_0_4],
        "Forest and Water Areas (1000 Hectare)": [forest_water_area],
        "Average Temperature": [avg_temperature],
        "Average Rainfall (mm)": [avg_rainfall],
        "Total Days of Rain": [rainy_days],
        "Average Humidity": [avg_humidity],
        "Average Solar Radiation": [avg_solar_radiation],
        "Average Elevation": [avg_elevation]
    })

    new_data_scaled = scaler_X.transform(new_data)

    # ==========================
    # Predict Button (INSTANT)
    # ==========================
    if st.button('Predict'):

        # Prediksi utama
        pred_scaled = svr_model.predict(new_data_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]

        # ==========================
        # Instant CI calculation (95%)
        # ==========================
        CI_lower = pred - 1.96 * error_std
        CI_upper = pred + 1.96 * error_std

        # ==========================
        # Output Prediksi
        # ==========================
        st.markdown(
            f"""
            <div style="
                background-color:#fafafa;
                border-radius:12px;
                padding:25px;
                margin-top:20px;
                box-shadow:0px 4px 15px rgba(0,0,0,0.1);
                text-align:center;
            ">
                <h2 style="color:#2c3e50;">🔮 Predicted Malaria Cases</h2>
                <p style="font-size:48px; font-weight:bold; color:#e74c3c; margin:0;">
                    {pred:,.0f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==========================
        # CI output
        # ==========================
        st.markdown(
            f"""
            <div style="
                background-color:#eef7ee;
                border-radius:12px;
                padding:25px;
                margin-top:25px;
                box-shadow:0px 4px 15px rgba(0,0,0,0.10);
                text-align:center;
            ">
                <h3 style="color:#2c3e50;">📏 95% Confidence Interval</h3>
                <p style="font-size:32px; font-weight:bold; color:#27ae60; margin:0;">
                    {CI_lower:,.0f} &mdash; {CI_upper:,.0f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


