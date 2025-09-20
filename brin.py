import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import numpy as np

data = pd.read_excel("Rekap Data BRIN_rev.xlsx",  sheet_name="Dashboard")
# Pilih variabel independen (X) dan dependen (y)
X = data[["Jumlah Penduduk", 
        "Luas hutan dan perairan (Ribu Hektar)", 
        "Rata-rata Suhu", 
        "Rata-rata curah hujan (mm)", 
        "Jumlah hari hujan", 
        "Rata-rata kelembapan", 
        "Rata-rata radiasi solar", 
        "Rata-rata Elevasi", 
        "Jumlah Penduduk 0-4 Tahun"]]

y = data["Jumlah penderita Malaria"]

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
    C=5.786482194922019, 
    epsilon=0.001, 
    gamma=0.6725901070568777
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
    st.image("logo.jpg", use_column_width=True)  
# Kondisi jika pengguna memilih Visualisasi
elif selected == "Visualization":
    graph_type = st.selectbox(
        "Pilih Jenis Grafik", 
        ["Box Plot", "Heatmap Korelasi", "Line Chart", "Stacked Bar Chart", "HeatMap"],
        key="graph_type"
    )
        # Applying the custom class to the title
    st.markdown('<h1 class="centered-title">Data Visualization</h1>', unsafe_allow_html=True)
    if graph_type == "Box Plot":
        # Buat daftar kolom yang tidak boleh dipakai sebagai fitur
        exclude_cols = ["Provinsi", "Tahun"]

        # Ambil hanya kolom numerik, lalu buang yang ada di exclude list
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

        fitur_boxplot = st.selectbox(
            'Pilih Fitur untuk Box Plot', 
            numeric_cols, 
            key='boxplot_fitur'
        )

        tahun_list = sorted(data["Tahun"].unique())
        tahun_pilihan = st.multiselect(
            "Pilih Tahun (kosongkan untuk semua tahun)", 
            tahun_list, 
            key='boxplot_tahun'
        )

        if st.button('Tampilkan Box Plot', key='show_boxplot'):
            data_filtered = data[data["Tahun"].isin(tahun_pilihan)] if tahun_pilihan else data
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(x="Tahun", y=fitur_boxplot, data=data_filtered)
            plt.title(f'Box Plot {fitur_boxplot} berdasarkan Tahun')
            st.pyplot(fig)

    # === Heatmap Korelasi ===
    elif graph_type == "Heatmap Korelasi":
        if st.button("Tampilkan Heatmap Korelasi"):
            # Ambil hanya kolom numerik
            corr = data.select_dtypes(include=["float64","int64"]).corr()

            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar=True, square=True, ax=ax
            )
            plt.title("Heatmap Korelasi antar Variabel")
            st.pyplot(fig)


    elif graph_type == "Line Chart":
        # Filter berdasarkan Provinsi
        provinsi_list = sorted(data["Provinsi"].unique())
        provinsi_pilihan = st.multiselect(
            "Pilih Provinsi", 
            provinsi_list, 
            key="line_prov"
        )

        # Jika ada provinsi dipilih → filter data
        data_filtered = data[data["Provinsi"].isin(provinsi_pilihan)] if provinsi_pilihan else data

        if data_filtered.empty:
            st.warning("Data kosong untuk provinsi yang dipilih.")
        else:
            exclude_cols = ["Provinsi", "Tahun", "Latitude", "Longitude"]
            numeric_cols = [col for col in data_filtered.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

            if len(numeric_cols) == 0:
                st.error("Dataset tidak memiliki variabel numerik untuk Line Chart.")
            else:
                var_line = st.selectbox("Pilih variabel untuk Line Chart", numeric_cols, key="line_var")

                if st.button("Tampilkan Line Chart", key="show_line"):
                    # === Buat Line Chart ===
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot tren per provinsi
                    for prov in data_filtered["Provinsi"].unique():
                        line_data = (
                            data_filtered[data_filtered["Provinsi"] == prov]
                            .groupby("Tahun")[var_line]
                            .sum()
                            .sort_index()
                        )
                        ax.plot(line_data.index, line_data.values, marker="o", linewidth=2, label=prov)

                        # Tambahkan label angka di dekat titik
                        for x, y in zip(line_data.index, line_data.values):
                            ax.text(x, y, f"{int(y):,}", fontsize=9, ha="center", va="bottom")

                    ax.set_title(f"Tren {var_line} per Tahun berdasarkan Provinsi")
                    ax.set_xlabel("Tahun")
                    ax.set_ylabel(var_line)
                    ax.legend(title="Provinsi", bbox_to_anchor=(1.05, 1), loc="upper left")

                    # Pastikan x-label bulat integer (2019, 2020, dst.)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                    # Tampilkan grafik
                    st.pyplot(fig)



# === Stacked Bar Chart (Top 7) ===
    elif graph_type == "Stacked Bar Chart":
        exclude_cols = ["Provinsi", "Tahun"]
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

        fitur_stacked = st.selectbox("Pilih Variabel untuk Stacked Bar Chart", numeric_cols, key="stacked_var")

        if st.button("Tampilkan Stacked Bar Chart"):
            # Ambil Top 7 provinsi
            top7_prov = (
                data.groupby("Provinsi")[fitur_stacked]
                .sum()
                .nlargest(7)
                .index
            )

            # Pivot tabel: Tahun x Provinsi
            pivot_df = data[data["Provinsi"].isin(top7_prov)].pivot_table(
                index="Tahun",
                columns="Provinsi",
                values=fitur_stacked,
                aggfunc="sum",
                fill_value=0
            )

            # Buat stacked bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10.colors  # ambil palet tab10
            pivot_df.plot(kind="bar", stacked=True, ax=ax, color=colors[:len(top7_prov)])

            plt.title(f"Top 7 Provinsi dengan {fitur_stacked} (Stacked per Tahun)")
            plt.xlabel("Tahun")
            plt.ylabel(fitur_stacked)
            plt.xticks(rotation=0)

            # Buat legend manual + label jumlah total provinsi
            prov_totals = pivot_df.sum(axis=0).sort_values(ascending=False)
            handles, labels = ax.get_legend_handles_labels()

            new_labels = []
            for prov, total in prov_totals.items():
                new_labels.append(f"{prov} ({int(total):,})")

            ax.legend(
                handles,
                new_labels,
                title="Provinsi (Total)",
                bbox_to_anchor=(1.05, 1),
                loc="upper left"
            )

            st.pyplot(fig)



    # === HeatMap dengan Latitude & Longitude ===
    elif graph_type == "HeatMap":
        if "Latitude" not in data.columns or "Longitude" not in data.columns:
            st.error("Kolom Latitude dan Longitude tidak ditemukan di dataset!")
        else:
            data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
            data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")

            exclude_cols = ["Latitude", "Longitude", "Provinsi", "Tahun"]
            numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

            if len(numeric_cols) == 0:
                st.error("Tidak ada variabel numerik untuk heatmap selain Latitude/Longitude!")
            else:
                var_heatmap = st.selectbox("Pilih variabel untuk HeatMap", numeric_cols, key="heatmap_var")

                # Tambahkan opsi filter tahun
                tahun_list = sorted(data["Tahun"].dropna().unique())
                tahun_pilihan = st.selectbox("Pilih Tahun untuk HeatMap", tahun_list, key="heatmap_tahun")

                if st.button("Tampilkan HeatMap", key="show_heatmap"):
                    # Filter data berdasarkan tahun terpilih
                    heat_data = data[data["Tahun"] == tahun_pilihan][
                        ["Latitude", "Longitude", "Provinsi", var_heatmap]
                    ].dropna()

                    if len(heat_data) == 0:
                        st.error(f"Tidak ada data untuk tahun {tahun_pilihan}!")
                    else:
                        m = folium.Map(location=[-2.5, 118], zoom_start=5)

                        # Tambahkan HeatMap
                        HeatMap(
                            heat_data[["Latitude", "Longitude", var_heatmap]].values.tolist(),
                            radius=20,
                            blur=15,
                            max_zoom=6
                        ).add_to(m)

                        # Tambahkan marker angka lebih estetik
                        for _, row in heat_data.iterrows():
                            folium.Marker(
                                location=[row["Latitude"], row["Longitude"]],
                                icon=folium.DivIcon(
                                    html=f"""
                                        <div style="
                                            font-size:9pt;
                                            color:#003366;
                                            text-shadow: 1px 1px 2px #ffffff;
                                            text-align:center;
                                        ">{int(row[var_heatmap])}</div>
                                    """
                                )
                            ).add_to(m)

                        st.markdown(f"### 🌍 HeatMap Indonesia berdasarkan variabel: **{var_heatmap}** ({tahun_pilihan})")
                        st_folium(m, width=800, height=600, returned_objects=[])

# Kondisi jika pengguna memilih Prediksi
elif selected == "Prediction":
        
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

        .center-button {
            display: flex;
            justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Enhanced Title
    st.markdown('<h1 class="title-style">Prediction Section</h1>', unsafe_allow_html=True)


        # === Input untuk prediksi ===
    jumlah_penduduk = st.number_input('Jumlah Penduduk', min_value=0, value=0, key='jumlah_penduduk')
    luas_hutan_air = st.number_input('Luas hutan & perairan (Ribu Hektar)', min_value=0.0, value=0.0, key='luas_hutan_air')
    rata_suhu = st.number_input('Rata-rata Suhu', min_value=0.0, value=0.0, key='rata_suhu')
    curah_hujan = st.number_input('Rata-rata Curah Hujan (mm)', min_value=0.0, value=0.0, key='curah_hujan')
    hari_hujan = st.number_input('Jumlah Hari Hujan', min_value=0, value=0, key='hari_hujan')
    kelembapan = st.number_input('Rata-rata Kelembapan', min_value=0.0, value=0.0, key='kelembapan')
    radiasi_solar = st.number_input('Rata-rata Radiasi Solar', min_value=0.0, value=0.0, key='radiasi_solar')
    elevasi = st.number_input('Rata-rata Elevasi', min_value=0.0, value=0.0, key='elevasi')
    penduduk_0_4 = st.number_input('Jumlah Penduduk 0-4 Tahun', min_value=0, value=0, key='penduduk_0_4')

    # === Data baru dalam DataFrame ===
    new_data = pd.DataFrame({
        "Jumlah Penduduk": [jumlah_penduduk],
        "Luas hutan dan perairan (Ribu Hektar)": [luas_hutan_air],
        "Rata-rata Suhu": [rata_suhu],
        "Rata-rata curah hujan (mm)": [curah_hujan],
        "Jumlah hari hujan": [hari_hujan],
        "Rata-rata kelembapan": [kelembapan],
        "Rata-rata radiasi solar": [radiasi_solar],
        "Rata-rata Elevasi": [elevasi],
        "Jumlah Penduduk 0-4 Tahun": [penduduk_0_4]
    })

    # === Preprocessing (pakai scaler dari training) ===
    new_data_scaled = scaler_X.transform(new_data)

    # === Prediksi dengan model SVR yang sudah dilatih ===
    if st.button('Prediksi', key='prediction_button'):
        pred_scaled = svr_model.predict(new_data_scaled)   # hasil masih skala standar
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()[0]  # balik ke skala asli

        st.markdown(
        f"""
        <div style="
            background-color:#f9f9f9;
            border-radius:12px;
            padding:20px;
            box-shadow:0px 4px 10px rgba(0,0,0,0.15);
            text-align:center;
            font-family:Segoe UI, sans-serif;
        ">
            <h2 style="color:#2c3e50; margin-bottom:10px;">
                🔮 Prediksi Jumlah Penderita Malaria
            </h2>
            <p style="font-size:42px; font-weight:bold; color:#e74c3c; margin:0;">
                {pred:,.0f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


