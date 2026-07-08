import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI DATABASE (FILE CSV)
# ==========================================
STEP_FILE = "Log_Langkah_Eksperimen.csv"
PIPE_FILE = "Log_Pipeline_EndToEnd.csv"

# Kolom disesuaikan dengan kebutuhan PostCheck dan Reval
COL_STEPS = [
    "Test_ID", "Run_Ke", "Planner", "Langkah_Ke", "Aksi_Dieksekusi",
    "Eval_Planner_Awal", "PostCheck_Model", "PostCheck_Prediksi",
    "PostCheck_KondisiNyata", "PostCheck_Metrik", "Reval_Model", 
    "Reval_Prediksi", "Reval_KondisiNyata", "Reval_Metrik", "Catatan"
]

# Kolom disesuaikan SAMA PERSIS dengan sheet "Pipeline_Eval.csv" Anda
COL_PIPE = [
    "Test_ID", "Planner", "Validator", "Instruction_Understood", 
    "Strategy_Correct", "Postcheck_Correct", "Pose_Correct", 
    "Execution_Success", "Retry_Count", "Replan_Count", 
    "Latency_sec", "Overall_Result"
]

def load_data(file_name, columns):
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    return pd.DataFrame(columns=columns)

def save_data(df, file_name):
    df.to_csv(file_name, index=False)

def hitung_verdict(prediksi, kondisi_nyata):
    if prediksi == "PASS" and kondisi_nyata == "Aman/Bersih": return "True PASS"
    if prediksi == "FAIL" and kondisi_nyata == "Gagal/Berubah": return "True FAIL"
    if prediksi == "PASS" and kondisi_nyata == "Gagal/Berubah": return "False PASS"
    if prediksi == "FAIL" and kondisi_nyata == "Aman/Bersih": return "False FAIL"
    return "N/A"

# ==========================================
# 2. MEMUAT DATA (Solusi Error df_steps)
# ==========================================
# Data dimuat di luar tab agar dikenali oleh seluruh bagian aplikasi
df_steps = load_data(STEP_FILE, COL_STEPS)
df_pipe = load_data(PIPE_FILE, COL_PIPE)

# ==========================================
# 3. ANTARMUKA STREAMLIT
# ==========================================
st.set_page_config(page_title="Thesis Dashboard", layout="wide")
st.title("🤖 Robot Pick-and-Place: Evaluation Dashboard")

# Membuat 4 Tab Utama
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 1. Input Per Langkah (Step)", 
    "🏁 2. Input Pipeline Eval (End-to-End)", 
    "📊 3. Analisis & Grafik", 
    "📂 4. Data Mentah"
])

# ------------------------------------------
# TAB 1: FORM INPUT DATA PER LANGKAH
# ------------------------------------------
with tab1:
    st.subheader("Catat Log Per Langkah (Oklusi Dinamis)")
    with st.form("input_step_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Info Utama**")
            test_id = st.selectbox("Test ID", [f"TC{str(i).zfill(2)}" for i in range(1, 15)])
            run_ke = st.selectbox("Percobaan Ke (Run)", ["Run 1", "Run 2", "Run 3"])
            planner = st.selectbox("Planner Utama", ["OpenAI", "Gemini"])
            
        with col2:
            st.markdown("**Detail Langkah**")
            langkah = st.text_input("Langkah Ke", "1")
            aksi = st.text_input("Aksi Dieksekusi (Contoh: remove blue_block)", "N/A")
            eval_planner = st.selectbox("Evaluasi Planner Awal", ["N/A", "PASS (Benar)", "FAIL (Salah Urutan)"])
            
        with col3:
            st.markdown("**Catatan Tambahan**")
            catatan = st.text_area("Catatan Kejadian", "")
            
        st.markdown("---")
        st.markdown("**Evaluasi Sensor Fisik & Penalaran Semantik**")
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("**📸 Post-Check (Visual Validation)**")
            pc_model = st.selectbox("Model Sensor", ["YOLO-World", "Qwen2.5-VL"])
            pc_pred = st.selectbox("PostCheck Prediksi", ["N/A", "PASS", "FAIL"])
            pc_kondisi = st.selectbox("PostCheck Kondisi Nyata di Meja", ["N/A", "Aman/Bersih", "Gagal/Berubah"])
            
        with col5:
            st.markdown("**🧠 Revalidation (Sisa Plan)**")
            rev_model = st.selectbox("Model Validator Lokal", ["Qwen2.5-VL"])
            rev_pred = st.selectbox("Reval Prediksi", ["N/A", "PASS", "FAIL"])
            rev_kondisi = st.selectbox("Reval Kondisi Nyata (Sisa Plan)", ["N/A", "Aman/Bersih", "Gagal/Berubah"])

        submit_step = st.form_submit_button("💾 Simpan Data Langkah", use_container_width=True)
        
        if submit_step:
            pc_metrik = hitung_verdict(pc_pred, pc_kondisi)
            rev_metrik = hitung_verdict(rev_pred, rev_kondisi)
            
            new_step = pd.DataFrame([[
                test_id, run_ke, planner, langkah, aksi, eval_planner, 
                pc_model, pc_pred, pc_kondisi, pc_metrik, 
                rev_model, rev_pred, rev_kondisi, rev_metrik, catatan
            ]], columns=COL_STEPS)
            
            df_steps = pd.concat([df_steps, new_step], ignore_index=True)
            save_data(df_steps, STEP_FILE)
            st.success("✅ Data Langkah berhasil disimpan!")
            st.rerun()

# ------------------------------------------
# TAB 2: FORM INPUT PIPELINE EVALUATION
# ------------------------------------------
with tab2:
    st.subheader("Catat Penilaian Pipeline (Sesuai Pipeline_Eval.csv)")
    st.info("Form ini digunakan untuk merangkum hasil keseluruhan (End-to-End) setelah satu eksperimen selesai.")
    
    with st.form("input_pipe_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_test_id = st.selectbox("Test ID ", [f"TC{str(i).zfill(2)}" for i in range(1, 15)])
            p_planner = st.selectbox("Planner ", ["OpenAI", "Gemini"])
            p_validator = st.selectbox("Validator (End-to-End)", ["YOLO-World", "Qwen2.5-VL"])
            
        with c2:
            inst_under = st.selectbox("Instruction Understood?", ["Yes", "No"])
            strat_corr = st.selectbox("Strategy Correct?", ["Yes", "No"])
            post_corr = st.selectbox("Postcheck Correct?", ["Yes", "No"])
            pose_corr = st.selectbox("Pose Correct?", ["Yes", "No"])
            
        with c3:
            exec_succ = st.selectbox("Execution Success?", ["Yes", "No"])
            retry_count = st.number_input("Retry Count", min_value=0, value=0)
            replan_count = st.number_input("Replan Count", min_value=0, value=0)
            latency = st.number_input("Latency (sec)", min_value=0.0, value=0.0, step=0.1)
            
        p_overall = st.selectbox("Overall Result", ["PASS", "FAIL"])
        submit_pipe = st.form_submit_button("🏁 Simpan Data Pipeline Eval", use_container_width=True)

        if submit_pipe:
            new_pipe = pd.DataFrame([[
                p_test_id, p_planner, p_validator, inst_under, strat_corr, 
                post_corr, pose_corr, exec_succ, retry_count, replan_count, 
                latency, p_overall
            ]], columns=COL_PIPE)
            df_pipe = pd.concat([df_pipe, new_pipe], ignore_index=True)
            save_data(df_pipe, PIPE_FILE)
            st.success("✅ Data Pipeline Eval berhasil disimpan!")
            st.rerun()

# ------------------------------------------
# TAB 3: GRAFIK & ANALISIS
# ------------------------------------------
with tab3:
    st.subheader("Visualisasi Metrik Evaluasi")
    
    # === A. Evaluasi Tingkat Langkah (PostCheck & Reval) ===
    st.markdown("### 1. Evaluasi Tingkat Langkah (Sensor & Validator)")
    if len(df_steps) == 0:
        st.warning("Belum ada data eksperimen per langkah.")
    else:
        def calc_metrics(df_filtered, col_name):
            counts = df_filtered[col_name].value_counts()
            tp = counts.get('True PASS', 0)
            tn = counts.get('True FAIL', 0)
            fp = counts.get('False PASS', 0)
            fn = counts.get('False FAIL', 0)
            
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            return tp, tn, fp, fn, acc, prec, rec, f1

        pc_tp, pc_tn, pc_fp, pc_fn, pc_acc, pc_prec, pc_rec, pc_f1 = calc_metrics(df_steps, "PostCheck_Metrik")
        rev_tp, rev_tn, rev_fp, rev_fn, rev_acc, rev_prec, rev_rec, rev_f1 = calc_metrics(df_steps, "Reval_Metrik")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Akurasi Sensor (YOLO)", f"{pc_acc*100:.1f}%")
        col_m2.metric("F1-Score Sensor", f"{pc_f1:.2f}")
        col_m3.metric("Akurasi Validator (Qwen)", f"{rev_acc*100:.1f}%")
        col_m4.metric("F1-Score Validator", f"{rev_f1:.2f}")

        # Confusion Matrix Heatmaps
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        sns.heatmap([[pc_tn, pc_fp], [pc_fn, pc_tp]], annot=True, fmt="d", cmap="Blues", ax=ax1,
                    xticklabels=['Prediksi FAIL', 'Prediksi PASS'], yticklabels=['Nyata Gagal', 'Nyata Bersih'])
        ax1.set_title("Post-Check")
        
        sns.heatmap([[rev_tn, rev_fp], [rev_fn, rev_tp]], annot=True, fmt="d", cmap="Greens", ax=ax2,
                    xticklabels=['Prediksi FAIL', 'Prediksi PASS'], yticklabels=['Nyata Berubah', 'Nyata Aman'])
        ax2.set_title("Revalidation")
        st.pyplot(fig2)

    st.markdown("---")
    
    # === B. Evaluasi Keseluruhan (Pipeline End-to-End) ===
    st.markdown("### 2. Evaluasi Keseluruhan Misi (End-to-End Pipeline)")
    if len(df_pipe) == 0:
        st.warning("Belum ada data evaluasi pipeline keseluruhan.")
    else:
        df_pipe["Success_Bool"] = df_pipe["Overall_Result"].apply(lambda x: 1 if x == "PASS" else 0)
        
        total_runs = len(df_pipe)
        overall_success = df_pipe["Success_Bool"].mean() * 100
        avg_latency = pd.to_numeric(df_pipe["Latency_sec"]).mean()
        total_retry = pd.to_numeric(df_pipe["Retry_Count"]).sum()
        total_replan = pd.to_numeric(df_pipe["Replan_Count"]).sum()
        planner_success = df_pipe.groupby("Planner")["Success_Bool"].mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall PASS Rate", f"{overall_success:.1f}%", f"Dari {total_runs} Eksperimen")
        c2.metric("Rata-rata Latency", f"{avg_latency:.1f} detik")
        c3.metric("Total Intervensi (Retry)", f"{total_retry} kali")
        c4.metric("Total Intervensi (Replan)", f"{total_replan} kali")

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.barplot(x=planner_success.index, y=planner_success.values, palette="magma", ax=ax3)
        ax3.set_title("Success Rate per Planner")
        ax3.set_ylabel("PASS (%)")
        ax3.set_ylim(0, 105)
        for p in ax3.patches:
            ax3.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        st.pyplot(fig3)

# ------------------------------------------
# TAB 4: RAW DATA & EXPORT
# ------------------------------------------
with tab4:
    st.subheader("Data Mentah (Export Ready)")
    
    st.markdown("**1. Data Langkah (PostCheck & Reval)**")
    st.dataframe(df_steps)
    st.download_button("📥 Download Log Langkah", df_steps.to_csv(index=False).encode('utf-8'), "Log_Langkah_Eksperimen.csv", "text/csv")
    
    st.markdown("---")
    st.markdown("**2. Data Pipeline Evaluation**")
    st.dataframe(df_pipe)
    st.download_button("📥 Download Pipeline Eval", df_pipe.to_csv(index=False).encode('utf-8'), "Pipeline_Eval_Export.csv", "text/csv")