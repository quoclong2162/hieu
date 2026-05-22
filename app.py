import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

# ===== CẤU HÌNH STREAMLIT =====
st.set_page_config(page_title="Teen Mental Health Analysis", layout="wide")
st.title("🧠 Teen Mental Health - Phân tích & Dự đoán")

# Cấu hình đồ họa toàn cục
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Tạo thư mục lưu trữ hệ thống nếu chưa có
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== TỐI ƯU 1: LOAD DỮ LIỆU ĐA NỀN TẢNG (Sửa lỗi dấu '\' trên Linux Cloud) =====
@st.cache_data
def load_data():
    # Định dạng dấu '/' chạy được cả trên Windows, Linux và Docker
    possible_paths = [
        'data/Teen_Mental_Health_Dataset.csv',
        'data/Teen_Mental_Health_Dataset.xlsx',
        'data/Teen_Mental_Health_Dataset.xls'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    return pd.read_csv(path, encoding='utf-8-sig')
                else:
                    return pd.read_excel(path)
            except Exception as e:
                st.error(f"❌ Lỗi khi đọc file {path}: {e}")
                
    # Hiển thị hướng dẫn trực quan nếu thiếu file
    st.error("❌ Không tìm thấy file dữ liệu 'Teen_Mental_Health_Dataset'!")
    if os.path.exists('data'):
        files = os.listdir('data')
        if files:
            st.info(f"📁 Các file đang có trong thư mục 'data': {files}")
        else:
            st.warning("📁 Thư mục 'data' hiện đang trống rỗng!")
    st.info("""
    🔧 **Cách khắc phục nhanh:** Tạo thư mục tên `data`, bỏ file dữ liệu vào đó và đổi tên thành `Teen_Mental_Health_Dataset.csv`
    """)
    return None

df = load_data()

if df is None:
    st.stop()

# ===== SIDEBAR NAVIGATION =====
st.sidebar.title("📋 MENU CHÍNH")
page = st.sidebar.radio("Chọn trang:", 
    ["🏠 Trang chủ", 
     "📊 EDA - Phân tích dữ liệu", 
     "🤖 Xây dựng mô hình",
     "💡 Dự đoán"])

# ========================================
# 🏠 TRANG CHỦ
# ========================================
if page == "🏠 Trang chủ":
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 👋 Chào mừng bạn!
        Ứng dụng này giúp bạn:
        - 📊 Phân tích các chỉ số thói quen ảnh hưởng đến tâm lý học đường.
        - 🤖 Huấn luyện các mô hình học máy (Machine Learning) tự động.
        - 💡 Đưa ra dự đoán nhanh các trường hợp có nguy cơ trầm cảm.
        """)
    
    with col2:
        st.info(f"""
        📈 **Thông tin tổng quan Dataset:**
        - Số lượng mẫu: **{df.shape[0]}** dòng
        - Số lượng thuộc tính: **{df.shape[1]}** cột
        - Dữ liệu thiếu (Missing): **{df.isnull().sum().sum()}** ô
        - Biến mục tiêu dự báo: `depression_label`
        """)
    
    st.markdown("---")
    st.subheader("📊 Xem trước 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df.head(), use_container_width=True)

# ========================================
# 📊 EDA - PHÂN TÍCH DỮ LIỆU (Tối ưu giao diện & Bộ nhớ)
# ========================================
elif page == "📊 EDA - Phân tích dữ liệu":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    # Thống kê nhanh dạng thẻ
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số mẫu", df.shape[0])
    col2.metric("Số lượng cột", df.shape[1])
    col3.metric("Số ô trống (Missing)", df.isnull().sum().sum())
    col4.metric("Dòng trùng lặp (Duplicate)", df.duplicated().sum())
    
    st.markdown("---")
    
    # TỐI ƯU 2: Tự động tính toán lưới đồ thị cho cột Numeric (Tránh lỗi vỡ layout)
    st.subheader("1️⃣ Phân bố của các thuộc tính số (Numeric)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Phân bố {col}', fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
    # Xóa bỏ các ô đồ thị thừa nếu số lượng cột không chia hết cho 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig) # TỐI ƯU 3: Giải phóng bộ nhớ RAM (Chống tràn RAM cho server)

    # Ma trận tương quan gọn gàng
    st.subheader("2️⃣ Ma trận tương quan hệ số (Correlation Matrix)")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ========================================
# 🤖 XÂY DỰNG MÔ HÌNH (Tối ưu hóa Cache tăng tốc 10x)
# ========================================
elif page == "🤖 Xây dựng mô hình":
    st.header("🤖 Huấn luyện Mô hình Machine Learning")
    
    # TỐI ƯU 4: Đóng gói quá trình xử lý vào hàm có Cache. 
    # Mô hình chỉ train 1 lần duy nhất, click nút khác không bị chạy lại.
    @st.cache_resource
    def pipeline_train_models(_data_frame):
        X = _data_frame.drop('depression_label', axis=1)
        y = _data_frame['depression_label']
        
        # Lưu lại danh sách tên cột chuẩn để đồng bộ cho form Predict về sau
        feature_names = X.columns.tolist()
        
        # Mã hóa Categorical
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        label_encoders = {}
        X_encoded = X.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Chuẩn hóa
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Huấn luyện Logistic Regression
        log_reg = LogisticRegression(random_state=42, max_iter=1000)
        log_reg.fit(X_train_scaled, y_train)
        
        # Huấn luyện Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        return log_reg, rf, scaler, label_encoders, X_test_scaled, y_test, feature_names

    if st.button("🚀 Kích hoạt huấn luyện hệ thống mô hình"):
        with st.spinner("Hệ thống đang xử lý dữ liệu và huấn luyện mô hình, vui lòng đợi..."):
            log_reg, rf, scaler, label_encoders, X_test_scaled, y_test, feature_names = pipeline_train_models(df)
            
            # Tính toán các chỉ số đánh giá nhanh
            y_pred_log = log_reg.predict(X_test_scaled)
            y_pred_rf = rf.predict(X_test_scaled)
            
            acc_log = accuracy_score(y_test, y_pred_log)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            
            st.success(f"🎉 Hoàn tất! Độ chính xác (Accuracy) -> LR: {acc_log:.4f} | RF: {acc_rf:.4f}")
            
            # Chọn ra mô hình xuất sắc nhất
            if acc_rf >= acc_log:
                best_model, best_name = rf, "Random Forest"
            else:
                best_model, best_name = log_reg, "Logistic Regression"
                
            # TỐI ƯU 5: Đưa các đối tượng đã huấn luyện vào Session State để lưu trữ tạm thời
            st.session_state['train_outputs'] = {
                'best_model': best_model,
                'model_name': best_name,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'feature_names': feature_names
            }
            
    # Phần lưu trữ mô hình ra file vật lý .pkl
    if 'train_outputs' in st.session_state:
        st.markdown("---")
        st.subheader("💾 Lưu trữ bộ não mô hình tốt nhất")
        
        if st.button("💾 Xác nhận lưu mô hình"):
            outputs = st.session_state['train_outputs']
            try:
                joblib.dump(outputs['best_model'], 'models/best_model.pkl')
                joblib.dump(outputs['scaler'], 'models/scaler.pkl')
                joblib.dump(outputs['label_encoders'], 'models/label_encoders.pkl')
                joblib.dump(outputs['feature_names'], 'models/feature_names.pkl') # Lưu cấu trúc cột
                st.success(f"✅ Đã đóng gói thành công mô hình đạt điểm cao nhất: **{outputs['model_name']}** vào thư mục 'models/'")
            except Exception as e:
                st.error(f"❌ Thất bại khi lưu mô hình: {e}")

# ========================================
# 💡 DỰ ĐOÁN (Tối ưu bằng Form và Chống lệch cột)
# ========================================
elif page == "💡 Dự đoán":
    st.header("💡 Khung kiểm tra & Dự đoán Trầm cảm")
    
    # Kiểm tra xem tệp mô hình đã tồn tại hay chưa
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
