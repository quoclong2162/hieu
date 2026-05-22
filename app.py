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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

# ===== CẤU HÌNH STREAMLIT =====
st.set_page_config(page_title="Teen Mental Health Analysis", layout="wide")
st.title("🧠 Teen Mental Health - Phân tích & Dự đoán")

# ===== CẤU HÌNH MATPLOTLIB =====
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ===== TẠO THƯ MỤC =====
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== LOAD DỮ LIỆU =====
@st.cache_data
def load_data():
    try:
        # Thử load từ CSV trước
        try:
            df = pd.read_csv('data/Teen_Mental_Health_Dataset.csv', encoding='utf-8-sig')
            st.success("✅ Đã load từ file CSV")
        except FileNotFoundError:
            # Nếu không có CSV, thử load từ Excel
            try:
                df = pd.read_excel('data/Teen_Mental_Health_Dataset.xlsx')
                st.success("✅ Đã load từ file Excel")
            except FileNotFoundError:
                # Cuối cùng thử file Excel tên khác
                try:
                    df = pd.read_excel('data/Teen_Mental_Health_Dataset.xls')
                    st.success("✅ Đã load từ file Excel")
                except FileNotFoundError:
                    # Liệt kê các file trong thư mục data
                    st.error("❌ File 'Teen_Mental_Health_Dataset' không tìm thấy!")
                    
                    if os.path.exists('data'):
                        files = os.listdir('data')
                        if files:
                            st.info(f"📁 Các file trong thư mục 'data': {files}")
                        else:
                            st.warning("📁 Thư mục 'data' trống!")
                    
                    st.info("""
                    🔧 **Cách khắc phục:**
                    1. Đặt file Dataset vào thư mục 'data/'
                    2. Tên file phải là một trong những cái sau:
                       - Teen_Mental_Health_Dataset.csv
                       - Teen_Mental_Health_Dataset.xlsx
                       - Teen_Mental_Health_Dataset.xls
                    """)
                    return None
        
        return df
    
    except Exception as e:
        st.error(f"❌ Lỗi khi load dữ liệu: {e}")
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
        ## 👋 Chào mừng!
        
        Ứng dụng này giúp:
        - 📊 Phân tích dữ liệu về sức khỏe tâm thần của thanh thiếu niên
        - 🤖 Xây dựng mô hình Machine Learning dự đoán trầm cảm
        - 💡 Đưa ra dự đoán cho các trường hợp mới
        
        **Dataset:** Teen Mental Health Dataset
        """)
    
    with col2:
        st.info(f"""
        📈 **Thống kê Dataset:**
        - Số mẫu: {df.shape[0]}
        - Số features: {df.shape[1]}
        - Missing values: {df.isnull().sum().sum()}
        - Target: depression_label
        """)
    
    st.markdown("---")
    st.subheader("📊 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df.head(), use_container_width=True)

# ========================================
# 📊 EDA - PHÂN TÍCH DỮ LIỆU
# ========================================
elif page == "📊 EDA - Phân tích dữ liệu":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    # === 1. THÔNG TIN CƠ BẢN ===
    st.subheader("1️⃣ Thông tin cơ bản Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Số mẫu", df.shape[0])
    with col2:
        st.metric("Số features", df.shape[1])
    with col3:
        st.metric("Missing values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate rows", df.duplicated().sum())
    
    # === 2. KIỂM TRA MISSING VALUES ===
    st.subheader("2️⃣ Kiểm tra Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"⚠️ Có {missing.sum()} giá trị thiếu!")
        fig, ax = plt.subplots(figsize=(12, 5))
        missing[missing > 0].plot(kind='bar', color='red', edgecolor='black', ax=ax)
        ax.set_title('Missing Values theo Cột', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cột')
        ax.set_ylabel('Số lượng giá trị thiếu')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.success("✅ Dữ liệu sạch - Không có missing values!")
    
    # === 3. THỐNG KÊ MÔ TẢ ===
    st.subheader("3️⃣ Thống kê mô tả (Describe)")
    st.dataframe(df.describe().round(3), use_container_width=True)
    
    # === 4. KIỂU DỮ LIỆU ===
    st.subheader("4️⃣ Kiểu dữ liệu")
    st.dataframe(pd.DataFrame({
        'Cột': df.columns,
        'Kiểu': df.dtypes
    }), use_container_width=True)
    
    # === 5. PHÂN TÍCH CỘT NUMERIC ===
    st.subheader("5️⃣ Phân tích cột Numeric")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.info(f"Tìm thấy {len(numeric_cols)} cột numeric: {', '.join(numeric_cols)}")
    
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Phân bố {col}', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Tần suất')
    plt.tight_layout()
    st.pyplot(fig)
    
    # === 6. PHÂN TÍCH CỘT CATEGORICAL ===
    st.subheader("6️⃣ Phân tích cột Categorical")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.info(f"Tìm thấy {len(categorical_cols)} cột categorical: {', '.join(categorical_cols)}")
    
    for col in categorical_cols:
        st.markdown(f"**📊 Cột: {col}**")
        col1, col2 = st.columns(2)
        
        with col1:
            value_counts = df[col].value_counts()
            st.dataframe(value_counts, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            value_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(f'Phân bố {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Tần suất')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # === 7. PHÂN TÍCH TARGET ===
    st.subheader("7️⃣ Phân tích Target - depression_label")
    col1, col2 = st.columns(2)
    
    with col1:
        depression_counts = df['depression_label'].value_counts()
        st.dataframe({
            'Depression': ['No (0)', 'Yes (1)'],
            'Count': depression_counts.values,
            'Percentage': (depression_counts.values / len(df) * 100).round(2)
        }, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        df['depression_label'].value_counts().plot(kind='bar', 
            color=['green', 'red'], edgecolor='black', ax=ax)
        ax.set_title('Phân bố Depression Label', fontsize=14, fontweight='bold')
        ax.set_xlabel('Depression Label')
        ax.set_ylabel('Tần suất')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    # === 8. MA TRẬN TƯƠNG QUAN ===
    st.subheader("8️⃣ Ma trận tương quan (Correlation)")
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    st.markdown("**🔍 Correlation với depression_label:**")
    corr_depression = correlation['depression_label'].sort_values(ascending=False)
    st.dataframe(corr_depression, use_container_width=True)
    
    # === 9. PHÁT HIỆN OUTLIERS ===
    st.subheader("9️⃣ Phát hiện Outliers (Boxplot)")
    fig, ax = plt.subplots(figsize=(16, 8))
    numeric_df.boxplot(ax=ax)
    ax.set_title('Boxplot - Phát hiện Outliers', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ========================================
# 🤖 XÂY DỰNG MÔ HÌNH
# ========================================
elif page == "🤖 Xây dựng mô hình":
    st.header("🤖 Xây dựng & Huấn luyện Mô hình Machine Learning")
    
    # === TÁCH DỮ LIỆU ===
    st.subheader("1️⃣ Tách X (Features) và y (Target)")
    X = df.drop('depression_label', axis=1)
    y = df['depression_label']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Features shape", X.shape)
    with col2:
        st.metric("Target shape", y.shape)
    
    # === XỬ LÝ CATEGORICAL ===
    st.subheader("2️⃣ Xử lý dữ liệu Categorical (Label Encoding)")
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    st.info(f"Cột categorical: {', '.join(categorical_cols)}")
    
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
        st.markdown(f"**✓ {col}:**")
        for i, class_name in enumerate(le.classes_):
            st.write(f"   {class_name} → {i}")
    
    st.success("✅ Dữ liệu categorical đã xử lý!")
    
    # === CHIA TRAIN/TEST ===
    st.subheader("3️⃣ Chia dữ liệu Train/Test (80/20)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train set size", X_train.shape[0])
    with col2:
        st.metric("Test set size", X_test.shape[0])
    
    st.markdown("**📊 Phân bố Train set:**")
    st.dataframe(y_train.value_counts(), use_container_width=True)
    
    st.markdown("**📊 Phân bố Test set:**")
    st.dataframe(y_test.value_counts(), use_container_width=True)
    
    # === CHUẨN HÓA DỮ LIỆU ===
    st.subheader("4️⃣ Chuẩn hóa dữ liệu (Standardization)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.success("✅ Dữ liệu đã chuẩn hóa!")
    
    # === HUẤN LUYỆN MÔ HÌNH ===
    st.subheader("5️⃣ Huấn luyện Mô hình")
    
    # Logistic Regression
    st.markdown("**📈 Logistic Regression:**")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    st.success("✅ Mô hình Logistic Regression đã huấn luyện!")
    
    y_pred_log = log_reg.predict(X_test_scaled)
    y_pred_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    # Random Forest
    st.markdown("**🌲 Random Forest:**")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    st.success("✅ Mô hình Random Forest đã huấn luyện!")
    
    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    # === ĐÁNH GIÁ MÔ HÌNH ===
    st.subheader("6️⃣ Đánh giá Mô hình")
    
    # Metrics Logistic Regression
    accuracy_log = accuracy_score(y_test, y_pred_log)
    precision_log = precision_score(y_test, y_pred_log)
    recall_log = recall_score(y_test, y_pred_log)
    f1_log = f1_score(y_test, y_pred_log)
    roc_auc_log = roc_auc_score(y_test, y_pred_proba_log)
    
    # Metrics Random Forest
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    
    # Bảng so sánh
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Logistic Regression': [accuracy_log, precision_log, recall_log, f1_log, roc_auc_log],
        'Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf]
    })
    
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # === BIỂU ĐỒ SO SÁNH ===
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df.set_index('Metric').plot(kind='bar', width=0.8, ax=ax)
    ax.set_title('So sánh Metrics - Logistic Regression vs Random Forest', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_ylim([0, 1.1])
    plt.xticks(rotation=45)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    # === CONFUSION MATRIX ===
    st.subheader("7️⃣ Confusion Matrix")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logistic Regression:**")
        cm_log = confusion_matrix(y_test, y_pred_log)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Depression', 'Depression'],
                    yticklabels=['No Depression', 'Depression'], ax=ax)
        ax.set_title('Confusion Matrix - LR', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Random Forest:**")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['No Depression', 'Depression'],
                    yticklabels=['No Depression', 'Depression'], ax=ax)
        ax.set_title('Confusion Matrix - RF', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
    
    # === ROC CURVE ===
    st.subheader("8️⃣ ROC Curve")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
    ax.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={roc_auc_log:.4f})', linewidth=2)
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_rf:.4f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline', linewidth=2)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - So sánh 2 Mô hình', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # === FEATURE IMPORTANCE ===
    st.subheader("9️⃣ Feature Importance (Random Forest)")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.dataframe(feature_importance, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    st.pyplot(fig)
    
    # === CHỌN MÔ HÌNH TỐTẤT ===
    st.subheader("🏆 Mô hình tốt nhất")
    if roc_auc_rf > roc_auc_log:
        best_model = "Random Forest"
        best_model_obj = rf
        best_auc = roc_auc_rf
        best_accuracy = accuracy_rf
    else:
        best_model = "Logistic Regression"
        best_model_obj = log_reg
        best_auc = roc_auc_log
        best_accuracy = accuracy_log
    
    st.success(f"""
    🏆 **Mô hình tốt nhất: {best_model}**
    - Accuracy: {best_accuracy:.4f}
    - ROC-AUC: {best_auc:.4f}
    """)
    
    # === LƯU MÔ HÌNH ===
    st.subheader("💾 Lưu Mô hình")
    if st.button("💾 Lưu Mô hình"):
        try:
            joblib.dump(best_model_obj, f'models/{best_model.replace(" ", "_").lower()}_model.pkl')
            joblib.dump(scaler, 'models/scaler.pkl')
            joblib.dump(label_encoders, 'models/label_encoders.pkl')
            st.success(f"✅ Mô hình {best_model} đã lưu thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi khi lưu: {e}")

# ========================================
# 💡 DỰ ĐOÁN
# ========================================
elif page == "💡 Dự đoán":
    st.header("💡 Dự đoán Trầm cảm cho Trường hợp Mới")
    
    # === LOAD MÔ HÌNH ===
    try:
        # Thử load Random Forest trước
        try:
            model = joblib.load('models/random_forest_model.pkl')
            model_name = "Random Forest"
        except:
            model = joblib.load('models/logistic_regression_model.pkl')
            model_name = "Logistic Regression"
        
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        
        st.success(f"✅ Đã load mô hình: {model_name}")
    except:
        st.error("❌ Chưa có mô hình được lưu! Vui lòng huấn luyện mô hình trước.")
        st.stop()
    
    st.markdown("---")
    st.subheader("📝 Nhập thông tin thanh niên:")
    
    # === TẠO FORM NHẬP ===
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("👤 Tuổi (Age)", min_value=13, max_value=19, value=16)
        daily_social_media_hours = st.slider("📱 Số giờ dùng mạng xã hội/ngày", 
                                             min_value=0.0, max_value=12.0, value=6.0, step=0.5)
        sleep_hours = st.slider("😴 Số giờ ngủ/đêm", 
                               min_value=3.0, max_value=12.0, value=8.0, step=0.5)
        screen_time_before_sleep = st.slider("📺 Thời gian dùng màn hình trước khi ngủ (phút)", 
                                            min_value=0.0, max_value=180.0, value=60.0, step=15.0)
    
    with col2:
        academic_performance = st.slider("📚 Hiệu suất học tập (0-100)", 
                                        min_value=0, max_value=100, value=75)
        physical_activity = st.slider("🏃 Hoạt động thể chất (giờ/tuần)", 
                                     min_value=0.0, max_value=14.0, value=5.0, step=0.5)
        stress_level = st.slider("😰 Mức độ căng thẳng (0-10)", 
                                min_value=0, max_value=10, value=5)
        anxiety_level = st.slider("😟 Mức độ lo âu (0-10)", 
                                 min_value=0, max_value=10, value=5)
    
    col3, col4 = st.columns(2)
    
    with col3:
        addiction_level = st.slider("🎮 Mức độ nghiện (0-10)", 
                                   min_value=0, max_value=10, value=5)
        gender = st.selectbox("👥 Giới tính", options=['Male', 'Female'])
    
    with col4:
        platform_usage = st.selectbox("📲 Nền tảng mạng xã hội", 
                                     options=['Instagram', 'TikTok', 'Both'])
        social_interaction_level = st.selectbox("🤝 Mức độ tương tác xã hội", 
                                               options=['Low', 'Medium', 'High'])
    
    st.markdown("---")
    
    # === CHUẨN BỊ DỮ LIỆU ===
    if st.button("🔮 Dự đoán"):
        # Tạo DataFrame từ input
        input_data = pd.DataFrame({
            'age': [age],
            'daily_social_media_hours': [daily_social_media_hours],
            'sleep_hours': [sleep_hours],
            'screen_time_before_sleep': [screen_time_before_sleep],
            'academic_performance': [academic_performance],
            'physical_activity': [physical_activity],
            'stress_level': [stress_level],
            'anxiety_level': [anxiety_level],
            'addiction_level': [addiction_level],
            'gender': [gender],
            'platform_usage': [platform_usage],
            'social_interaction_level': [social_interaction_level]
        })
        
        # Xử lý categorical
        input_data_encoded = input_data.copy()
        for col, le in label_encoders.items():
            if col in input_data_encoded.columns:
                input_data_encoded[col] = le.transform(input_data_encoded[col])
        
        # Chuẩn hóa
        input_data_scaled = scaler.transform(input_data_encoded)
        
        # Dự đoán
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0]
        
        # === HIỂN THỊ KẾT QUẢ ===
        st.markdown("---")
        st.subheader("📊 Kết quả Dự đoán:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 0:
                st.success("✅ KHÔNG CÓ TRẦM CẢM")
            else:
                st.error("⚠️ CÓ TRẦM CẢM")
        
        with col2:
            st.metric("Xác suất Không trầm cảm", f"{probability[0]*100:.2f}%")
        
        with col3:
            st.metric("Xác suất Trầm cảm", f"{probability[1]*100:.2f}%")
        
        # === BIỂU ĐỒ XÁC SUẤT ===
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Không trầm cảm', 'Có trầm cảm']
        colors = ['green', 'red']
        bars = ax.bar(categories, probability, color=colors, edgecolor='black', width=0.5)
        
        # Thêm giá trị trên cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Xác suất', fontsize=12)
        ax.set_title('Xác suất Dự đoán', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        st.pyplot(fig)
        
        # === LỜI KHUYÊN ===
        st.markdown("---")
        st.subheader("💡 Lời khuyên:")
        
        if prediction == 1:
            st.warning("""
            ⚠️ **Có dấu hiệu trầm cảm!**
            
            Vui lòng:
            - 🏥 Tham khảo ý kiến bác sĩ tâm thần
            - 💬 Nói chuyện với gia đình, bạn bè
            - 🧘 Tập các kỹ năng thư giãn
            - 🏃 Tăng hoạt động thể chất
            - 📱 Giảm thời gian dùng mạng xã hội
            - 😴 Cải thiện chất lượng giấc ngủ
            """)
        else:
            st.success("""
            ✅ **Không có dấu hiệu trầm cảm!**
            
            Tiếp tục duy trì:
            - 🏃 Hoạt động thể chất đều đặn
            - 😴 Giấc ngủ đủ 7-9 giờ/đêm
            - 🤝 Tương tác xã hội tích cực
            - 📚 Cân bằng học tập và vui chơi
            - 📱 Sử dụng mạng xã hội một cách khôn ngoan
            """)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 <strong>Teen Mental Health Analysis App</strong> | Made with ❤️ by Hieu-L206</p>
    <p>Dataset: Teen Mental Health Dataset | ML Models: Logistic Regression & Random Forest</p>
</div>
""", unsafe_allow_html=True)
