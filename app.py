import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="KNN Customer Purchase Prediction (SSG 535 GROUP E)",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-success {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        color: #155724;
    }
    .prediction-failure {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header"> KNN Customer Purchase Prediction (SSG 535 GROUP E)</p>',
            unsafe_allow_html=True)
st.markdown("---")

# Load and prepare data


@st.cache_data
def load_data():
    df = pd.read_csv("Social_Network_Ads.csv")
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    return df


df = load_data()

# Prepare features and target
X = df[['Age', 'EstimatedSalary', 'Gender']]
y = df['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar for hyperparameters
st.sidebar.header("🎛️ Model Hyperparameters")
st.sidebar.markdown(
    "Adjust the parameters below to see how they affect the model's performance.")

# Initialize session state for k_value if not exists
if 'k_value' not in st.session_state:
    st.session_state.k_value = 6

# K value with buttons and slider
st.sidebar.markdown("**Number of Neighbors (k)**")
btn_col1, btn_col2, btn_col3 = st.sidebar.columns([1, 2, 1])

with btn_col1:
    if st.button("➖", key="decrease_k", use_container_width=True):
        if st.session_state.k_value > 1:
            st.session_state.k_value -= 1

with btn_col2:
    st.markdown(
        f"<div style='text-align: center; font-size: 1.5rem; font-weight: bold; padding: 5px;'>{st.session_state.k_value}</div>", unsafe_allow_html=True)

with btn_col3:
    if st.button("➕", key="increase_k", use_container_width=True):
        if st.session_state.k_value < 20:
            st.session_state.k_value += 1

k_value = st.sidebar.slider(
    "Slide to adjust k",
    min_value=1,
    max_value=20,
    value=st.session_state.k_value,
    key="k_slider",
    help="The number of nearest neighbors to consider for classification",
    label_visibility="collapsed"
)

# Sync slider with session state
if k_value != st.session_state.k_value:
    st.session_state.k_value = k_value
    st.rerun()

distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    options=["euclidean", "manhattan", "chebyshev",
             "minkowski", "cosine", "haversine"],
    index=0,
    help="The distance metric used to find nearest neighbors"
)

weights = st.sidebar.selectbox(
    "Weights",
    options=["uniform", "distance"],
    index=0,
    help="Weight function used in prediction"
)

# Train model with selected hyperparameters


@st.cache_data
def train_model(k, metric, weight):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    return knn, y_pred


knn, y_pred = train_model(k_value, distance_metric, weights)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Main content tabs
tab1, tab2, tab3 = st.tabs(
    ["📊 Model Performance", "📈 Visualizations", "🔮 Make Predictions"])

# Tab 1: Model Performance
with tab1:
    st.header("Model Performance Metrics")

    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Accuracy",
            value=f"{accuracy:.1%}",
            delta=f"{(accuracy - 0.8):.1%} vs baseline" if accuracy > 0.8 else None
        )

    with col2:
        st.metric(
            label="🔍 Precision",
            value=f"{precision:.1%}",
            help="When model predicts 'Purchase', how often is it correct?"
        )

    with col3:
        st.metric(
            label="📡 Recall",
            value=f"{recall:.1%}",
            help="Of all actual purchases, how many did we catch?"
        )

    with col4:
        st.metric(
            label="⚖️ F1 Score",
            value=f"{f1:.1%}",
            help="Harmonic mean of precision and recall"
        )

    st.markdown("---")

    # Confusion Matrix
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Purchase', 'Purchase'],
                    yticklabels=['No Purchase', 'Purchase'],
                    annot_kws={'size': 16})
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(
            f'KNN Confusion Matrix (k={k_value}, {distance_metric})', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Confusion Matrix Breakdown")
        st.markdown(f"""
        | Metric | Value | Meaning |
        |--------|-------|---------|
        | **True Negatives** | {cm[0, 0]} | Correctly predicted NO purchase |
        | **False Positives** | {cm[0, 1]} | Incorrectly predicted purchase |
        | **False Negatives** | {cm[1, 0]} | Missed actual purchases |
        | **True Positives** | {cm[1, 1]} | Correctly predicted purchase |
        """)

        st.markdown("---")
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=[
                                       'No Purchase', 'Purchase'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(
            "{:.2f}"), use_container_width=True)

# Tab 2: Visualizations
with tab2:
    st.header("Data Visualizations")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader("📍 Decision Boundary")

        # Decision boundary plot
        fig, ax = plt.subplots(figsize=(10, 8))

        x_min, x_max = X_train['Age'].min() - 5, X_train['Age'].max() + 5
        y_min, y_max = X_train['EstimatedSalary'].min(
        ) - 5000, X_train['EstimatedSalary'].max() + 5000

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                             np.arange(y_min, y_max, 500))

        gender_value = X_train['Gender'].median()
        mesh_data = np.c_[xx.ravel(), yy.ravel(), np.full(
            xx.ravel().shape, gender_value)]

        Z = knn.predict(scaler.transform(mesh_data))
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
        scatter = ax.scatter(X_train['Age'], X_train['EstimatedSalary'],
                             c=y_train, cmap='RdYlGn', edgecolors='black', s=50, alpha=0.7)

        ax.set_xlabel('Age', fontweight='bold', fontsize=12)
        ax.set_ylabel('Estimated Salary', fontweight='bold', fontsize=12)
        ax.set_title(
            f'Decision Boundary (k={k_value}, {distance_metric})', fontweight='bold', fontsize=14)
        ax.legend(*scatter.legend_elements(),
                  title='Purchased', loc='upper left')

        st.pyplot(fig)
        plt.close()

    with viz_col2:
        st.subheader("📊 Accuracy vs K Value")

        # K value comparison
        @st.cache_data
        def get_k_accuracies(metric, weight):
            accuracies = []
            for k in range(1, 21):
                knn_temp = KNeighborsClassifier(
                    n_neighbors=k, metric=metric, weights=weight)
                knn_temp.fit(X_train_scaled, y_train)
                y_pred_temp = knn_temp.predict(X_test_scaled)
                accuracies.append(accuracy_score(y_test, y_pred_temp))
            return accuracies

        accuracies = get_k_accuracies(distance_metric, weights)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(range(1, 21), accuracies, marker='o',
                linewidth=2, markersize=8, color='#1f77b4')
        ax.axvline(x=k_value, color='red', linestyle='--',
                   label=f'Current k={k_value}')
        ax.set_xlabel('Number of Neighbors (K)',
                      fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax.set_title(
            f'Model Accuracy vs K Value ({distance_metric})', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Highlight best k
        best_k = np.argmax(accuracies) + 1
        ax.scatter([best_k], [max(accuracies)], color='green',
                   s=200, zorder=5, label=f'Best k={best_k}')

        st.pyplot(fig)
        plt.close()

        st.info(
            f"🏆 **Best K Value**: {best_k} with accuracy of {max(accuracies):.1%}")

    # Additional visualizations
    st.markdown("---")

    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        st.subheader("💰 Salary Distribution by Purchase Status")

        purchased_customers = df[df['Purchased'] == 1]
        non_purchased_customers = df[df['Purchased'] == 0]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(purchased_customers['EstimatedSalary'], bins=20, alpha=0.7,
                label='Purchased', color='green', edgecolor='black')
        ax.hist(non_purchased_customers['EstimatedSalary'], bins=20, alpha=0.7,
                label='Did NOT Purchase', color='red', edgecolor='black')
        ax.set_xlabel('Annual Salary ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
        ax.set_title('Salary Distribution: Purchasers vs Non-Purchasers',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with viz_col4:
        st.subheader("👤 Age vs Salary Scatter Plot")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(non_purchased_customers['Age'],
                   non_purchased_customers['EstimatedSalary']/1000,
                   alpha=0.5, s=100, label='Did NOT Purchase', color='red',
                   edgecolors='darkred', linewidth=0.5)
        ax.scatter(purchased_customers['Age'],
                   purchased_customers['EstimatedSalary']/1000,
                   alpha=0.5, s=100, label='Purchased', color='green',
                   edgecolors='darkgreen', linewidth=0.5)

        ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annual Salary ($1000s)', fontsize=12, fontweight='bold')
        ax.set_title('Customer Profile: Age vs Salary',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close()

# Tab 3: Make Predictions
with tab3:
    st.header("🔮 Predict Customer Purchase")
    st.markdown(
        "Enter customer details below to predict whether they will make a purchase.")

    pred_col1, pred_col2, pred_col3 = st.columns(3)

    with pred_col1:
        input_age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Enter customer's age"
        )

    with pred_col2:
        input_gender = st.selectbox(
            "⚧ Gender",
            options=["Male", "Female"],
            help="Select customer's gender"
        )
        gender_encoded = 0 if input_gender == "Male" else 1

    with pred_col3:
        input_salary = st.number_input(
            "💵 Estimated Salary ($)",
            min_value=10000,
            max_value=200000,
            value=50000,
            step=1000,
            help="Enter customer's estimated annual salary"
        )

    st.markdown("---")

    # Make prediction button
    if st.button("🔮 Predict Purchase", type="primary", use_container_width=True):
        # Prepare input data
        input_data = np.array([[input_age, input_salary, gender_encoded]])
        input_scaled = scaler.transform(input_data)

        # Get prediction and probability
        prediction = knn.predict(input_scaled)[0]
        probabilities = knn.predict_proba(input_scaled)[0]

        st.markdown("---")

        # Display prediction result
        if prediction == 1:
            st.markdown("""
            <div class="prediction-success">
                <h2>✅ PURCHASE LIKELY!</h2>
                <p>Based on the customer profile, this customer is <strong>likely to make a purchase</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-failure">
                <h2>❌ PURCHASE UNLIKELY</h2>
                <p>Based on the customer profile, this customer is <strong>unlikely to make a purchase</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Show probabilities
        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:
            st.metric(
                label="Probability of NO Purchase",
                value=f"{probabilities[0]:.1%}"
            )

        with prob_col2:
            st.metric(
                label="Probability of Purchase",
                value=f"{probabilities[1]:.1%}"
            )

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#dc3545', '#28a745']
        bars = ax.barh(['No Purchase', 'Purchase'],
                       probabilities, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability', fontweight='bold')
        ax.set_title('Prediction Confidence', fontweight='bold')

        for bar, prob in zip(bars, probabilities):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1%}', va='center', fontweight='bold')

        st.pyplot(fig)
        plt.close()

        # Customer profile summary
        st.markdown("### 📋 Customer Profile Summary")
        profile_data = {
            "Feature": ["Age", "Gender", "Estimated Salary"],
            "Value": [f"{input_age} years", input_gender, f"${input_salary:,}"],
            "Comparison to Purchasers": [
                f"{'Above' if input_age > df[df['Purchased'] == 1]['Age'].mean() else 'Below'} average ({df[df['Purchased'] == 1]['Age'].mean():.1f} years)",
                f"{'Matches' if gender_encoded == 1 else 'Does not match'} higher-purchasing demographic",
                f"{'Above' if input_salary > df[df['Purchased'] == 1]['EstimatedSalary'].mean() else 'Below'} average (${df[df['Purchased'] == 1]['EstimatedSalary'].mean():,.0f})"
            ]
        }
        st.table(pd.DataFrame(profile_data))

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.header("📊 Dataset Info")
st.sidebar.markdown(f"""
- **Total Samples**: {len(df)}
- **Training Set**: {len(X_train)}
- **Test Set**: {len(X_test)}
- **Features**: Age, Salary, Gender
- **Target**: Purchased (0/1)
""")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app demonstrates K-Nearest Neighbors (KNN) classification for predicting customer purchase behavior.

**How it works:**
1. Adjust hyperparameters in the sidebar
2. View real-time performance metrics
3. Explore visualizations
4. Make predictions for new customers
""")
