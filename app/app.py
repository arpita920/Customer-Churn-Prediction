import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# MODELS_DIR = r"C:\Users\Arpita\Desktop\Customer Churn Prediction\models"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

@st.cache_resource
def load_model():
    with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_scaler():
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as file:
        scaler = pickle.load(file)
    return scaler

@st.cache_data
def load_feature_columns():
    with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "rb") as file:
        feature_columns = pickle.load(file)
    return feature_columns

@st.cache_data
def load_high_churn_cities():
    with open(os.path.join(MODELS_DIR, "high_churn_cities.pkl"), "rb") as file:
        high_churn_cities = pickle.load(file)
    return high_churn_cities

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# Load all artifacts
model = load_model()
scaler = load_scaler()
feature_columns = load_feature_columns()
high_churn_cities = load_high_churn_cities()

# Load dataset for EDA charts
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/churned_or_stayed_df.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("📡 Churn Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home",
     "📊 Business Insights",
     "🔍 Customer Prediction",
     "📋 Prediction Results",
     "⚠️ EDA Context",
     "ℹ️ Model Info"]
)

if page == "🏠 Home":
    st.title("📡 Telecom Customer Churn Prediction")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", "6,475")
    with col2:
        st.metric("Churn Rate", "28.4%", delta="8.4pp above benchmark")
    with col3:
        st.metric("Est. Annual Revenue Loss", "$1.65M")
    
    st.markdown("---")
    st.subheader("About This Project")
    st.write("""
    This application analyzes customer churn for a California-based telecom company 
    using the Maven Telecom dataset (6,475 customers). It combines exploratory data 
    analysis with a trained Logistic Regression model to identify at-risk customers 
    and recommend retention actions.
    """)
    
    st.subheader("Project Pipeline")
    st.write("""
    **Phase 1 — Data Cleaning:** Handled structural nulls, excluded Joined customers, 
    finalized 6,475 rows across 38 columns.
    
    **Phase 2 — Exploratory Data Analysis:** Investigated 14 business questions 
    identifying key churn drivers including tenure, contract type, offer type, 
    and geographic clusters.
    
    **Phase 3 — Machine Learning:** Built and cross-validated Logistic Regression 
    and Random Forest models. Final model: Logistic Regression at threshold 0.30, 
    selected based on lower mean cross-validation business cost (\$58,540 vs \$61,910).
    """)


elif page == "📊 Business Insights":
    st.title("📊 Business Insights")
    st.markdown("---")
    st.write("Key findings from analysis of 6,475 California telecom customers. "
             "Red bars indicate churn rate above company average (28.4%). "
             "Green bars indicate below average.")

    BENCHMARK = 28.4

    def bar_colors(values):
        return ['#e74c3c' if v > BENCHMARK else '#27ae60' for v in values]

    def annotate_bars(ax, values):
        for i, v in enumerate(values):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', 
                   va='bottom', fontweight='bold', fontsize=9)

    # Finding 1 - Tenure
    st.subheader("1. Early Tenure Customers Are Highest Risk")
    tenure_order = ['0-6', '7-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    tenure_churn = (df.groupby('Tenure Groups in Months')['Churned']
                   .mean() * 100).reindex(tenure_order)
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    vals1 = tenure_churn.values
    ax1.bar(tenure_churn.index, vals1, color=bar_colors(vals1))
    ax1.axhline(y=BENCHMARK, color='black', linestyle='--', 
                label=f'Company Average {BENCHMARK}%')
    ax1.set_xlabel("Tenure Group (Months)")
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_title("Churn Rate by Tenure Group")
    ax1.legend()
    annotate_bars(ax1, vals1)
    st.pyplot(fig1)
    plt.close()
    st.error("⚠️ 0–6 months: 77.01% churn — 48.61pp above company average.")
    st.info("💡 Prioritize onboarding experience and early retention offers for new customers.")

    st.markdown("---")

    # Finding 2 - Contract
    st.subheader("2. Month-to-Month Contracts Drive Highest Churn")
    contract_churn = df.groupby('Contract')['Churned'].mean() * 100
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    vals2 = contract_churn.values
    ax2.bar(contract_churn.index, vals2, color=bar_colors(vals2))
    ax2.axhline(y=BENCHMARK, color='black', linestyle='--',
                label=f'Company Average {BENCHMARK}%')
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_title("Churn Rate by Contract Type")
    ax2.legend()
    annotate_bars(ax2, vals2)
    st.pyplot(fig2)
    plt.close()
    st.error("⚠️ Month-to-Month: 51.65% vs Two Year: 2.62% — a 49.03pp difference.")
    st.info("💡 Incentivize customers to upgrade to longer-term contracts.")

    st.markdown("---")

    # Finding 3 - Offers
    st.subheader("3. Offer Type Significantly Impacts Churn")
    offer_churn = df.groupby('Offer')['Churned'].mean() * 100
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    vals3 = offer_churn.values
    ax3.bar(offer_churn.index, vals3, color=bar_colors(vals3))
    ax3.axhline(y=BENCHMARK, color='black', linestyle='--',
                label=f'Company Average {BENCHMARK}%')
    ax3.set_ylabel("Churn Rate (%)")
    ax3.set_title("Churn Rate by Offer Type")
    ax3.legend()
    annotate_bars(ax3, vals3)
    st.pyplot(fig3)
    plt.close()
    st.error("⚠️ Offer E: 67.91% churn — 39.51pp above benchmark. Offer A: 6.82%.")
    st.info("💡 Redesign or restrict Offer E. Expand Offer A to high-risk segments.")

    st.markdown("---")

    # Finding 4 - Protective Services
    st.subheader("4. Protective Services Create Switching Costs")
    protective_churn = df.groupby('Protective Services Count')['Churned'].mean() * 100
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.plot(protective_churn.index, protective_churn.values,
             marker='o', color='#2c3e50', linewidth=2.5, markersize=8)
    ax4.axhline(y=BENCHMARK, color='black', linestyle='--',
                label=f'Company Average {BENCHMARK}%')
    for x, y in zip(protective_churn.index, protective_churn.values):
        ax4.text(x, y + 1.5, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=9)
    ax4.set_xlabel("Number of Protective Services")
    ax4.set_ylabel("Churn Rate (%)")
    ax4.set_title("Churn Rate by Protective Services Count")
    ax4.legend()
    st.pyplot(fig4)
    plt.close()
    st.error("⚠️ 0 services: 63.65% → 4 services: 5.43%. Clear monotonic decline.")
    st.info("💡 Bundle and promote protective services to increase switching costs.")

    st.markdown("---")

    # Finding 5 - Internet Type
    st.subheader("5. Fiber Optic Customers Churn Most")
    internet_churn = df.groupby('Internet Type')['Churned'].mean() * 100
    fig5, ax5 = plt.subplots(figsize=(7, 4))
    vals5 = internet_churn.values
    ax5.bar(internet_churn.index, vals5, color=bar_colors(vals5))
    ax5.axhline(y=BENCHMARK, color='black', linestyle='--',
                label=f'Company Average {BENCHMARK}%')
    ax5.set_ylabel("Churn Rate (%)")
    ax5.set_title("Churn Rate by Internet Type")
    ax5.legend()
    annotate_bars(ax5, vals5)
    st.pyplot(fig5)
    plt.close()
    st.warning("⚠️ Fiber Optic customers show highest churn — investigate whether "
               "service quality or competitor offerings are driving dissatisfaction.")
    st.info("💡 Audit Fiber Optic service quality and benchmark against competitors.")

    st.markdown("---")

    # Finding 6 - Monthly Charge Quartiles
    st.subheader("6. Monthly Charge vs Churn Is Non-Monotonic")
    charge_churn = df.groupby('Monthly Charge Quartile')['Churned'].mean() * 100
    fig6, ax6 = plt.subplots(figsize=(7, 4))
    vals6 = charge_churn.values
    ax6.bar(charge_churn.index, vals6, color=bar_colors(vals6))
    ax6.axhline(y=BENCHMARK, color='black', linestyle='--',
                label=f'Company Average {BENCHMARK}%')
    ax6.set_xlabel("Monthly Charge Quartile")
    ax6.set_ylabel("Churn Rate (%)")
    ax6.set_title("Churn Rate by Monthly Charge Quartile")
    ax6.legend()
    annotate_bars(ax6, vals6)
    st.pyplot(fig6)
    plt.close()
    st.warning("⚠️ Churn peaks at Q3 (38.81%) then dips at Q4 (35%) — relationship "
               "is non-monotonic. Higher charge does not simply equal higher churn.")
    st.info("💡 Price reduction alone is not the answer. Investigate value perception "
            "and service quality — Q4 customers may receive greater bundled value.")

    st.markdown("---")

    # Finding 7 - Marital Status
    st.subheader("7. Unmarried Customers Churn at Higher Rates")
    married_churn = df.groupby('Married')['Churned'].mean() * 100
    fig7, ax7 = plt.subplots(figsize=(5, 4))
    vals7 = married_churn.values
    ax7.bar(married_churn.index, vals7, color=bar_colors(vals7))
    ax7.axhline(y=BENCHMARK, color='black', linestyle='--',
                label=f'Company Average {BENCHMARK}%')
    ax7.set_ylabel("Churn Rate (%)")
    ax7.set_title("Churn Rate by Marital Status")
    ax7.legend()
    annotate_bars(ax7, vals7)
    st.pyplot(fig7)
    plt.close()
    st.warning("⚠️ Unmarried: 36.72% vs Married: 20.22% — a 16.50pp difference.")
    st.info("💡 Directional finding — marital status may correlate with household "
            "stability and multi-line plans. Investigate bundling opportunities.")

    st.markdown("---")

    # Finding 8 - Churn Reason
    st.subheader("8. Competitor-Related Issues Drive Majority of Churn")
    churn_reason = (df[df['Churn Category'].notna()]
                   .groupby('Churn Category')['Churned']
                   .count()
                   .sort_values(ascending=False))
    churn_pct = churn_reason / churn_reason.sum() * 100
    fig8, ax8 = plt.subplots(figsize=(9, 4))
    ax8.bar(churn_pct.index, churn_pct.values, color='#e74c3c')
    ax8.set_ylabel("% of Churned Customers")
    ax8.set_title("Churn Category Distribution (Among Churned Customers Only)")
    for i, v in enumerate(churn_pct.values):
        ax8.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    st.pyplot(fig8)
    plt.close()
    st.error("⚠️ Competitor-related churn accounts for 45.2% of all churned customers.")
    st.info("💡 Historical finding only — churn category is unknown before a customer "
            "leaves. Use this to inform competitive pricing and offer strategy.")
    
    st.markdown("---")
    st.subheader("🎯 Top 5 Strategic Recommendations")
    st.success("""
    1. Early Tenure Intervention Program: 
    Customers in first 6 months have 77.01% churn rate. 
    Implement structured onboarding, check-in calls, and 
    early loyalty incentives within the first 90 days.
    """)
    st.success("""
    2. Contract Upgrade Incentives: 
    Month-to-Month customers churn at 51.65% vs 2.62% for Two Year. 
    Offer discounts or service upgrades to customers willing 
    to commit to longer contracts.
    """)
    st.success("""
    3. Redesign Offer E — Expand Offer A: 
    Offer E drives 67.91% churn — highest of all offers. 
    Investigate targeting and restructure or discontinue. 
    Expand Offer A (6.82% churn) to high-risk segments immediately.
    """)
    st.success("""
    4. Bundle Protective Services: 
    Each additional protective service reduces churn significantly 
    (63.65% → 5.43% from 0 to 4 services). 
    Create bundled packages at discounted rates to increase 
    switching costs and customer stickiness.
    """)
    st.success("""
    5. Fiber Optic Service Quality Audit: 
    Fiber Optic customers show disproportionately high churn. 
    Benchmark service quality and pricing against competitors. 
    Address dissatisfaction before it drives further revenue loss.
    """)

    st.info("""
    **Note:** These recommendations are derived from population-level 
    EDA patterns across 6,475 customers. Individual customer 
    interventions should be validated using the Customer Prediction 
    tool and business judgment.
    """)

    st.markdown("---")
    st.subheader("🧪 Proposed Next Step: A/B Test")
    st.write("""
    **Hypothesis:** Offering a targeted discount to Month-to-Month 
    customers in high-churn cities will increase contract upgrades 
    and reduce churn.

    **Test Design:**
    - **Target group:** Month-to-Month customers in high-churn cities 
    with tenure ≤ 12 months
    - **Offer:** 15% discount for 3 months upon signing a One Year contract
    - **Control group:** Similar customers receiving no targeted offer
    - **Success metric:** 30-day contract upgrade rate and 
    90-day retention rate
    - **Expected outcome:** If Offer A's 6.82% churn rate is 
    partially attributable to contract lock-in, targeted 
    incentives should measurably reduce Month-to-Month churn 
    in the test group
    """)

elif page == "🔍 Customer Prediction":
    st.title("🔍 Customer Prediction")
    st.markdown("---")
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure in Months", 0, 72, 12)
        monthly_charge = st.slider("Monthly Charge ($)", 20, 120, 65)
        contract = st.selectbox("Contract Type", 
                    ["Month-to-Month", "One Year", "Two Year"])
        age = st.slider("Age", 18, 80, 35)
    with col2:
        offer = st.selectbox("Offer", 
                    ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
        num_referrals = st.slider("Number of Referrals", 0, 11, 0)
        city = st.selectbox("City", sorted(df['City'].unique()))
        internet_type = st.selectbox("Internet Type", 
                    ["None", "Cable", "DSL", "Fiber Optic"])

    with st.expander("⚙️ Advanced Inputs (defaults used if unchanged)"):
        col3, col4 = st.columns(2)
        with col3:
            gender = st.selectbox("Gender", ["Female", "Male"])
            married = st.selectbox("Married", ["No", "Yes"])
            num_dependents = st.slider("Number of Dependents", 0, 9, 0)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["Yes", "No"])
            unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col4:
            avg_ld_charges = st.slider("Avg Monthly Long Distance Charges ($)", 0, 50, 20)
            payment_method = st.selectbox("Payment Method", 
                        ["Bank Withdrawal", "Credit Card", "Mailed Check"])
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection Plan", ["No", "Yes"])
            premium_tech = st.selectbox("Premium Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            streaming_music = st.selectbox("Streaming Music", ["No", "Yes"])

    if st.button("🔍 Predict Churn Risk", type="primary"):
        # Encode all inputs to match training feature matrix
        input_dict = {
            'Gender': 1 if gender == "Male" else 0,
            'Age': age,
            'Married': 1 if married == "Yes" else 0,
            'Number of Dependents': num_dependents,
            'Number of Referrals': num_referrals,
            'Tenure in Months': tenure,
            'Phone Service': 1 if phone_service == "Yes" else 0,
            'Avg Monthly Long Distance Charges': avg_ld_charges,
            'Multiple Lines': 1 if multiple_lines == "Yes" else 0,
            'Internet Service': 1 if internet_service == "Yes" else 0,
            'Unlimited Data': 1 if unlimited_data == "Yes" else 0,
            'Paperless Billing': 1 if paperless_billing == "Yes" else 0,
            'Monthly Charge': monthly_charge,
            'Online Security Int': 1 if online_security == "Yes" else 0,
            'Online Backup Int': 1 if online_backup == "Yes" else 0,
            'Device Protection Plan Int': 1 if device_protection == "Yes" else 0,
            'Premium Tech Support Int': 1 if premium_tech == "Yes" else 0,
            'Streaming TV Int': 1 if streaming_tv == "Yes" else 0,
            'Streaming Movies Int': 1 if streaming_movies == "Yes" else 0,
            'Streaming Music Int': 1 if streaming_music == "Yes" else 0,
            'High Churn City': 1 if city in high_churn_cities else 0,
            'Offer_Offer A': 1 if offer == "Offer A" else 0,
            'Offer_Offer B': 1 if offer == "Offer B" else 0,
            'Offer_Offer C': 1 if offer == "Offer C" else 0,
            'Offer_Offer D': 1 if offer == "Offer D" else 0,
            'Offer_Offer E': 1 if offer == "Offer E" else 0,
            'Internet Type_Cable': 1 if internet_type == "Cable" else 0,
            'Internet Type_DSL': 1 if internet_type == "DSL" else 0,
            'Internet Type_Fiber Optic': 1 if internet_type == "Fiber Optic" else 0,
            'Contract_One Year': 1 if contract == "One Year" else 0,
            'Contract_Two Year': 1 if contract == "Two Year" else 0,
            'Payment Method_Credit Card': 1 if payment_method == "Credit Card" else 0,
            'Payment Method_Mailed Check': 1 if payment_method == "Mailed Check" else 0
        }

        # Build DataFrame in exact column order model was trained on
        input_df = pd.DataFrame([input_dict])[feature_columns]

        # Scale using the same scaler from training
        input_scaled = scaler.transform(input_df)

        # Predict
        probability = model.predict_proba(input_scaled)[0][1]
        threshold = 0.30
        prediction = "High Risk" if probability >= threshold else "Low Risk"

        # Store results for Prediction Results page
        st.session_state['prediction'] = {
            'probability': probability,
            'prediction': prediction,
            'tenure': tenure,
            'contract': contract,
            'offer': offer,
            'city': city,
            'monthly_charge': monthly_charge
        }

        st.success("✅ Prediction complete! Navigate to 'Prediction Results' in the sidebar.")

elif page == "📋 Prediction Results":
    st.title("📋 Prediction Results")
    st.markdown("---")

    if 'prediction' not in st.session_state:
        st.warning("⚠️ No prediction made yet. Please go to Customer Prediction first.")
    else:
        p = st.session_state['prediction']
        probability = p['probability']
        prediction = p['prediction']

        # Main result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{probability:.1%}")
        with col2:
            st.metric("Threshold Used", "0.30")
        with col3:
            st.metric("Risk Level", prediction)

        st.markdown("---")

        # Risk alert
        if prediction == "High Risk":
            st.error(f"🚨 HIGH CHURN RISK — This customer has a {probability:.1%} probability of churning.")
        else:
            st.success(f"✅ LOW CHURN RISK — This customer has a {probability:.1%} probability of churning.")

        st.markdown("---")

        st.markdown("---")
        st.subheader("👤 Customer Profile Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Tenure:** {p['tenure']} months")
            st.write(f"**Contract:** {p['contract']}")
            st.write(f"**Offer:** {p['offer']}")
            st.write(f"**City:** {p['city']}")
        with col_b:
            st.write(f"**Monthly Charge:** ${p['monthly_charge']}")
            st.write(f"**Churn Probability:** {p['probability']:.1%}")
            st.write(f"**Risk Level:** {p['prediction']}")
            high_churn = "Yes" if p['city'] in high_churn_cities else "No"
            st.write(f"**High Churn City:** {high_churn}")

        st.markdown("---")

        # Business impact
        st.subheader("💰 Business Impact")
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Revenue at Risk (Annual)", 
                     f"${p['monthly_charge'] * 12:,.0f}")
        with col5:
            st.metric("Retention Call Cost", "$50")
        
        if prediction == "High Risk":
            st.warning(f"Cost of inaction: \${p['monthly_charge'] * 12:,.0f} annual revenue lost if this customer churns. Retention call costs only $50 — a {(p['monthly_charge'] * 12) / 50:.0f}x return if successful.")

        st.markdown("---")

        # Recommendation
        st.subheader("💡 Recommended Action")
        if prediction == "High Risk":
            if p['contract'] == "Month-to-Month":
                st.info("📋 **Contract Upgrade:** Offer incentive to switch to One Year or Two Year contract.")
            if p['offer'] == "Offer E" or p['offer'] == "None":
                st.info("🎁 **Offer Review:** Consider replacing with Offer A — historically associated with 6.82% churn vs 67.91% for Offer E.")
            if p['tenure'] <= 6:
                st.info("🤝 **Early Retention:** Customer is in highest-risk tenure group (0–6 months, 77.01% historical churn). Prioritize onboarding support.")
            st.info("📞 **Immediate Action:** Flag for retention team outreach.")
        else:
            st.info("👍 Customer appears stable. Continue standard engagement.")

elif page == "⚠️ EDA Context":
    st.title("⚠️ EDA Context")
    st.markdown("---")
    st.write("Population-level risk flags based on historical data analysis. These are not model explanations — they are patterns observed across 6,475 customers.")

    if 'prediction' not in st.session_state:
        st.warning("⚠️ No prediction made yet. Please go to Customer Prediction first.")
    else:
        p = st.session_state['prediction']
        st.subheader("Risk Flags for This Customer Profile")

        if p['tenure'] <= 6:
            st.error(f"🚨 Tenure = {p['tenure']} months → Customers in 0–6 month group have 77.01% historical churn rate — highest risk segment.")
        elif p['tenure'] <= 12:
            st.warning(f"⚠️ Tenure = {p['tenure']} months → 7–12 month group remains above company average churn rate.")
        else:
            st.success(f"✅ Tenure = {p['tenure']} months → Beyond early high-risk tenure window.")

        if p['contract'] == "Month-to-Month":
            st.error("🚨 Contract = Month-to-Month → Historically 51.65% churn rate vs company average 28.4%.")
        elif p['contract'] == "One Year":
            st.warning("⚠️ Contract = One Year → Lower risk than Month-to-Month but not as protected as Two Year.")
        else:
            st.success("✅ Contract = Two Year → Historically lowest churn at 2.62%.")

        if p['offer'] == "Offer E":
            st.error("🚨 Offer E → Historically associated with 67.91% churn — 39.51pp above benchmark.")
        elif p['offer'] == "None":
            st.warning("⚠️ No Offer → Customers with no offer churn at 29.27%, close to company average.")
        elif p['offer'] == "Offer A":
            st.success("✅ Offer A → Historically lowest churn at 6.82% — 21.58pp below benchmark.")

        if p['city'] in high_churn_cities:
            st.error(f"🚨 City = {p['city']} → Located in high-churn geographic cluster (Southern California region).")
        else:
            st.success(f"✅ City = {p['city']} → Not in identified high-churn geographic cluster.")

elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Model")
        st.success("✅ Logistic Regression @ Threshold 0.30")
        st.write("""
        **Why Logistic Regression over Random Forest:**
        
        Initial single train-test split showed Random Forest winning by $2,000 
        ($52,500 vs $54,500). However, this margin was insufficient evidence — 
        cross-validation was required to validate the conclusion.
        
        5-fold Stratified cross-validation reversed the result:
        """)
        st.metric("LR Mean CV Business Cost", "$58,540", delta="-$3,370 vs RF")
        st.metric("LR Cost Std Deviation", "$4,276", delta="Lower variance")
        st.metric("RF Mean CV Business Cost", "$61,910")
        st.metric("RF Cost Std Deviation", "$5,767")
        st.write("LR won 4 out of 5 CV folds. Selected for lower mean cost AND lower variance.")

        st.markdown("---")
        st.subheader("⚠️ Model Limitations")
        st.write("""
        - Model trained on historical California telecom data only — 
          may not generalize to other regions or time periods.
        - Churn reasons (competitor, dissatisfaction, etc.) are only 
          known after a customer leaves and are NOT used for prediction.
        - Revenue loss estimate assumes one full year of retained revenue 
          at the entered monthly charge.
        - High_Churn_City flag is based on cities with churn rate >20% 
          and minimum 30 customers — smaller cities may be misclassified.
        - Recommendations should complement business judgment, 
          not replace it.
        - Model reflects patterns as of training data — recommend 
          retraining every 6-12 months as customer behavior evolves.
        """)

    with col2:
        st.subheader("Model Performance @ Threshold 0.30")
        st.metric("Recall", "0.87")
        st.metric("False Negatives", "49", delta="Missed churners — cost $950 each")
        st.metric("False Positives", "159", delta="Unnecessary calls — cost $50 each")
        st.metric("Total Business Cost (test set)", "$54,500")

        st.subheader("Business Cost Framework")
        st.markdown("""
        **False Negative (Missed Churner): \\$950**

        * Median monthly charge = \\$72.90
        * Conservative retained lifetime = 13 months
        * *Calculation:* \\$72.90 × 13 = \\$947.70 (Rounded to \\$950)

        **False Positive (Unnecessary Retention Call): \\$50**

        * Agent call cost ≈ \\$30
        * Goodwill gesture/discount ≈ \\$20
        * *Calculation:* \\$30 + \\$20 = \\$50

        **The 19:1 Cost Asymmetry**

        This massive financial imbalance justifies lowering the model's decision threshold from 0.50 to 0.30 to prioritize recall over precision. Catching just **one** extra churner financially offsets **19** unnecessary retention calls.
        """)

        st.markdown("---")
        st.subheader("Feature Engineering")
        st.write("""
        - 33 features after encoding
        - High_Churn_City binary flag derived from EDA geographic analysis
        - Individual protective service columns retained over count 
          (each service has distinct churn impact)
        - One-hot encoding for Offer, Contract, Internet Type, Payment Method
        """)
        
