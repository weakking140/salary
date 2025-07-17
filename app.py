import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_linear_model1.pkl')
        scaler = joblib.load('scaler1.pkl')
        return model, scaler
    except:
        st.error("Model files not found. Please ensure 'best_linear_model1.pkl' and 'scaler1.pkl' are in the same directory.")
        return None, None

# Main app
def main():
    st.title("💰 Salary Prediction App")
    st.markdown("---")
    
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("Enter Job Details")
    
    # Input fields based on your model features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Company Information")
        
        # Company Rating
        rating = st.slider("Company Rating (1-5)", 1.0, 5.0, 3.5, 0.1)
        
        # Company Age
        founded_year = st.number_input("Company Founded Year", 1900, 2025, 2000)
        current_year = 2025
        age = current_year - founded_year
        
        # Type of ownership
        ownership_options = ["Private", "Public", "Government", "Non-profit", "Other"]
        ownership = st.selectbox("Type of Ownership", ownership_options)
        
        # Industry
        industry_options = ["Technology", "Healthcare", "Finance", "Education", "Manufacturing", 
                           "Retail", "Consulting", "Other"]
        industry = st.selectbox("Industry", industry_options)
        
        # Sector
        sector_options = ["Information Technology", "Healthcare", "Financial Services", 
                         "Education", "Manufacturing", "Retail", "Business Services", "Other"]
        sector = st.selectbox("Sector", sector_options)
        
        # Company text/size indicator
        company_txt = st.slider("Company Size Score", 0, 100, 50)
        
    with col2:
        st.subheader("Job Information")
        
        # Job characteristics
        hourly = st.checkbox("Hourly Position")
        employer_provided = st.checkbox("Employer Provided Benefits")
        
        # Salary range inputs
        min_salary = st.number_input("Minimum Salary ($)", 30000, 300000, 50000)
        max_salary = st.number_input("Maximum Salary ($)", 35000, 500000, 80000)
        
        # Location
        job_state_options = ["CA", "NY", "TX", "FL", "IL", "WA", "MA", "Other"]
        job_state = st.selectbox("Job State", job_state_options)
        
        same_state = st.checkbox("Company HQ in Same State")
        
        # Experience level
        experience_options = ["Entry Level", "Mid Level", "Senior Level", "Executive"]
        experience_category = st.selectbox("Experience Level", experience_options)
        
        # Job type
        job_simp_options = ["Data Scientist", "Data Engineer", "Analyst", "Machine Learning Engineer", 
                           "Software Engineer", "Other"]
        job_simp = st.selectbox("Job Type", job_simp_options)
        
        # Description length
        desc_len = st.number_input("Job Description Length (characters)", 100, 5000, 1000)
        
        # Number of competitors
        num_comp = st.number_input("Number of Competitors", 0, 50, 5)
    
    # Technical skills section
    st.subheader("Technical Skills")
    col3, col4, col5, col6, col7 = st.columns(5)
    
    with col3:
        python_yn = st.checkbox("Python")
    with col4:
        r_yn = st.checkbox("R")
    with col5:
        spark = st.checkbox("Spark")
    with col6:
        aws = st.checkbox("AWS")
    with col7:
        excel = st.checkbox("Excel")
    
    # Additional features
    st.subheader("Additional Information")
    col8, col9 = st.columns(2)
    
    with col8:
        is_remote = st.checkbox("Remote Position")
        location_match = st.checkbox("Job Location Matches Company HQ")
    
    with col9:
        # Tech skills count
        tech_skills = sum([python_yn, r_yn, spark, aws, excel])
        st.write(f"**Tech Skills Count:** {tech_skills}")
    
    # Prediction button
    if st.button("🔮 Predict Salary", type="primary"):
        # Prepare input data
        input_data = prepare_input_data(
            rating, age, ownership, industry, sector, hourly, employer_provided,
            min_salary, max_salary, company_txt, job_state, same_state,
            python_yn, r_yn, spark, aws, excel, job_simp, desc_len, num_comp,
            experience_category, is_remote, location_match, tech_skills
        )
        
        # Make prediction
        prediction = make_prediction(model, scaler, input_data)
        
        # Display results
        display_results(prediction, input_data)

def prepare_input_data(rating, age, ownership, industry, sector, hourly, employer_provided,
                      min_salary, max_salary, company_txt, job_state, same_state,
                      python_yn, r_yn, spark, aws, excel, job_simp, desc_len, num_comp,
                      experience_category, is_remote, location_match, tech_skills):
    """Prepare input data for prediction"""
    
    # Create label encoders for categorical variables
    # Note: In a real application, you should save these encoders from training
    # For now, we'll use simple mapping
    
    ownership_map = {"Private": 0, "Public": 1, "Government": 2, "Non-profit": 3, "Other": 4}
    industry_map = {"Technology": 0, "Healthcare": 1, "Finance": 2, "Education": 3, 
                   "Manufacturing": 4, "Retail": 5, "Consulting": 6, "Other": 7}
    sector_map = {"Information Technology": 0, "Healthcare": 1, "Financial Services": 2,
                 "Education": 3, "Manufacturing": 4, "Retail": 5, "Business Services": 6, "Other": 7}
    state_map = {"CA": 0, "NY": 1, "TX": 2, "FL": 3, "IL": 4, "WA": 5, "MA": 6, "Other": 7}
    job_map = {"Data Scientist": 0, "Data Engineer": 1, "Analyst": 2, 
              "Machine Learning Engineer": 3, "Software Engineer": 4, "Other": 5}
    exp_map = {"Entry Level": 0, "Mid Level": 1, "Senior Level": 2, "Executive": 3}
    
    # Create input array matching the model's expected features
    input_array = np.array([
        0,  # Unnamed: 0 (index)
        rating,
        age + 1900,  # Founded year
        ownership_map.get(ownership, 4),
        industry_map.get(industry, 7),
        sector_map.get(sector, 7),
        int(hourly),
        int(employer_provided),
        min_salary,
        max_salary,
        company_txt,
        state_map.get(job_state, 7),
        int(same_state),
        age,
        int(python_yn),
        int(r_yn),
        int(spark),
        int(aws),
        int(excel),
        job_map.get(job_simp, 5),
        desc_len,
        num_comp,
        exp_map.get(experience_category, 0),
        int(is_remote),
        int(location_match),
        tech_skills
    ]).reshape(1, -1)
    
    return input_array

def make_prediction(model, scaler, input_data):
    """Make salary prediction"""
    try:
        # Note: Only scale the features that were scaled during training
        # Features to scale: age, Rating, desc_len, num_comp, min_salary, max_salary
        scaled_features = input_data.copy()
        
        # Scale specific columns (indices based on your original scaling)
        features_to_scale = [1, 13, 20, 21, 8, 9]  # Rating, age, desc_len, num_comp, min_salary, max_salary
        
        for i in features_to_scale:
            if i < scaled_features.shape[1]:
                # Simple scaling (you may need to adjust this based on your actual scaler)
                scaled_features[0, i] = (scaled_features[0, i] - scaler.mean_[i % len(scaler.mean_)]) / scaler.scale_[i % len(scaler.scale_)]
        
        # Make prediction
        prediction = model.predict(scaled_features)
        return prediction[0]
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def display_results(prediction, input_data):
    """Display prediction results"""
    if prediction is not None:
        # Create results display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Salary", f"${prediction:,.0f}")
        
        with col2:
            min_sal = input_data[0, 8]  # min_salary index
            max_sal = input_data[0, 9]  # max_salary index
            avg_range = (min_sal + max_sal) / 2
            difference = prediction - avg_range
            st.metric("Difference from Range Average", f"${difference:,.0f}")
        
        with col3:
            if prediction > avg_range:
                st.success("Above Average Range! 📈")
            else:
                st.info("Below Average Range 📊")
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Min Salary', 'Predicted Salary', 'Max Salary'],
            y=[min_sal, prediction, max_sal],
            marker_color=['lightblue', 'gold', 'lightcoral'],
            text=[f'${min_sal:,.0f}', f'${prediction:,.0f}', f'${max_sal:,.0f}'],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Salary Prediction vs Range',
            xaxis_title='Salary Type',
            yaxis_title='Salary ($)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.subheader("💡 Insights")
        
        tech_skills_count = input_data[0, 25]  # tech_skills index
        experience = input_data[0, 22]  # experience_category index
        
        insights = []
        
        if tech_skills_count >= 3:
            insights.append("✅ Strong technical skill set may positively impact salary")
        elif tech_skills_count <= 1:
            insights.append("⚠️ Limited technical skills may affect salary potential")
        
        if experience >= 2:  # Senior level or Executive
            insights.append("✅ Senior experience level typically commands higher salaries")
        
        if input_data[0, 23]:  # is_remote
            insights.append("🏠 Remote position may offer competitive compensation")
        
        for insight in insights:
            st.write(insight)
    
    else:
        st.error("Unable to make prediction. Please check your inputs.")

# Footer
def add_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Salary Prediction Model</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    add_footer()