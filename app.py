import streamlit as st
import re
import pickle
import fitz

# Function to clean the resume text
def clean_resume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Function to make prediction
def predict_category(resume_text):
    cleaned_resume = clean_resume(resume_text)
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    return prediction_id

# Mapping category ID to category name
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Streamlit app
def main():
    st.title("Resume Category Prediction")
    st.write("Upload your resume (PDF or TXT) and we'll predict the category!")

    # File uploader
    uploaded_file = st.file_uploader("Upload Files", type=["pdf", "txt"])

    if uploaded_file is not None:
        file_contents = ""
        if uploaded_file.type == "application/pdf":
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                file_contents += page.get_text()
        else:
            file_contents = uploaded_file.getvalue().decode("utf-8")

        st.write("Resume Content:")
        st.text(file_contents)

        prediction_id = predict_category(file_contents)
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.markdown(f"<h2 style='color:white;'>Predicted Category: {category_name}</h2>", unsafe_allow_html=True)

    st.sidebar.title("Classifier Information")
    st.sidebar.write("This classifier has been trained to categorize resumes into various job roles based on their content. Below are the available categories:")

    # Display the category mapping in a user-friendly way
    for category_name in category_mapping.values():
        st.sidebar.write(f"- {category_name}")

    st.sidebar.write("The model has been trained on the following dataset: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset")
if __name__ == "__main__":
    main()
