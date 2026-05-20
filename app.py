import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Simple training data - we hardcode a few examples to keep it mini
training_data = {
    'text': [
        'Win a free iPhone now click here',
        'You won 1,000,000 dollars claim prize',
        'Urgent: Your account will be closed',
        'Meeting at 3pm tomorrow in conference room',
        'Can we reschedule the project call',
        'Thanks for your email, see you Monday',
        'Limited time offer buy now 90% discount',
        'Hi mom, I will be home for dinner'
    ],
    'label': [1, 1, 1, 0, 0, 0, 1, 0] # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(training_data)

# 2. Build simple ML pipeline: text -> numbers -> model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# 3. Train the model on our small data
model.fit(df['text'], df['label'])

# 4. Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Paste email text below and I’ll predict if it’s spam")

user_input = st.text_area("Email content:", height=150,
                          placeholder="Congratulations! You won a free lottery...")

if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("Please enter some email text")
    else:
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0]

        if prediction == 1:
            st.error(f"🚨 This looks like SPAM! Confidence: {probability[1]*100:.1f}%")
        else:
            st.success(f"✅ This looks SAFE. Confidence: {probability[0]*100:.1f}%")

st.markdown("---")
st.caption("Mini project using Naive Bayes + Streamlit | Not 100% accurate - demo only")