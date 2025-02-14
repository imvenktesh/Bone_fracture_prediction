# Import necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Load the trained model
model = load_model('bone_fracture_cnn.h5')

# Set the title and description of the app
st.title("Bone Fracture Detection App")
st.write("Upload an X-ray image to detect the presence of a bone fracture.")

# Sidebar for navigation
st.sidebar.header("Navigation")
st.sidebar.write("This app uses a deep learning model to detect bone fractures from X-ray images. "
                 "Please ensure that the uploaded image is clear and in the correct format.")

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale if needed
    img = img.resize((373, 454))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to the range [0, 1]
    return img_array

# File uploader widget
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_column_width=True)
    
    # Preprocess the image
    processed_img = preprocess_image(img)

    # Predict the class
    prediction = model.predict(processed_img)[0][0]

    # Plot the prediction probabilities as a bar graph
    labels = ['No Fracture', 'Fracture']
    probabilities = [prediction, 1 - prediction]  # Inverse probability for 'Fracture'
    
    # Display the bar graph
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities, color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probability')
    
    st.pyplot(fig)

    # Display the result based on the prediction
    if prediction > 0.5:
        st.success("Prediction: **No Fracture Detected**")
    else:
        st.error("Prediction: **Fracture Detected**")

    # Detailed explanation of results (optional, for a professional touch)
    st.markdown("""
    ### What does this mean?
    - **No Fracture Detected**: The model predicts that the image does not show evidence of a bone fracture.
    - **Fracture Detected**: The model predicts that the image shows evidence of a bone fracture.
    
    ### Disclaimer
    This tool is designed for educational purposes only and is not a substitute for professional medical advice. 
    Always consult a healthcare provider for an accurate diagnosis.
    """)

# Add footer with professional note
st.sidebar.info("Developed for clinical support and research purposes.")

# Additional developer details in the sidebar
st.sidebar.write('**© Built by: Venkatesh Sahadevan**')
st.sidebar.write('**LinkedIn** [Venkatesh S](https://www.linkedin.com/in/venkatesh-s-78554a29a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app )')
st.sidebar.write('**GitHub:** [imvenktesh](https://github.com/vkvenkat18/)')
