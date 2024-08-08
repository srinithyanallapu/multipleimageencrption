import streamlit as st
from paste import Multiple_Image
from Decryption import decrypt

#st.title("Multiple-Image Encryption and Decryption")
# Sidebar menu
menu_option = st.sidebar.selectbox("Choose an option", ["Encrypt Image", "Decrypt Image"])

# Main content based on the selected option
if menu_option == "Encrypt Image":
    Multiple_Image()

elif menu_option == "Decrypt Image":
    decrypt()