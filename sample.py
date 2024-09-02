import streamlit as st
import tempfile
import os

files = st.file_uploader("upload", type="pdf", accept_multiple_files=True)

temp_dir = tempfile.TemporaryDirectory()
st.write(temp_dir.name)

if files != None:
    file_dir = temp_dir.name
    # os.makedirs(file_dir, exist_ok=True)
    for file in files:
        with open(os.path.join(file_dir, file.name), "wb") as f:
            f.write(file.read())

    st.success("written")
    



