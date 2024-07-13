import streamlit as st
import numpy as np
from transformers import AutoTokenizer, pipeline
import transformers
import torch
from googletrans import Translator

@st.cache(allow_output_mutation=True)
def get_model():
    model = "DumbKid-AI007/Espada-S1-7B"
tokenizer = AutoTokenizer.from_pretrained(model)
generator = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer,model = get_model()

user_input = st.text_area('enter text ')
button = st.button("send")

d= {
    1:'executed' ,
    2:'nt'
}

if user_input and button :
    test_sample = tokenizer([user_input],padding=True,truncation=True, max_length200,return_tensors='pt')
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.agrmax(output.logits.detach().numpy(),axis=1)
    st.write("Logits:",d[y_pred[0]])