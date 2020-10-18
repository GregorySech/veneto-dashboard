import streamlit as st
from fastai.vision.all import PILImage, Learner, load_learner
from fastcore.dispatch import TypeDispatch


@st.cache(hash_funcs={TypeDispatch: hash}, allow_output_mutation=True)
def load_model(path: str) -> Learner:
    return load_learner(path)


'''
# Veneto Dashboard

Some interesting facts and data driven applications all about Veneto.  
This dashboard is a toy application that I'm using to familiarize myself with 
[Streamlit](https://www.streamlit.io/) for creating machine learning tools 
quickly.  

## Introduction
[**Veneto**](https://en.wikipedia.org/wiki/Veneto) 
is the fifth, out of twenty, most populated region of Italy.
The capital (capoluogo) of Veneto is [Venice](https://en.wikipedia.org/wiki/Venice).  
'''

st.sidebar.write("# Applications")

'''
### Murano's Vase Checker
This is a toy classifier implemented using a 
[ResNet](https://arxiv.org/abs/1512.03385) that will tell you if the 
photo of a vase is a Murano piece or not.  
It's by no way an application that I would trust in a
professional environment but the validated error rate of the classifier is 3%.
This validation metric was calculated on a random 70-30 split of the whole data
set that I've created scraping through various websites on the internet.  
I'm somewhat confident on the classifier working but the low error rate might be 
affected by the quality of the data I'm using so it might not be a trust-worthy 
estimator of the model's real-world performance.  

I will work on a mechanism for people that use this application to report a 
miss-classification as soon as it will be safe to do so. Right now there is 
no way of having backend secrets in a shared Streamlit application.
'''

st.sidebar.write("## Murano's Vase Checker")
uploaded_vase_image = st.sidebar.file_uploader("Upload vase image",
                                               type=['png', 'jpg'])
if uploaded_vase_image is not None:
    image = PILImage.create(uploaded_vase_image)
    learner = load_model('models/murano_vases/resnet50.pkl')
    label, index, probs = learner.predict(image)
    if label == 'not_murano':
        st.write("This **doesn't** seem like a Venetian vase!")
    else:
        st.write("Looks like a Venetian vase to me!")
    st.image(image, width=240)
