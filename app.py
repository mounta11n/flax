import streamlit as st
import requests
import io


# Designing the interface
st.title("🖼️ Image Captioning Demo 📝")
st.write("[Yih-Dar SHIEH](https://huggingface.co/ydshieh)")

st.sidebar.markdown(
    """
    An image captioning model by combining ViT model with GPT2 model.
    The encoder (ViT) and decoder (GPT2) are combined using Hugging Face transformers' [Vision-To-Text Encoder-Decoder
    framework](https://huggingface.co/transformers/master/model_doc/visionencoderdecoder.html).
    The pretrained weights of both models are loaded, with a set of randomly initialized cross-attention weights.
    The model is trained on the COCO 2017 dataset for about 6900 steps (batch_size=256).
    [Follow-up work of [Huggingface JAX/Flax event](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/).]\n
    """
)

with st.spinner('Loading and compiling ViT-GPT2 model ...'):
    from model import *

random_image_id = get_random_image_id()

st.sidebar.title("Select a sample image")
sample_image_id = st.sidebar.selectbox(
    "Please choose a sample image",
    sample_image_ids
)

if st.sidebar.button("Random COCO 2017 (val) images"):
    random_image_id = get_random_image_id()
    sample_image_id = "None"

bytes_data = None
with st.sidebar.form("file-uploader-form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Choose a file")
    submitted = st.form_submit_button("Upload")
    if submitted and uploaded_file is not None:
        bytes_data = io.BytesIO(uploaded_file.getvalue())

if (bytes_data is None) and submitted:

    st.write("No file is selected to upload")

else:

    image_id = random_image_id
    if sample_image_id != "None":
        assert type(sample_image_id) == int
        image_id = sample_image_id

    sample_name = f"COCO_val2017_{str(image_id).zfill(12)}.jpg"
    sample_path = os.path.join(sample_dir, sample_name)

    if bytes_data is not None:
        image = Image.open(bytes_data)
    elif os.path.isfile(sample_path):
        image = Image.open(sample_path)
    else:
        url = f"http://images.cocodataset.org/val2017/{str(image_id).zfill(12)}.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    width, height = image.size
    resized = image.resize(size=(width, height))
    if height > 384:
        width = int(width / height * 384)
        height = 384
        resized = resized.resize(size=(width, height))
    width, height = resized.size
    if width > 512:
        width = 512
        height = int(height / width * 512)
        resized = resized.resize(size=(width, height))

    if bytes_data is None:
        st.markdown(f"[{str(image_id).zfill(12)}.jpg](http://images.cocodataset.org/val2017/{str(image_id).zfill(12)}.jpg)")
    show = st.image(resized)
    show.image(resized, '\n\nSelected Image')
    resized.close()

    # For newline
    st.sidebar.write('\n')

    with st.spinner('Generating image caption ...'):

        caption = predict(image)

        caption_en = caption
        st.header(f'Predicted caption:\n\n')
        st.subheader(caption_en)

    st.sidebar.header("ViT-GPT2 predicts: ")
    st.sidebar.write(f"{caption}")

    image.close()
