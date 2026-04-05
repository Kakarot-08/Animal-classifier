import pickle

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

st.set_page_config(page_title="Animal Classifier", layout="centered")

st.markdown(
    """
    <style>
        .main { background-color: #f5f5f5; }
        .result-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #ddd;
            margin-top: 20px;
        }
        .label-text {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
            margin-top: 10px;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ── Model ────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128 * 16 * 16), 128)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(128, 3)

    def forward(self, x):
        x = self.relu(self.pooling(self.conv1(x)))
        x = self.relu(self.pooling(self.conv2(x)))
        x = self.relu(self.pooling(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        return self.output(x)


@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("animal-classifier.pth", map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
label_encoder = load_encoder()

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ]
)


# ── UI ───────────────────────────────────────────────────────
st.title("Animal Classifier")
st.write("Upload gambar hewan, model akan menebak jenisnya.")

st.divider()

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

    with col2:
        with st.spinner("Menganalisis..."):
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred = torch.argmax(output, axis=1).item()
                label = label_encoder.inverse_transform([pred])[0]
                confidence = probs[pred].item() * 100

        st.markdown(
            f"""
            <div class="result-box">
                <p style="color:#888;">Hasil Prediksi</p>
                <div class="label-text">{label.upper()}</div>
                <p style="color:#888;">Confidence: {confidence:.1f}%</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.write("Probabilitas semua kelas:")
        for i, cls in enumerate(label_encoder.classes_):
            prob = probs[i].item() * 100
            st.progress(int(prob), text=f"{cls}: {prob:.1f}%")

else:
    st.info("Upload gambar untuk memulai.")
