import cv2
import streamlit as st
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("isaretDili_best.pt")

# Streamlit başlık
st.title("İşaret Dili Tespiti")

# Stop button state
if 'stop' not in st.session_state:
    st.session_state.stop = False

def stop_stream():
    st.session_state.stop = True

# Durdur butonu
st.button("Durdur", on_click=stop_stream)

# Web kamerası akışını başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan web kamerasını belirtir

if not cap.isOpened():
    st.error("Web kamerası açılamadı!")
else:
    stframe = st.empty()
    
    while not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Kare alınamadı, çıkılıyor...")
            break

        # YOLO modelini kullanarak tahmin yap
        results = model(frame)

        # Sonuçları çerçeveye çiz
        annotated_frame = results[0].plot()

        # Çerçeveyi Streamlit ile göster
        stframe.image(annotated_frame, channels="BGR")

# Her şeyi serbest bırak
cap.release()
cv2.destroyAllWindows()
