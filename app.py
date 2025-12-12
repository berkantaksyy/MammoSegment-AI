import streamlit as st
import cv2
import numpy as np
import math
import base64
from io import BytesIO

# Sayfa başlığı ve temel düzen ayarları
st.set_page_config(
    page_title="MammoSegment AI",
    page_icon="🧬",
    layout="wide"
)

# Proje bağlantıları ve sabit dosya yolları
LINKEDIN_URL = "https://www.linkedin.com/in/berkantaksyy/"
SCHOLAR_URL = "https://ubs.ikc.edu.tr/ABPDS/AcademicInformation/BilgiGoruntulemev2/Index?pid=EIhzKKxabFdJ93CizPaPJg!xGGx!!xGGx!" 
GITHUB_URL = "https://github.com/berkantaksyy" 
PDF_FILE_NAME = "210402043_final_project.pdf" 

# --- YARDIMCI FONKSİYONLAR ---

# Binary dosyaları arka plan videosu için base64 formatına çevir
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# Arka plan videosunu sayfaya yerleştirme işlemi
def set_background(video_file):
    video_base64 = get_base64_of_bin_file(video_file)
    if video_base64:
        video_tag = f'''
        <style>
        #myVideo {{
            position: fixed; right: 0; bottom: 0;
            min-width: 100%; min-height: 100%;
            z-index: -1; opacity: 0.4;
        }}
        .stApp {{ background: rgba(0,0,0,0.85); }}
        </style>
        <video autoplay muted loop id="myVideo">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        '''
        st.markdown(video_tag, unsafe_allow_html=True)
    else:
        # Video yüklenemezse varsayılan koyu tema
        st.markdown('<style>.stApp {background-color: #0e1117;}</style>', unsafe_allow_html=True)

# OpenCV görüntüsünü Streamlit'te göstermek/indirmek için buffer'a alma
def convert_image(img_array):
    is_success, buffer = cv2.imencode(".png", img_array)
    io_buf = BytesIO(buffer)
    return io_buf

# Proje raporunu (PDF) iframe içinde gösterme fonksiyonu
def show_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf" style="border-radius:10px;"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"PDF dosyası '{file_path}' dizinde bulunamadı.")

# --- ARAYÜZ TASARIMI (CSS) ---
# Modern ve koyu tema için özel CSS tanımlamaları
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        color: #ffffff;
    }
    
    /* Streamlit varsayılan elemanlarını gizle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;} 
    
    /* Üst bilgi çubuğu stili */
    .academic-header {
        position: fixed; top: 0; left: 0; width: 100%;
        padding: 15px 0;
        background: linear-gradient(to bottom, #000000, rgba(0,0,0,0.9), transparent);
        z-index: 999;
        text-align: center;
        font-size: 13px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .link-style {
        color: #4cc9f0; text-decoration: none; font-weight: 700;
        transition: all 0.3s ease;
    }
    .link-style:hover { color: #fff; text-shadow: 0 0 10px rgba(76, 201, 240, 1); }
    .supervisor-title { color: #ccc; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }

    /* Glassmorphism efekti (paneller için) */
    .glass-panel {
        background: rgba(12, 16, 23, 0.80);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(30px);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.6);
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Upload butonu özelleştirmesi */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Buton stilleri */
    div.stButton > button {
        background: #4cc9f0; color: #000; border: none; padding: 14px 24px;
        border-radius: 6px; font-weight: 800; letter-spacing: 1px;
        width: 100%; transition: 0.3s; text-transform: uppercase; font-size: 14px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 0 20px rgba(76, 201, 240, 0.6); background: #fff;
    }
    
    .stDownloadButton > button {
        background-color: transparent !important; border: 1px solid rgba(255,255,255,0.3) !important;
        color: #ccc !important; border-radius: 6px; font-weight: 500;
        transition: 0.3s; text-transform: uppercase; font-size: 12px; width: 100%;
    }
    .stDownloadButton > button:hover {
        border-color: #4cc9f0 !important; color: #4cc9f0 !important;
        background-color: rgba(76, 201, 240, 0.1) !important;
    }
    
    h1 { font-weight: 300; letter-spacing: -0.5px; margin-bottom: 0; font-size: 2.2rem; }
    .ai-text { color: #4cc9f0; font-weight: 700; }
    h3 { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-bottom: 15px; font-weight: 600; text-align: left; }
</style>
""", unsafe_allow_html=True)

# Arka plan videosunu yükle
set_background('background.mp4') 

# Sayfa üstü bilgi çubuğunu oluştur
st.markdown(f"""
<div class="academic-header">
    <div style="margin-bottom: 5px;">
        Designed by: <a href="{LINKEDIN_URL}" target="_blank" class="link-style">Berkant AKSOY</a>
    </div>
    <div class="supervisor-title">
        Supervisor: 
        <a href="{SCHOLAR_URL}" target="_blank" class="link-style">Asst. Prof. Dr. Onan GÜREN</a>
    </div>
</div>
<div style="height: 80px;"></div>
""", unsafe_allow_html=True)

# --- GÖRÜNTÜ İŞLEME ALGORİTMASI ---
def process_pipeline(image_file):
    # Görüntüyü stream'den okuyup grayscale moda çeviriyoruz
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Ön İşleme: CLAHE ile kontrast dengeleme
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    # Gürültü azaltmak için Median Blur uyguluyoruz
    img_blur = cv2.medianBlur(img_clahe, 5)
    
    # Otsu metodu ile otomatik eşikleme (Doku ayrımı için)
    _, thresh = cv2.threshold(img_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morfolojik işlemlerle maskeyi temizleme (Açma ve Kapama)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    final_mask = mask.copy()
    
    # Pektoral kası tespit etmek için referans noktası bulma
    scan_row = mask[5, :] 
    white_pixels = np.where(scan_row > 0)[0]
    # Eğer beyaz piksel varsa sonuncusunu al, yoksa genişliğin %30'unu varsay
    x_cut = white_pixels[-1] if len(white_pixels) > 0 else int(w * 0.3)
    
    # Kasın bitişi olabilecek tahmini Y koordinatı
    y_cut = int(x_cut * 2.5)
    if y_cut > h * 0.6: y_cut = int(h * 0.6)
    
    # Canny kenar tespiti için bölge sınırlama
    canny_mask = np.zeros_like(mask)
    cv2.rectangle(canny_mask, (0,0), (x_cut + 100, y_cut + 100), 255, -1)
    
    masked_img = cv2.bitwise_and(img_blur, img_blur, mask=mask)
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask=canny_mask)
    
    # Kenar tespiti
    edges = cv2.Canny(masked_img, 30, 100)
    
    # Hough dönüşümü ile kas çizgisini bulma
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=30, maxLineGap=20)
    cut_performed = False
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    if lines is not None:
        best_line = None
        max_score = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Sadece belirli açılardaki çizgileri kabul et (Pektoral kasın anatomik duruşu)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -85 < angle < -20:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > max_score:
                    max_score = length
                    best_line = (x1, y1, x2, y2)
        
        # En iyi çizgiyi bulduysak maskeden o bölgeyi çıkar
        if best_line:
            x1, y1, x2, y2 = best_line
            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                # Poligon oluşturup maskeden siliyoruz
                poly_pts = np.array([[0, 0], [w, 0], [w, int(m*w + c)], [0, int(c)]])
                cv2.fillPoly(final_mask, [poly_pts], 0)
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cut_performed = True

    # Hough çizgisi bulunamazsa manuel kesim yap
    if not cut_performed:
        pts = np.array([[0,0], [x_cut, 0], [0, y_cut]])
        cv2.fillPoly(final_mask, [pts], 0)
        cv2.line(debug_img, (x_cut, 0), (0, y_cut), (255, 0, 0), 5)

    # Görüntünün alt kısmındaki artefaktları temizle
    target_y = 2600
    if h > target_y: final_mask[target_y:, :] = 0
    
    # En büyük bileşeni (meme dokusunu) seç, gerisini at
    num_labels, labels_im = cv2.connectedComponents(final_mask)
    if num_labels > 1:
        areas = np.bincount(labels_im.flatten())[1:]
        largest_label = np.argmax(areas) + 1
        final_mask = np.zeros_like(final_mask)
        final_mask[labels_im == largest_label] = 255
        
    return img, debug_img, final_mask

# --- SAYFA DÜZENİ (LAYOUT) ---
# Ekranı iki ana sütuna böl: Sol (Bilgi), Sağ (Uygulama)
col_info, col_app = st.columns([4, 6], gap="large")

# --- SOL SÜTUN: PROJE BİLGİLERİ ---
with col_info:
    # Proje Özeti
    st.markdown("""
    <div class="glass-panel">
        <h3>PROJECT OVERVIEW</h3>
        <p style="font-size: 13px; color: #ccc; line-height: 1.6;">
        <b>MammoSegment AI</b> is an advanced biomedical tool designed for automated pectoral muscle removal and breast tissue isolation in MLO mammograms.
        <br><br>
        <b>Core Methodology:</b><br>
        • <b>CLAHE:</b> Contrast enhancement<br>
        • <b>Otsu's Thresholding:</b> Initial segmentation<br>
        • <b>Hough Transform:</b> Geometric muscle detection
        </p>
        <a href="{}" target="_blank" style="text-decoration: none;">
            <div style="background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); text-align: center; color: #4cc9f0; font-size: 12px; transition: 0.3s; margin-top: 10px;">
                📂 View Source Code on GitHub
            </div>
        </a>
    </div>
    """.format(GITHUB_URL), unsafe_allow_html=True)
    
    # PDF Görüntüleyici
    st.markdown('<div class="glass-panel" style="padding: 15px;"><h3>PROJECT REPORT</h3>', unsafe_allow_html=True)
    show_pdf(PDF_FILE_NAME)
    st.markdown('</div>', unsafe_allow_html=True)

# --- SAĞ SÜTUN: ANALİZ UYGULAMASI ---
with col_app:
    # Uygulama Başlığı
    st.markdown("""
    <div class="glass-panel" style="text-align: center;">
        <h1>MammoSegment <span class="ai-text">AI</span></h1>
        <p style="color: #888; letter-spacing: 3px; font-size: 10px; margin-top: 8px; font-weight: 500;">
            AUTOMATED MEDICAL IMAGE ANALYSIS
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dosya Yükleme Alanı
    st.markdown('<div class="glass-panel" style="text-align: left;"><h3>INPUT DATA</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload MLO Mammogram", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
        if st.button("ANALYZE SCAN"):
            with st.spinner('Calculating Segmentation...'):
                img, debug, mask = process_pipeline(uploaded_file)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Sonuçların Gösterimi
                st.markdown('<div class="glass-panel"><h3>ANALYSIS REPORT</h3>', unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.image(img, caption="Original", use_container_width=True)
                with c2: st.image(debug, caption="Boundaries", use_container_width=True)
                with c3: st.image(mask, caption="Mask", use_container_width=True)
                
                st.markdown("---")
                mask_file = convert_image(mask)
                st.download_button(label="DOWNLOAD MASK (PNG)", data=mask_file, file_name="segmented_mask.png", mime="image/png")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)

# Alt Bilgi (Footer)
st.markdown("""
<div style="position: fixed; bottom: 10px; width: 100%; text-align: center; font-size: 9px; color: rgba(255,255,255,0.3); letter-spacing: 1px;">
    DEVELOPED BY BERKANT AKSOY | IZMIR KATIP CELEBI UNIVERSITY
</div>
""", unsafe_allow_html=True)