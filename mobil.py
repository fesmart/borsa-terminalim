import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from google import genai
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# API Anahtarın
API_KEY = st.secrets["GEMINI_API_KEY"]

# --- STREAMLIT SAYFA AYARLARI ---
st.set_page_config(page_title="AI Quant Terminal", page_icon="📈", layout="centered")

# --- HAFIZA (SESSION STATE) ---
# Web siteleri her butona basıldığında yenilendiği için, cüzdanımızın sıfırlanmaması adına hafızaya kazıyoruz.
if 'portfoy' not in st.session_state:
    st.session_state['portfoy'] = {}

# --- YAN MENÜ (CÜZDAN YÖNETİMİ) ---
st.sidebar.header("💼 Cüzdan Yönetimi")
st.sidebar.write("Hisselerini buradan ekle/çıkar.")

h_kodu = st.sidebar.text_input("Hisse Kodu (Örn: FROTO)").upper()
h_lot = st.sidebar.number_input("Lot Sayısı", min_value=0.0, step=1.0)
h_maliyet = st.sidebar.number_input("Maliyet", min_value=0.0, step=0.1)

if st.sidebar.button("➕ Portföye Ekle/Güncelle"):
    if h_kodu:
        kod = h_kodu.strip()
        if not kod.endswith('.IS') and kod not in ['SCHG', 'SCHD', 'AAPL', 'MSFT', 'NVDA', 'VOO']:
            kod += '.IS'
        st.session_state['portfoy'][kod] = {'lot': h_lot, 'maliyet': h_maliyet}
        st.sidebar.success(f"{kod} eklendi!")

st.sidebar.subheader("Mevcut Cüzdanın:")
if st.session_state['portfoy']:
    for k, v in st.session_state['portfoy'].items():
        st.sidebar.text(f"📌 {k} | Lot: {v['lot']} | Mal: {v['maliyet']}")
        if st.sidebar.button(f"❌ Sil {k}", key=f"sil_{k}"):
            del st.session_state['portfoy'][k]
            st.rerun()
else:
    st.sidebar.info("Cüzdanın şu an boş.")

# --- ANA EKRAN (ANALİZ MODÜLLERİ) ---
st.title("⚖️ Yapay Zeka & Portföy Terminali")

arama_kutusu = st.text_input("🔍 Yeni bir hisse keşfet (Örn: SCHG, TUPRS):")

# Butonları yan yana koymak için kolonlar oluşturuyoruz
col1, col2 = st.columns(2)

def analiz_motoru(hisseler, portfoy_modu=False):
    veriler = []
    toplam_deger = 0
    toplam_kar = 0
    
    # İlerleme çubuğu (Telefonda çok şık durur)
    progress_bar = st.progress(0)
    
    for i, hisse in enumerate(hisseler):
        try:
            sirket = yf.Ticker(hisse)
            hist = sirket.history(period="6mo")
            if hist.empty: continue
            
            fiyat = hist['Close'].iloc[-1]
            
            # Monte Carlo
            getiriler = hist['Close'].pct_change().dropna()
            mu, sigma = getiriler.mean(), getiriler.std()
            sim_df = pd.DataFrame([ [fiyat * (1 + sapma) for sapma in np.random.normal(mu, sigma, 30)] for _ in range(50) ]).T
            mc_beklenti = sim_df.mean(axis=1).iloc[-1]
            
            # Grafik Çizimi (Streamlit Uyumlu)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hist.index, hist['Close'], label='Geçmiş', color='blue')
            for x in range(sim_df.shape[1]):
                ax.plot(pd.date_range(hist.index[-1], periods=30, freq='B'), sim_df[x], color='green', alpha=0.03)
            ax.plot(pd.date_range(hist.index[-1], periods=30, freq='B'), sim_df.mean(axis=1), color='red', linestyle='--')
            ax.set_title(f"{hisse.replace('.IS', '')} - 30 Günlük MC Simülasyonu")
            ax.grid(True)
            st.pyplot(fig) # Grafiği web sitesine basar

            # Temel Veriler
            info = sirket.info
            fk = info.get('trailingPE', 'N/A')
            roe = info.get('returnOnEquity', 'N/A')
            sma50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) > 50 else "N/A"
            
            veri = {
                'Hisse': hisse.replace('.IS', ''),
                'Fiyat': round(fiyat, 2),
                'SMA50': round(sma50, 2) if isinstance(sma50, float) else sma50,
                'MC_Tahmin': round(mc_beklenti, 2),
                'F/K': round(fk, 2) if isinstance(fk, float) else fk,
                'ROE': f"%{round(roe * 100, 1)}" if isinstance(roe, float) and roe != 'N/A' else 'N/A'
            }

            if portfoy_modu and hisse in st.session_state['portfoy']:
                lot = st.session_state['portfoy'][hisse]['lot']
                mal = st.session_state['portfoy'][hisse]['maliyet']
                guncel_hacim = lot * fiyat
                kar_zarar = guncel_hacim - (lot * mal)
                toplam_deger += guncel_hacim
                toplam_kar += kar_zarar
                veri['Kâr/Zarar'] = round(kar_zarar, 2)

            veriler.append(veri)
        except: pass
        progress_bar.progress((i + 1) / len(hisseler))
        
    return veriler, toplam_deger, toplam_kar

# --- BUTON TETİKLEYİCİLERİ ---
with col1:
    if st.button("🔍 Bireysel Analiz", use_container_width=True):
        if arama_kutusu:
            arananlar = [h.strip().upper() for h in arama_kutusu.split(',')]
            hisseler = [h if h in ['SCHG', 'SCHD', 'AAPL', 'MSFT', 'NVDA', 'VOO'] else (h + '.IS' if not h.endswith('.IS') else h) for h in arananlar]
            
            with st.spinner("Bağımsız temel analiz yapılıyor..."):
                veriler, _, _ = analiz_motoru(hisseler, portfoy_modu=False)
                if veriler:
                    df = pd.DataFrame(veriler)
                    st.dataframe(df) # Tabloyu şık bir şekilde çizer
                    
                    prompt = f"Şu hisseleri tamamen bağımsız bir temel analiz uzmanı gibi incele. Teknik (Fiyat, SMA50) ve Temel (F/K, ROE) değerlerine bakarak iskontosunu, pahalılığını ve Monte Carlo (MC_Tahmin) verisine göre potansiyelini değerlendir:\n{df.to_string(index=False)}"
                    try:
                        client = genai.Client(api_key=API_KEY)
                        res = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                        st.success("🧠 Yapay Zeka Temel Analiz Raporu:")
                        st.write(res.text)
                    except Exception as e: st.error(f"Hata: {e}")

with col2:
    if st.button("💼 Portföyümü İncele", use_container_width=True):
        if not st.session_state['portfoy']:
            st.warning("⚠️ Cüzdanın boş! Sol menüden hisse ekle.")
        else:
            with st.spinner("Cüzdan toplanıyor ve fon analizi yapılıyor..."):
                hisseler = list(st.session_state['portfoy'].keys())
                veriler, top_deger, top_kar = analiz_motoru(hisseler, portfoy_modu=True)
                
                if veriler:
                    df = pd.DataFrame(veriler)
                    st.metric(label="💰 Toplam Cüzdan Büyüklüğü", value=f"{round(top_deger, 2)} BİRİM", delta=f"{round(top_kar, 2)} Kâr/Zarar")
                    st.dataframe(df)
                    
                    prompt = f"Sen bir fon yöneticisisin. Toplam Cüzdan: {round(top_deger, 2)} | Toplam Kâr/Zarar: {round(top_kar, 2)}. Buna göre cüzdan performansını, risk dağılımını ve stratejik önerilerini yaz:\n{df.to_string(index=False)}"
                    try:
                        client = genai.Client(api_key=API_KEY)
                        res = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                        st.info("🧠 Fon Yöneticisi Raporu:")
                        st.write(res.text)

                    except Exception as e: st.error(f"Hata: {e}")
