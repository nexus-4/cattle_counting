import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os
import time
import random

# Configurações avançadas
class TrackerConfig:
    FRACTAL_SCALES = [32, 64]
    MAX_FEATURES = 5
    MIN_CONFIDENCE = 0.4
    DRONE_ANGLE_CHANGE_THRESH = 0.3

# Carregar modelo YOLO (com cache)
@st.cache_resource
def load_model():
    return YOLO('yolov10x.pt')

# Classe principal de tracking
class AdvancedCattleTracker:
    def __init__(self):
        self.reset_tracker()

    def reset_tracker(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker = DeepSort(
            max_age=st.session_state.get('max_age', 70),
            n_init=5,
            nn_budget=100,
            max_iou_distance=0.7
        )
        self.cow_database = {}
        self.color_map = {}
        self.total_count = 0
        self.current_ids = set()
        self.prev_frame = None
        self.angle_change = False
        self.id_history = set()
        self.fractal_signatures = {} # Dicionário para armazenar assinaturas fractais
        
    def detect_angle_change(self, current_frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return False
            
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        self.angle_change = np.mean(magnitude) > st.session_state.get('angle_sensitivity', 0.3)
        self.prev_frame = current_gray
        
        return self.angle_change
    
    def update(self, frame, detections, frame_idx):
        self.detect_angle_change(frame)
        self.current_ids = set()
        
        self.tracker.max_age = st.session_state.get('max_age', 70)
            
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            self.current_ids.add(track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            
            current_feat = self._extract_robust_features(frame, x1, y1, x2, y2)
            
            # Verifica se a vaca já está no banco de dados
            if track_id in self.cow_database:
                continue
            # Se não estiver, tenta encontrar uma correspondência
            if track_id not in self.cow_database:
                matched_id = self._find_match(current_feat)
                if matched_id:
                    self._transfer_id(track_id, matched_id)
                else:
                    self.total_count += 1
                    self._init_new_cow(track_id, current_feat, frame_idx)
                    self.id_history.add(track_id)  # Adiciona o ID ao histórico
            
            self._update_cow_features(track_id, current_feat, frame_idx)
            self._draw_enhanced_box(frame, track_id, x1, y1, x2, y2)
        
        self._clean_old_ids(frame_idx)
        
        # Adiciona texto no canto superior esquerdo
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Vacas: {self.total_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame

    def _init_new_cow(self, track_id, features, frame_idx):
        self.cow_database[track_id] = {
            'features': [features] if features else [],
            'last_seen': frame_idx
        }
        self.color_map[track_id] = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200))
    
    def _update_cow_features(self, track_id, features, frame_idx):
        if features:
            self.cow_database[track_id]['features'].append(features)
            self.cow_database[track_id]['last_seen'] = frame_idx
            if len(self.cow_database[track_id]['features']) > st.session_state.get('max_features', 5):
                self.cow_database[track_id]['features'].pop(0)
    
    def _extract_robust_features(self, frame, x1, y1, x2, y2):
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Converter ROI para escala de cinza
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Calcular a Transformada de Fourier
        f_transform = np.fft.fft2(gray)
        f_transform_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shift)

        # Converter para escala logarítmica para evitar valores extremos
        magnitude_spectrum_log = np.log1p(magnitude_spectrum)

        # Criar vetor de frequências e calcular expoente do espectro fractal
        rows, cols = magnitude_spectrum_log.shape
        cx, cy = cols // 2, rows // 2  # Centro da imagem na frequência
        radii = np.sqrt((np.arange(rows)[:, None] - cy) ** 2 + (np.arange(cols) - cx) ** 2)
        log_radii = np.log1p(radii.flatten())
        log_magnitude = magnitude_spectrum_log.flatten()

        # Ajuste de reta para estimar expoente fractal
        finite_mask = np.isfinite(log_magnitude) & np.isfinite(log_radii)  # Evita valores NaN
        if np.sum(finite_mask) > 10:  # Garante dados suficientes para ajuste
            coeffs = np.polyfit(log_radii[finite_mask], log_magnitude[finite_mask], 1)
            fractal_exponent = coeffs[0]  # Inclinação da reta ajustada
        else:
            fractal_exponent = 0  # Valor neutro caso não haja dados suficientes

        # Histogramas de cores
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Momentos de Hu para forma
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()

        rel_size = (x2-x1)*(y2-y1)/(frame.shape[0]*frame.shape[1])

        return {
            'hist': hist,
            'hu': hu_moments,
            'size': rel_size,
            'position': ((x1+x2)/2/frame.shape[1], (y1+y2)/2/frame.shape[0]),
            'fractal_exponent': fractal_exponent  # Adiciona a textura fractal ao conjunto de features
        }

    
    def _find_match(self, current_feat):
        if current_feat is None:
            return None
            
        min_similarity = st.session_state.get('similarity_thresh', 0.7)
        if self.angle_change:
            min_similarity *= 0.9
            
        for tid, data in self.cow_database.items():
            if tid in self.current_ids:
                continue
                
            similarity = self._compare_features(current_feat, data['features'][-1])
            if similarity > min_similarity:
                return tid
        return None
    
    def _compare_features(self, feat1, feat2):
        if None in (feat1, feat2):
            return 0

        hist_sim = cv2.compareHist(feat1['hist'], feat2['hist'], cv2.HISTCMP_BHATTACHARYYA)
        hu_diff = np.linalg.norm(feat1['hu'] - feat2['hu'])
        size_diff = abs(feat1['size'] - feat2['size'])

        fractal_diff = abs(feat1['fractal_exponent'] - feat2['fractal_exponent'])  # Diferença de textura

        return 0.4*(1-hist_sim) + 0.25*(1-hu_diff) + 0.2*(1-size_diff) + 0.15*(1-fractal_diff)

    
    def _transfer_id(self, new_id, old_id):
        self.cow_database[new_id] = self.cow_database.pop(old_id)
        self.color_map[new_id] = self.color_map.pop(old_id)
    
    def _clean_old_ids(self, current_frame):
        """Remove IDs não vistos há muito tempo"""
        max_age = st.session_state.get('max_age', 60)
        to_delete = [tid for tid, data in self.cow_database.items() 
                    if tid not in self.current_ids and 
                    (current_frame - data['last_seen']) > max_age]
        for tid in to_delete:
            del self.cow_database[tid]
            del self.color_map[tid]
            
    
    def _draw_enhanced_box(self, frame, track_id, x1, y1, x2, y2):
        color = self.color_map.get(track_id, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5, lineType=cv2.LINE_AA)
        
        label = f"ID:{track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1-h-5), (x1+w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, lineType=cv2.LINE_AA)

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = "cattle_tracking_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = AdvancedCattleTracker()
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, classes=[19], conf=st.session_state.get('confidence', 0.4), verbose=False)
        detections = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                detections.append(([x1, y1, x2-x1, y2-y1], conf, None))
        
        processed_frame = tracker.update(frame.copy(), detections, frame_count)
        out.write(processed_frame)
        
        # Aumenta o tamanho da visualização (75% da largura original)
        display_width = int(width * 0.75)
        display_height = int(height * 0.75)
        display_frame = cv2.resize(processed_frame, (display_width, display_height))
        
        frame_placeholder.image(display_frame, channels="BGR")
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processando frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    return tracker.total_count, output_path

def main():
    st.title("🐄 Sistema Avançado de Rastreamento de Gado")
    
    model = load_model()
    
    with st.sidebar:
        st.header("Configurações")
        uploaded_video = st.file_uploader("Carregar vídeo", type=["mp4", "mov", "avi"])
        
        st.subheader("Detecção")
        st.slider("Confiança mínima", 0.1, 0.9, 0.4, 0.01, key='confidence')
        
        st.subheader("Reidentificação")
        st.slider("Similaridade mínima", 0.1, 0.9, 0.7, 0.01, key='similarity_thresh')
        st.slider("Histórico de features", 1, 10, 5, 1, key='max_features')
        
        st.subheader("Tracking")
        st.slider("Frames até perder tracking", 10, 120, 60, 1, key='max_age')
        st.slider("Sensibilidade a ângulo", 0.1, 1.0, 0.3, 0.01, key='angle_sensitivity')
    
    if uploaded_video:
        if st.button("Iniciar Processamento"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            try:
                start_time = time.time()
                total, output_path = process_video(video_path, model)
                
                st.success(f"""
                **Processamento concluído!**  
                ⏱️ Tempo: {time.time()-start_time:.1f}s  
                🐄 Vacas detectadas: {total}  
                """)
                
                st.subheader("Vídeo Processado")
                st.video(output_path)
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Baixar Vídeo Processado",
                        data=f,
                        file_name="gado_rastreado.mp4",
                        mime="video/mp4"
                    )
                
            except Exception as e:
                st.error(f"Erro durante o processamento: {str(e)}")
                st.exception(e)
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
                if 'output_path' in locals() and os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass

if __name__ == "__main__":
    main()