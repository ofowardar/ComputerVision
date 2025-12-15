# 📚 Görüntü İşleme Ders Notları - Nesne Sayma Sistemi

## 1️⃣ TEMEL KAVRAMLAR

### Kütüphaneler Nedir?
Hazır kod parçalarının toplandığı kütüphaneleri projemizde kullanırız:

```python
import cv2                    # OpenCV - Görüntü işleme kütüphanesi
from ultralytics import YOLO  # YOLOv8 - Nesne algılama modeli
```

- **cv2**: Görüntü işlemek, kamera kontrol etmek, çizim yapmak için
- **YOLO**: Görüntüdeki nesneleri otomatik tanımak için

---

## 2️⃣ ADIM 1: MODELİ YÜKLEMEK

```python
model = YOLO('yolov8n.pt')
```

**Ne işe yarar?**
- YOLOv8n (nano) adlı önceden eğitilmiş bir model yüklüyoruz
- Bu model 80 farklı nesne türünü tanıyabiliyor
- `.pt` uzantısı PyTorch ağırlık dosyasıdır

**Modelin Özellikleri:**
- `n` = nano (en küçük) - hızlı işlem
- Daha fazla doğruluk istersen: `yolov8s`, `yolov8m`, `yolov8l` kullan

---

## 3️⃣ ADIM 2: KAMERAYı AÇMAK

```python
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Can not open the webcam!")
    exit()
```

**Kod Açıklaması:**
- `VideoCapture(0)` = Bilgisayarın 1. kamerasını aç (0. kameradan başla)
- Eğer birden fazla kameran varsa: `VideoCapture(1)` 2. kamerayı açar
- `isOpened()` = Kamera başarıyla açıldı mı? kontrol et
- Açılmazsa: Hata mesajı yazdır ve programı kapat

---

## 4️⃣ ADIM 3: SONSUZ DÖNGÜ (Ana Oyun)

```python
while True:
    ret, frame = capture.read()
    
    if not ret:
        print("Can not take the frame!!")
        break
```

**Bu Bölüm:**
- `while True:` = Programı sürekli çalıştır
- `capture.read()` = Kameradan bir görüntü al (frame)
  - `ret` = Başarılı mı? (True/False)
  - `frame` = Alınan görüntü verisi
- `if not ret:` = Görüntü alınamazsa döngüyü kapat

**Frame Nedir?**
Video, saniyede 30+ tane resimden oluşur. Her birine `frame` diyoruz.

---

## 5️⃣ ADIM 4: MODELLE TAHMİN YAPMAK

```python
results = model(frame)
object_counter = 0
```

**Neler Oluyor?**
- Framemizi modele veriyoruz
- Model bize algıladığı nesnelerin listesini döndürüyor
- `results` = Bulunan tüm nesneler
- `object_counter = 0` = Sayaç başlatıyoruz

**Sonuç Nedir?**
Model bize şunları söyler:
- Nesneler nerede? (Koordinatlar)
- Nedir bu nesne? (Sınıf adı)
- Ne kadar emin? (Güven yüzdesi)

---

## 6️⃣ ADIM 5: NESNELER ÜZERİNDE İŞLEM YAPMAK

### 5.A - Her Nesneyi Dolaşmak

```python
for r in results:              # Her sonuç için
    for box in r.boxes:         # Her kutu (nesne) için
        object_counter += 1     # Sayacı artır
```

**Mantığı:**
```
results
├── Nesne 1 (box)
├── Nesne 2 (box)
└── Nesne 3 (box)
```

Her nesneyi tek tek işleyeceğiz.

### 5.B - Nesne Bilgilerini Çıkarmak

```python
x1,y1,x2,y2 = map(int,box.xyxy[0])
confidence = float(box.conf[0])
class_id = int(box.cls[0])
label = model.names[class_id]
```

**Her satırın anlamı:**

| Kod | Anlamı | Örnek |
|-----|--------|-------|
| `x1, y1, x2, y2` | Nesnenin 4 köşesinin koordinatları | (100, 150, 350, 500) |
| `confidence` | Modelin ne kadar emin olduğu | 0.95 (yani %95) |
| `class_id` | Nesne türünün numarası | 0 = kişi, 1 = araba... |
| `label` | Nesne türünün adı | "person", "car"... |

**Koordinatlar Nasıl Çalışır?**
```
(x1, y1) ──────── Üst sol köşe
│                 │
│    NESNE        │
│                 │
└─────── (x2, y2) Alt sağ köşe
```

### 5.C - Güvensiniz Filtresi

```python
if confidence < 0.5:
    continue
```

**Mantığı:**
- Eğer modelin %50'den az emin olduğu bir şey varsa
- Onu görmezden gel, bir sonrakine geç
- Bu sayede hatalı algılamaları filtreleriz

---

## 7️⃣ ADIM 6: EKRANA ÇİZİM YAPMAK

### 6.A - Nesnenin Etrafına Kutu Çizmek

```python
cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
```

**Parametreler:**
- `frame` = Hangi resme çizeceğiz
- `(x1, y1)` = Üst-sol köşe
- `(x2, y2)` = Alt-sağ köşe
- `(0, 255, 0)` = Renk (BGR formatında: Mavi, Yeşil, Kırmızı) → Yeşil
- `2` = Çizgi kalınlığı (pixel)

**Renk Sistemi (BGR):**
```
(255, 0, 0)   = Mavi
(0, 255, 0)   = Yeşil
(0, 0, 255)   = Kırmızı
(255, 255, 0) = Açık Mavi
(255, 0, 255) = Magenta
(0, 255, 255) = Sarı
```

### 6.B - Metni Yazmak

```python
cv2.putText(
    frame,
    f"{label}---{confidence:.2f}",
    (x1, y1-10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    1
)
```

**Her Parametrenin Anlamı:**
- `frame` = Nereye yazacağız
- `f"{label}---{confidence:.2f}"` = Yazacağımız metin
  - `{label}` = Nesnenin adı (person, car...)
  - `{confidence:.2f}` = Güven yüzdesi (2 ondalak basamak)
  - Örnek çıktı: "person---0.95"
- `(x1, y1-10)` = Metin nereye başlayacak (kutudan 10 pixel yukarı)
- `cv2.FONT_HERSHEY_SIMPLEX` = Font tipi
- `0.6` = Font boyutu
- `(255, 0, 0)` = Metin rengi (Mavi)
- `1` = Metin kalınlığı

### 6.C - Toplam Nesne Sayısını Yazmak

```python
cv2.putText(
    frame,
    f"Total Objects: {object_counter}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)
```

**Farklılıklar:**
- `(10, 30)` = Ekranın sol üst köşesinden 10, 30 pixel uzakta
- `1` = Daha büyük font boyutu
- `(255, 255, 255)` = Beyaz renk
- `2` = Daha kalın yazı

### 6.D - Ekran Arka Planında Yarı Saydam Kutu (İsteğe Bağlı)

```python
overlay = frame.copy()
cv2.rectangle(overlay, (0, 0), (350, 60), (0, 0, 0), -1)
alpha = 0.7
frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
```

**Amacı:**
Metin arkasında siyah bir yarı saydam dikdörtgen çizmek (okunabilirlik için)

**Adım Adım:**
1. `overlay = frame.copy()` - Frameden bir kopya al
2. `cv2.rectangle(overlay, (0, 0), (350, 60), (0, 0, 0), -1)` - Siyah kutu çiz
3. `cv2.addWeighted()` - İki resmi karıştır (transparans için)

---

## 8️⃣ ADIM 7: EKRANDA GÖSTERMEK

```python
cv2.imshow("Live Object Counter", frame)
```

**Yapısı:**
- `cv2.imshow()` = Pencere aç ve resmi göster
- `"Live Object Counter"` = Pencere başlığı
- `frame` = Gösterilecek resim

---

## 9️⃣ ADIM 8: ÇIKIŞ KOŞULU

```python
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

**Bu Kod Neyi Yapıyor?**
- `cv2.waitKey(1)` = 1 milisaniye bekle ve tuşa basıldı mı bak
- `& 0xFF` = Sadece ASCII kodu al (teknik ayrıntı)
- `ord("q")` = "q" tuşunun ASCII kodu
- `if ... break:` = Eğer "q" basıldıysa döngüyü kapat

**Pratik Açıklama:**
Kullanıcı "q" tuşuna basarsa program kapanır.

---

## 🔟 ADIM 9: TEMIZLEME

```python
capture.release()
cv2.destroyAllWindows()
```

**Yapması Gerekenler:**
- `capture.release()` = Kamerayı serbest bırak
- `cv2.destroyAllWindows()` = Açık tüm pencereleri kapat

**Neden Gerekli?**
- Kaynakları temiz şekilde serbest bırakması lazım
- Sonraki çalıştırmalarda problem olmasın diye

---

## 📊 TOPLAM AKIŞ DİYAGRAMI

```
┌─────────────────────┐
│ 1. Model Yükle      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 2. Kamera Aç        │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 3. Her Frame İçin:  │
│    (SONSUZ DÖNGÜ)   │
└──────────┬──────────┘
           ↓
    ┌──────────────┐
    │ Frame Al     │
    └──────┬───────┘
           ↓
    ┌──────────────────┐
    │ Model ile Tahmin │
    └──────┬───────────┘
           ↓
    ┌──────────────────────────┐
    │ Her Nesne İçin:          │
    │ - Bilgiler Al            │
    │ - Güven Filtresi         │
    │ - Kutu Çiz               │
    │ - Metin Yaz              │
    └──────┬───────────────────┘
           ↓
    ┌──────────────────────┐
    │ Ekranda Göster       │
    └──────┬───────────────┘
           ↓
    ┌──────────────────┐
    │ q'ye Bastı mı?   │◄─────── HAYIR
    └──┬───────────────┘         │
       │ EVET                    │
       ↓                         │
    ┌──────────────────┐         │
    │ Döngüyü Kapat    │         │
    └────┬─────────────┘         │
         ↓                       │
    ┌──────────────────┐         │
    │ Kaynakları Serbest│        │
    └──────────────────┘         │
                                 │
                    Geri Döngüye ┘
```

---

## 💾 TEMEL FONKSİYON ÖZETI

| Fonksiyon | Amacı | Parametreler |
|-----------|-------|--------------|
| `YOLO()` | Model yükle | Model adı (.pt dosyası) |
| `VideoCapture()` | Kamera aç | Kamera numarası (0, 1, 2...) |
| `read()` | Frame al | - |
| `rectangle()` | Kutu çiz | frame, köşe1, köşe2, renk, kalınlık |
| `putText()` | Metin yaz | frame, metin, konum, font, boyut, renk, kalınlık |
| `imshow()` | Pencerede göster | başlık, frame |
| `waitKey()` | Tuş bekle | milisaniye |
| `release()` | Kamerayı kapat | - |
| `destroyAllWindows()` | Pencereleri kapat | - |

---

## 🎯 ÖĞRENME HEDEFLERİ

Bu kodla öğrendiklerimiz:

- ✅ Yapay Zeka modelini nasıl yükleyeceğimiz
- ✅ Canlı video akışıyla nasıl çalışacağımız
- ✅ Görüntü işleme temel işlemlerini
- ✅ Nesneleri tanımlama ve sınıflandırma
- ✅ Gerçek zamanlı işleme mantığını
- ✅ OpenCV temel fonksiyonlarını
