# T.C. Kimlik Kartı Kontrol Servisleri – Deneysel Baseline

Bu FastAPI prototipinin ana belge akışı tek bir kimlik kartı görüntüsü alır:

1. Kartın ön veya arka yüz olduğunu belirler.
2. Tespit edilen yüze uygun özelliklerle gerçek kart/fotokopi risk sınıflandırması yapar.
3. Yalnızca `front + real_candidate` sonucunda tek kullanımlık hologram analizi oturumu oluşturur.

Mevcut model sürümü: `baseline-0.13.0`

Sistem kesin belge gerçekliği kararı vermez. Sonuçlar açıklanabilir görüntü sezgisellerine dayanır ve kalibre edilmiş olasılık değildir. Üretim kullanımı öncesinde daha geniş, bağımsız ve etiketli bir veri setiyle doğrulanmalıdır.

## Mimari

Proje, DDD katmanlarını gereksiz yere çoğaltmadan sade bir katman ayrımı kullanır:

```text
HTTP isteği
    ↓
api/                 Endpointler, upload kontrolleri ve Pydantic modelleri
    ↓
services/            Kullanım akışları ve belge/hologram karar kuralları
    ↓
infrastructure/      OpenCV, crop saklama ve geçici oturum altyapısı

core/                Ortak ayarlar ve hata sınıfları
```

Temel dosya yapısı:

```text
hologram_api/
├── main.py
├── api/
│   ├── document_check_endpoints.py       # Ana tek-görsel endpoint
│   ├── hologram_endpoints.py
│   ├── image_uploads.py
│   └── models/
│       ├── common.py
│       ├── document.py
│       └── hologram.py
├── services/
│   ├── document_classification.py        # Tek-görsel ana kullanım akışı
│   ├── authorized_hologram_analysis.py
│   ├── document/
│   │   ├── classification_policy.py      # Yüze göre gerçek/fotokopi eşikleri
│   │   ├── frame_analysis.py
│   │   ├── front_classifier.py
│   │   ├── front_color_analysis.py
│   │   ├── back_security_analysis.py
│   │   ├── side_detection.py             # Tekli ve çoklu ön/arka tespiti
│   │   ├── side_classifier.py
│   │   └── response_builder.py
│   └── hologram/
│       ├── feature_analysis.py
│       ├── frame_analysis.py
│       └── response_builder.py
├── infrastructure/
│   ├── session/hologram_authorization.py
│   ├── storage/
│   │   ├── crop_storage.py
│   │   └── jpeg_writer.py
│   └── vision/card/                      # Kart çözme, bulma ve düzeltme
├── core/
│   ├── config.py
│   └── exceptions.py
├── tests/
├── .env.example
├── .gitignore
└── requirements.txt
```

Endpoint katmanı yalnızca HTTP, upload ve cookie işlemlerini yönetir. Ana karar akışı `services/document_classification.py` içindedir. OpenCV tabanlı teknik kart işleme, dosya yazma ve bellek içi oturum yönetimi `infrastructure` altında tutulur.

## Endpointler

| Yöntem | Adres | Durum | Görevi |
|---|---|---|---|
| `POST` | `/v1/document/check` | Ana endpoint | Tek görüntüde önce ön/arka, sonra gerçek/fotokopi analizi yapar. |
| `POST` | `/v1/hologram/check` | Aktif | Yetkilendirilmiş gerçek ön yüz üzerinde statik hologram analizi yapar. |
| `GET` | `/health` | Aktif | Uygulama durumunu ve model sürümünü döndürür. |

FastAPI arayüzleri:

- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- OpenAPI şeması: `GET /openapi.json`

## Tek-görsel belge kontrolü

### İstek

`POST /v1/document/check`

Endpoint `application/json` gövdesinde Base64 kodlanmış bir ön veya arka yüz JPEG/PNG görüntüsü alır. Kullanıcı yüz veya gerçek/fotokopi etiketi göndermez.

Örnek JSON:

```json
{
  "frame_base64": "/9j/4AAQSkZJRgABAQ...",
  "media_type": "image/jpeg",
  "filename": "card.jpg"
}
```

`frame_base64`, yalnız Base64 içeriği veya `data:image/jpeg;base64,...` / `data:image/png;base64,...` biçiminde eşleşen bir data URL olabilir. PowerShell ile örnek istek:

```powershell
$base64 = [Convert]::ToBase64String(
    [IO.File]::ReadAllBytes("card.jpg")
)
$body = @{
    frame_base64 = $base64
    media_type = "image/jpeg"
    filename = "card.jpg"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/v1/document/check" `
    -ContentType "application/json" `
    -Body $body
```

### İşlem akışı

```text
Base64 JSON görüntüsünü al
→ Base64, media type ve decode edilmiş byte boyutunu kontrol et
→ görüntüyü bellekte byte dizisine çevir
→ piksel sınırını kontrol et
→ kartı tespit et ve 856x540 boyutuna normalize et
→ çekim kalitesini değerlendir
→ ön-yüz kanıt skorunu hesapla
→ front / back / retake_required kararı ver
→ tespit edilen yüze ait gerçek/fotokopi skorunu hesapla
→ real_candidate / photocopy_suspected / retake_required kararı ver
→ kesin sonuçta normalize kart crop'unu kaydet
→ yalnızca front + real_candidate için hologram oturumu oluştur
```

Kart çözme, bulma, perspektif düzeltme ve kalite analizi ortak hazırlık aşamasında bir kez yapılır. Arka yüz tespit edilirse yalnızca yüze özgü arka güvenlik analizi ayrıca çalıştırılır.

### Karar eşikleri

Ön/arka yüz tespiti:

- Ön-yüz skoru `0.80` ve üzerindeyse `front`
- Ön-yüz skoru `0.70` ve altındaysa `back`
- Aradaki değerlerde `retake_required`

Ön yüz gerçek/fotokopi sınıflandırması:

- Skor `0.68` ve altındaysa `photocopy_suspected`
- Skor `0.76` ve üzerindeyse `real_candidate`
- Aradaki değerlerde `retake_required`

Arka yüz gerçek/fotokopi sınıflandırması:

- Skor `0.65` ve altındaysa `photocopy_suspected`
- Skor `0.70` ve üzerindeyse `real_candidate`
- Aradaki değerlerde `retake_required`

Bu eşikler prototip değerleridir. Özellikle arka yüz eşikleri bağımsız veriyle yeniden kalibre edilmelidir.

### Response

Temel response alanları:

- `decision`: `real_candidate`, `photocopy_suspected` veya `retake_required`
- `detected_side`: `front`, `back` veya kararsız sonuçta `null`
- `front_side_score`: ön-yüz kanıt skoru
- `document_score`: tespit edilen yüze ait gerçek/fotokopi skoru
- `side_thresholds` ve `classification_thresholds`: kullanılan karar sınırları
- `reason_codes`: açıklanabilir karar nedenleri
- `capture_quality`, `features`, `normalized_components`: analiz ayrıntıları
- `saved_crop`: yalnızca kesin sınıflandırmada oluşturulan anonim crop bilgisi
- `hologram_ready`: hologram kontrolünün kullanılabilir olup olmadığı
- `document_session_id`: yalnızca yetkilendirilmiş ön-gerçek sonuçta dönen geçici kimlik

Sınıflandırma veya yüz tespiti kararsızsa endpoint HTTP `200` ile `decision: "retake_required"` döndürür. Bu durum teknik hata değil, yeni çekim gerektiren bir analiz sonucudur.

İstek hataları:

- Geçersiz Base64, data URL/media type uyuşmazlığı veya çözülemeyen görüntü: `422`
- Desteklenmeyen `media_type`: `422`
- Decode edilmiş boyut sınırını aşan görüntü: `413`
- Crop saklama hatası: `500`

## Crop çıktıları

Ham yüklemeler ayrı dosyalar olarak kaydedilmez. Yalnızca kesin sınıflandırılan, perspektifi düzeltilmiş kart crop'ları saklanır:

```text
cropped_images/
├── real/front/
├── real/back/
├── photocopy/front/
└── photocopy/back/
```

Her kayıt anonim bir `sample_id` alır. `retake_required` sonucunda crop kaydedilmez.

## Hologram analizi

`POST /v1/hologram/check`

Yalnızca `/v1/document/check` sonucunun `front + real_candidate` olması hologram analizi için yetki verir. Backend bu durumda:

- Gerçek ön yüz crop'unu kısa süreyle bellekte tutar.
- Rastgele bir `document_session_id` üretir.
- `/v1/hologram` yoluyla sınırlandırılmış `HttpOnly` cookie oluşturur.
- Response içinde `hologram_ready: true` döndürür.

Hologram endpointi dosya veya manuel `document_session_id` almaz; yetki cookie üzerinden okunur. Oturum tek kullanımlıdır, varsayılan olarak 15 dakika geçerlidir ve uygulama yeniden başlatıldığında silinir. Arka yüz, fotokopi veya yeniden çekim sonucunda hologram yetkisi verilmez ve varsa önceki cookie silinir.

Statik hologram analizi normalize kart üzerindeki sabit ROI'de desen detayı, kontrast, renk çeşitliliği ve parlama gibi sinyalleri değerlendirir. Tek kare, hologramın hareket boyunca renk değişimini doğrulayamaz.

## Kurulum ve çalıştırma

```powershell
cd hologram_api
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Ardından Swagger arayüzü `http://127.0.0.1:8000/docs` adresinden açılabilir.

### Ortam ayarları

`.env` yerel çalışma ayarlarını içerir ve Git tarafından yok sayılır. Uygulama `.env` olmadan güvenli varsayılanlarla çalışır. Özelleştirmek için:

```powershell
Copy-Item .env.example .env
```

Sistem veya deployment ortam değişkenleri `.env` değerlerinden önceliklidir.

| Değişken | Varsayılan | Görevi |
|---|---:|---|
| `LOG_LEVEL` | `INFO` | Uygulama log seviyesi. |
| `CROP_OUTPUT_DIR` | `cropped_images` | Proje altındaki güvenli crop klasörü. |
| `CROP_RETENTION_DAYS` | `30` | Crop saklama süresi. |
| `CROP_CLEANUP_INTERVAL_SECONDS` | `3600` | Süresi dolan crop'ları kontrol aralığı. |
| `DOCUMENT_SESSION_TTL_SECONDS` | `900` | Belge oturumunun geçerlilik süresi. |
| `MAX_DOCUMENT_SESSIONS` | `32` | Bellekte tutulabilecek belge oturumu sayısı. |
| `MAX_FRAME_SIZE_MB` | `10` | Tek görüntü için byte boyutu sınırı. |
| `MAX_FRAME_PIXELS` | `16000000` | Çözülen görüntü için piksel sınırı. |
| `HOLOGRAM_SESSION_COOKIE` | `hologram_session` | Oturum cookie adı. |
| `COOKIE_SECURE` | `false` | HTTPS cookie zorunluluğu; production'da `true` olmalıdır. |
| `COOKIE_SAMESITE` | `strict` | Cookie SameSite politikası. |

## Testler

Testler Python'un yerleşik `unittest` modülüyle çalışır; ek test paketi gerekmez:

```powershell
python -m unittest discover -s tests -v
```

Test paketi; görüntü girdi korumalarını, karar sınırlarını, servis dallarını, response modellerini, crop yönlendirmesini, cookie davranışını, hologram oturumunu ve OpenAPI sözleşmesini kapsar.

## Güvenlik ve sınırlamalar

- Decode edilmiş her görüntü en fazla 10 MB ve 16 milyon piksel olabilir. Base64 JSON aktarımı ham dosyadan yaklaşık `%33` daha büyüktür.
- Ham kimlik görüntüleri ayrı dosyalar olarak saklanmaz veya loglanmaz.
- Crop dosyaları kimlik görüntüsü içerdiği için hassas veri kabul edilmelidir.
- `cropped_images` klasörü Git'e veya herkese açık deployment paketine eklenmemelidir.
- Mevcut gerçek/fotokopi ayrımı eğitilmiş bir model değil, statik görüntü sezgiselleridir.
- Kamera, ışık, yansıma, baskı türü ve renk yönetimi sonucu etkileyebilir.
- Hologram analizi tek kare üzerinden statik risk sinyali üretir.
- Üretim kullanımı için etiketli veri seti, bağımsız değerlendirme, eşik kalibrasyonu ve hata oranı raporu gerekir.
