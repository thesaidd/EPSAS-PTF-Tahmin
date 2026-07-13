# 10 Dakikalık Türkçe Demo Konuşma Metni

## 0:00-1:00 — Açılış

“Bu demo, Türkiye elektrik piyasası için geliştirdiğimiz EPİAŞ PTF Tahmin MVP'sini gösteriyor. Sistem, saatlik PTF verisini kullanarak bir sonraki gün için 24 saatlik fiyat beklentisi, güven aralığı ve risk seviyesi üretiyor.”

Gösterilecek adresler:

- Dashboard: <http://localhost:8501>
- Swagger API: <http://localhost:8000/docs>
- MLflow: <http://localhost:5000>

## 1:00-2:00 — Problem tanımı

“Enerji tedarik ve portföy yönetimi ekipleri için PTF volatilitesi operasyonel ve finansal risk yaratır. Tek bir fiyat tahmini çoğu zaman yeterli değildir; belirsizlik bandını ve yüksek riskli saatleri de görmek gerekir.”

Vurgulanacak değer:

- saatlik fiyat beklentisi;
- belirsizlik bandı;
- riskli saatlerin erken görünmesi;
- veri/model/pipeline sağlığının takip edilmesi.

## 2:00-3:00 — Sistem mimarisi

“Sistem EPİAŞ verisini alır, saatlik PTF tablosuna kaydeder, özellikler üretir, XGBoost ile nokta tahmini yapar, GPR ile belirsizlik bandı hesaplar ve karar katmanı ile kullanıcıya gösterilecek en güvenilir tahmini seçer.”

Akış:

```text
EPİAŞ → Veri Tabanı → Özellikler → XGBoost → GPR Belirsizlik
      → Karar Katmanı → 24 Saatlik Tahmin → Dashboard
```

## 3:00-4:00 — Dashboard tanıtımı

“Dashboard iş kullanıcıları için tasarlandı. Üst bölüm sistemin nasıl çalıştığını anlatıyor. Sonraki bölümlerde monitoring, pipeline durumu, gün öncesi tahmin ve model karar katmanı yer alıyor.”

Göster:

- Sistem Nasıl Çalışıyor?
- İzleme ve Kalite Kontrol
- Pipeline Durumu
- Gün Öncesi PTF Tahmini
- Model Karar Katmanı

## 4:00-5:00 — 24 saatlik PTF tahmini

“Bu bölüm hedef gün için 24 saatlik PTF beklentisini gösterir. Ortalama, minimum ve maksimum tahminlerle birlikte yüksek riskli saat sayısı da görünür.”

Anlat:

- Hedef tarih
- Ortalama PTF tahmini
- Minimum tahmin
- Maksimum tahmin
- 24 saatlik tahmin tablosu
- Belirsizlik bandı grafiği

## 5:00-6:00 — Risk ve güven aralığı

“Alt ve üst bantlar, tahminin beklenen belirsizlik aralığını gösterir. Bant genişledikçe piyasa belirsizliği artar. Risk seviyesi LOW, MEDIUM ve HIGH olarak sınıflandırılır.”

Terimler:

- LOW: düşük belirsizlik
- MEDIUM: orta belirsizlik
- HIGH: yüksek belirsizlik
- Güven aralığı: beklenen tahmin bandı
- Bant genişliği: belirsizlik düzeyi

## 6:00-7:00 — Model karar katmanı

“Karar katmanı, nokta tahmin için hangi modelin kullanılacağını seçer. Bu MVP'de GPR düzeltilmiş tahmin aynı test döneminde XGBoost'tan daha iyi sonuç vermediği için nokta tahmin modeli olarak XGBoost seçilmiştir. GPR ise güven aralığı ve risk seviyesi için kullanılmaya devam eder.”

Vurgula:

- XGBoost ana tahmini üretir.
- GPR belirsizlik bilgisini üretir.
- Karar katmanı iş kullanıcılarına daha güvenli ürün çıktısı verir.

## 7:00-8:00 — Pipeline ve monitoring

“Pipeline durumu günlük tahmin üretim akışının başarıyla tamamlanıp tamamlanmadığını gösterir. Monitoring bölümü ise veri güncelliği, eksik saatler, pipeline başarısı, model kalitesi, güven aralığı kapsaması ve risk dağılımını kontrol eder.”

Statüler:

- Sağlıklı: sistem hazır
- Uyarı: incelenmesi gereken durum var
- Kritik: operasyon öncesi müdahale gerekir

## 8:00-9:00 — Teknik altyapı

“Backend FastAPI ile, veri katmanı PostgreSQL/TimescaleDB ile, dashboard Streamlit ile, model takip sistemi MLflow ile çalışıyor. Tüm servisler Docker Compose ile tek komutla ayağa kalkıyor.”

Göster:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/api/system/readiness
```

## 9:00-9:30 — Mevcut sınırlamalar

“Bu bir MVP'dir. Üretim trading sistemi değildir. Canlı EPİAŞ ingestion için credential gerekir. Otomatik retraining, authentication, RBAC, production monitoring ve cloud secrets yönetimi sonraki faz konularıdır.”

## 9:30-10:00 — Sonraki adımlar ve kapanış

“Sonraki adımlarda talep, hava durumu, yenilenebilir üretim, outage, FX ve yakıt fiyatı gibi ek değişkenleri modele dahil etmek; model registry/promotion akışı kurmak; production scheduler, alerting ve kullanıcı yönetimi eklemek hedeflenir.”

Kapanış cümlesi:

“Bu MVP, enerji şirketleri için saatlik PTF beklentisini, belirsizliği ve operasyonel kaliteyi tek bir dashboard ve API üzerinden görünür hale getiren uçtan uca bir temel sunuyor.”
