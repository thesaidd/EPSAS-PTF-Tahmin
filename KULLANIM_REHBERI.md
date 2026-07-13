# EPİAŞ PTF Tahmin MVP Kullanım Rehberi

Bu rehber, EPİAŞ PTF Forecasting MVP sistemini iş kullanıcıları ve demo izleyicileri için açık bir dille anlatır.

## Proje ne yapar?

Bu MVP, Türkiye elektrik piyasasında Gün Öncesi Piyasası Piyasa Takas Fiyatı (PTF/MCP) için saatlik tahmin üretir. Amaç, enerji tedarik, perakende, portföy yönetimi ve risk ekiplerinin bir sonraki gün için saatlik fiyat beklentisini ve belirsizlik seviyesini daha erken görmesini sağlamaktır.

Sistem şu çıktıları üretir:

- 24 saatlik gün öncesi PTF tahmini;
- alt/üst güven aralığı;
- saatlik risk seviyesi;
- model performans metrikleri;
- pipeline ve veri kalitesi durumu;
- dashboard ve API üzerinden izlenebilir sonuçlar.

## Kimler kullanabilir?

Bu MVP özellikle şu ekipler için tasarlanmıştır:

- enerji şirketi yöneticileri;
- portföy yöneticileri;
- elektrik perakende ve tedarik ekipleri;
- piyasa analiz ekipleri;
- operasyon ve veri ekipleri;
- teknik demo değerlendirme ekipleri.

## Sistem nasıl çalışır?

Akış basitçe şöyledir:

```text
EPİAŞ → Veri Tabanı → Özellikler → XGBoost → GPR Belirsizlik
      → Karar Katmanı → 24 Saatlik Tahmin → Dashboard
```

1. EPİAŞ Şeffaflık Platformu verisi alınır.
2. Saatlik PTF verisi PostgreSQL/TimescaleDB veritabanına kaydedilir.
3. PTF verisinden takvim, gecikme ve hareketli ortalama özellikleri üretilir.
4. XGBoost modeli ana PTF tahminini üretir.
5. GPR modeli tahmin belirsizliğini ve güven aralığını hesaplar.
6. Model karar katmanı, en güvenilir nokta tahmin çıktısını seçer.
7. Gün öncesi tahmin katmanı 24 saatlik PTF beklentisini üretir.
8. Monitoring katmanı veri, model ve pipeline sağlığını kontrol eder.
9. Dashboard sonuçları iş kullanıcılarına gösterir.

## Dashboard nasıl okunur?

Dashboard beş ana bölümden oluşur:

1. **Sistem Nasıl Çalışıyor?**  
   Veri ve model akışını özetler.

2. **İzleme ve Kalite Kontrol**  
   Veri tazeliği, veri kalitesi, pipeline sağlığı, model kalitesi, belirsizlik kalitesi ve risk dağılımını gösterir.

3. **Pipeline Durumu**  
   Günlük tahmin pipeline çalışmasının başarılı olup olmadığını gösterir.

4. **Gün Öncesi PTF Tahmini**  
   Hedef gün için 24 saatlik PTF beklentisini, güven aralığını ve risk seviyelerini gösterir.

5. **Model Karar Katmanı**  
   XGBoost ve GPR düzeltilmiş tahmin performanslarını karşılaştırır ve hangi nokta tahminin ürün çıktısı olarak seçildiğini açıklar.

## Metrikler ne anlama gelir?

- **Ortalama Mutlak Hata (MAE):** Tahminlerin gerçekleşen PTF değerinden ortalama sapmasıdır. Daha düşük olması daha iyidir.
- **Kök Ortalama Kare Hata (RMSE):** Büyük hataları daha fazla cezalandıran hata metriğidir. Daha düşük olması daha iyidir.
- **R² Skoru:** Modelin genel açıklama gücünü gösterir. 1'e yaklaştıkça uyum artar.
- **Güven Aralığı Kapsama Oranı:** Gerçek değerlerin tahmin bandı içinde kalma oranıdır.
- **Ortalama Bant Genişliği:** Belirsizlik bandının ortalama genişliğidir. Geniş bant daha yüksek piyasa belirsizliği anlamına gelir.
- **Risk Seviyesi:** LOW düşük, MEDIUM orta, HIGH yüksek belirsizlik anlamına gelir.

## Gün öncesi tahmin nasıl üretilir?

API üzerinden:

```powershell
$body = @{
  horizon_hours = 24
  model_version = "day_ahead_v1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/forecasts/ptf/day-ahead/generate `
  -ContentType "application/json" `
  -Body $body
```

CLI üzerinden:

```powershell
docker compose exec api python scripts/generate_day_ahead_ptf.py --horizon-hours 24 --model-version day_ahead_v1
```

Son tahmini görmek için:

```powershell
Invoke-RestMethod http://localhost:8000/api/forecasts/ptf/day-ahead/latest
```

## Pipeline nasıl çalıştırılır?

Demo için güvenli çalışma modu:

```powershell
$body = @{
  skip_ingestion = $true
  skip_feature_build = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/pipelines/daily-forecast/run `
  -ContentType "application/json" `
  -Body $body
```

Bu mod canlı EPİAŞ veri çekimini ve özellik yeniden üretimini atlar. Mevcut yerel verilerle tahmin üretir.

Durum kontrolü:

```powershell
Invoke-RestMethod http://localhost:8000/api/pipelines/daily-forecast/status
```

## Monitoring sonucu nasıl yorumlanır?

Monitoring üç ana statü üretir:

- **Sağlıklı (HEALTHY):** Sistem demo/operasyon için hazır görünüyor.
- **Uyarı (WARNING):** Kritik olmayan ama incelenmesi gereken sinyal var.
- **Kritik (CRITICAL):** Demo veya operasyon öncesi müdahale gerektiren sorun var.

Monitoring özellikle şunları kontrol eder:

- veri güncelliği;
- eksik saat sayısı;
- duplicate timestamp varlığı;
- son pipeline başarısı;
- son tahmin satır sayısı;
- model kalitesi;
- güven aralığı kapsama oranı;
- yüksek riskli saat sayısı.

Snapshot üretmek için:

```powershell
docker compose exec api python scripts/build_monitoring_snapshot.py --max-ptf-age-hours 168 --expected-forecast-horizon-hours 24
```

## Demo nasıl yapılır?

Servisleri başlatın:

```powershell
docker compose up -d --build
```

Demo yüzeyleri:

- Dashboard: <http://localhost:8501>
- Swagger API: <http://localhost:8000/docs>
- MLflow: <http://localhost:5000>

Tek komutluk demo yardımcısı:

```powershell
docker compose exec api python scripts/demo_local_mvp.py
```

Tam güvenli demo akışı:

```powershell
docker compose exec api python scripts/demo_local_mvp.py --all
```

## Sınırlamalar

- Bu sistem bir MVP'dir; üretim trading sistemi değildir.
- Canlı EPİAŞ ingestion için geçerli kullanıcı bilgileri gerekir.
- Model retraining manuel olarak yapılır.
- Monitoring snapshot bazlıdır; Prometheus/Grafana seviyesinde üretim izleme değildir.
- Dashboard'da authentication ve rol bazlı yetki bulunmaz.
- Talep, hava durumu, yenilenebilir üretim, arıza, yakıt fiyatı ve FX gibi ek dış değişkenler henüz modele dahil edilmemiştir.

## Sonraki geliştirme adımları

- Üretim seviye scheduler ve alarm sistemi eklemek.
- Model registry ve promotion workflow kurmak.
- Backtesting raporlarını zenginleştirmek.
- Talep, hava durumu, yenilenebilir üretim, outage ve FX verilerini eklemek.
- Kullanıcı yönetimi, authentication ve RBAC eklemek.
- Cloud deployment, CI/CD, yedekleme ve secrets yönetimi kurmak.
