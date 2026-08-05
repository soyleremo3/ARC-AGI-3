# Birleşik Entegrasyon Yol Haritası — ARC-AGI-3 Kaggle Ajanımız

**Agent 5 sentez raporu.** Bu belge, Agent 1–4'ün dört raporunu (`01-analiz.md`,
`02-teknik-degerlendirme.md`, `03-entegrasyon-analizi.md`,
`04-gelistirme-arastirmasi.md`) ve `agent/my_agent.py` (2218 satır),
`scripts/build_notebook.py`, `README.md`, `Makefile`, `scripts/play_local.py`
dosyalarının doğrudan okunmasını temel alan **tek, önceliklendirilmiş, somut**
bir uygulama planıdır. Amaç dört raporu yan yana koymak değil, aralarındaki
gerilimleri çözüp **ne yapacağımıza karar vermek**.

Önce iki şeyi net söylemek gerekiyor:

1. İncelenen 5 projenin **4'ü** (astroseger'in en güçlü varyantı, Tycho,
   symbolica arcgentica, RGB-Agent/PRO-LONG) temelde barındırılan/internet
   gerektiren modellere (Codex CLI, Claude Code CLI, Anthropic/OpenAI API,
   Agentica SDK) veya Docker sandbox'a dayanıyor. `scripts/build_notebook.py`
   satır 602'de `"isInternetEnabled": False` sabit — puanlanan koşuda bu
   dördünün "tam sürümü" **taşınamaz**, spekülasyon değil, kod kanıtı. Tek
   mimari olarak bize yakın olan (duck-harness) bile gerçek mekanizmasını
   (vLLM native tool-calling, `qwen3_coder` parser) gemma-4-31b-it'in
   destekleyip desteklemediği doğrulanmamış bir varsayıma dayandırıyor.
2. Bu yüzden bu yol haritası **"beş projeden birini taşıma" değil, "beş
   projenin ve ötesindeki araştırmanın süzülmüş özünü, mevcut tek-dosya /
   tek-GPU / internetsiz mimarimizin üstüne, küçük ve geri alınabilir
   adımlarla eklemek"** ilkesiyle kurulmuştur. Her adım `make play-local` /
   `make verify-local` ile gerçek oyun motoruna karşı doğrulanabilir olmalı,
   ve her adımın "işe yaramazsa" davranışı **mevcut hâlden daha kötü
   olmamalı**.

---

## 1. Felsefe: RHAE metriği bu yol haritasının omurgası

`docs.arcprize.org/methodology`'den (Agent 4, Bölüm 1.1, birincil kaynak
doğrulaması):

- `level_score = (human_baseline_actions / agent_actions)²`, üst sınır 1.15×.
- **İç işlemler (tool call, reasoning adımı, retry) aksiyon sayılmıyor —
  sadece ortamı değiştiren gerçek etkileşimler sayılıyor.**
- 14 Nisan 2026'dan itibaren insan temeli "medyan ilk-kez-oynayan insan"; geç
  seviyeler daha ağır puanlanıyor; bir oyunun %100 alması için **tüm**
  seviyelerin tamamlanması gerekiyor.

Sonuç: metrik kareyle cezalandırıyor, ama sadece **gerçek** aksiyonları.
Model içi düşünme, prompt içindeki "önce anla sonra hareket et" disiplini,
reflection, hipotez etiketleme — bunların hiçbiri resmi skoru düşürmüyor.
Bu, tüm yol haritasının felsefesini belirliyor: **gerçek aksiyon harcamadan
önce ucuza "anlamaya" yatırım yapmak metriğin kendisi tarafından ödüllendiriliyor.**

**Ama kritik bir uyarı (Agent 4, Bölüm 4.5 ve Agent 3'ün bulgusuyla
birleştirilmiş, bu raporun kendi sentezi):** "reasoning ücretsiz" resmi
skor için doğru, ama bizim gerçek kısıtımız skor formülü değil, **Kaggle'ın
9 saatlik duvar-saati + tek GPU + tek vLLM süreci** bütçesi. `my_agent.py`
zaten bunun farkında: `GLOBAL_TIME_LIMIT_SECONDS = 9*3600` (satır 61),
`GAME_TIME_LIMIT_S = 8*3600` (satır 58), ve üretim profilinde
(`build_notebook.py` satır 50-76) `LLM_ACTION_THINKING=1` +
`LLM_MAX_NEW_TOKENS=3072` + `LLM_ACTION_CANDIDATES=1` +
`LLM_CANDIDATE_ARBITER=0` — yani ekip zaten commit `e0ae610` ile **"çoklu
aday yerine tek derin düşünen çağrı"** kararını vermiş. Bu doğru bir karar:
çoklu-aday mimarisi (`_generate_action_response`, satır 800;
`_select_candidate_with_arbiter`, satır 996) hâlâ kodda duruyor ama üretimde
kapalı, çünkü her ek aday tam bir ek `chat.completions.create` çağrısı demek
— RHAE'de "bedava" olsa da GPU-saatinde bedava değil. **Bu yol haritası bu
kararı bozmuyor; "ücretsiz reasoning" ilkesini tek çağrı içinde derinlik
(thinking, yapılandırılmış hipotez takibi, zengin durum tanımı) olarak
uyguluyor, paralel çağrı sayısını artırarak değil.**

Bu felsefe üç somut ilkeye dönüşüyor:
1. **Önce ucuza anla, sonra pahalıya dene** — prompt/hafıza zenginleştirmesi
   önce, kod-çalıştırma/çoklu-model gibi pahalı mekanizmalar sonra.
2. **"Bedava" olan şey reasoning token'ı, GPU-saati değil** — her yeni
   mekanizma toplam duvar-saati bütçesine karşı ölçülmeli
   (`GLOBAL_TIME_LIMIT_SECONDS`, `game_time_remaining_s`).
3. **Hiçbir yeni mekanizma mevcut güvenlik ağını (fallback zinciri,
   deadline kontrolü, `except Exception` sarmalayıcılar) atlayamaz** —
   kullanıcının açık talebi: sıfır bug, yüksek güvenilirlik.

---

## 2. Referans noktası: repo'nun gerçek mimarisi (doğrulanmış)

`agent/my_agent.py` tek dosya, notebook'a `scripts/build_notebook.py` satır
578'de `%%writefile /tmp/my_agent.py` ile ham metin olarak gömülüyor — yerel
`vendor/`, `scripts/` klasörleri hiç notebook'a girmiyor. Mevcut alt
sistemler (hafife alınmamalı, çoğu önerilen "kısmi versiyon" bunların
üstüne inşa ediliyor):

- **Reflection memory** — `_build_reflection_prompt` (satır 1412),
  `_run_reflection` (satır 1472), oyun başına diskte saklanıyor
  (`_reflection_memory_path`, satır 153, `game_id`'ye göre anahtarlanmış —
  **oyunlar arasında paylaşılmıyor**, bu Faz 2a'nın çıkış noktası).
  `## Rules / ## Goal / ## Progress / ## Avoid` başlıkları zaten var.
- **Durum-anahtarlı "etkisiz aksiyon" hafızası** — `failed_state_actions`,
  `failed_abstract_actions`, `_state_abstraction` (satır 2021, bağlı-bileşen
  tabanlı, animasyon/sayaç gürültüsüne dayanıklı bir imza — duck-harness'ın
  segmentation'ına kavramsal olarak yakın, zaten var).
  `_ineffective_actions_for_current_state` (satır 1397).
- **Plan kuyruğu** — `pending_actions`, `MAX_PLAN_ACTIONS=4`,
  `_dequeue_action` (satır 1293); `_observe_frame` (satır 1639) durum
  tekrarında/aksiyon etkisizse kuyruğu otomatik temizliyor.
- **Çoklu-aday + arbiter mimarisi** — kodda tam, üretimde
  `LLM_ACTION_CANDIDATES=1` ile kapalı (yukarıda açıklandı).
- **Zaman bütçesi** — sınıf-seviyeli paylaşılan `_SUBMISSION_STARTED_AT`
  (satır 34, tüm `Swarm` thread'leri arasında paylaşılıyor — oyunlar
  **eşzamanlı** çalışıyor, yorum satır 31-33'te açık), `_global_deadline`
  (satır 175), `is_done` (satır 494) her adımda global/oyun bazlı deadline'ı
  kontrol ediyor.
- **vLLM yaşam döngüsü** — `_load_vllm_once` (satır 202), başarısızlıkta
  `_degrade_startup_settings` ile belleği düşürerek yeniden deneme
  (`_ensure_vllm_available`, satır 300), `VLLM_STARTUP_ATTEMPTS` (varsayılan
  3) tükenince `_vllm_startup_error` kilitleniyor ve ajan kalıcı olarak
  `_fallback_action`'a düşüyor — **çöküş yok, sonsuz döngü yok.**
- **JSON çıkarma + onarım** — `_extract_action_json` (satır 1197, serbest
  metinden JSON tarama), başarısızsa `_build_json_repair_prompt` (satır
  1179) ile tek bir onarım turu.
- **İzleme** — `_write_llm_trace` (satır 1540) her adımı JSONL olarak
  `LLM_TRACE_PATH`'e yazıyor; `_timing_enabled`/`_log_timing` performans
  telemetrisi veriyor.

Üretim profili (`build_notebook.py` satır 50-76,
`PROFILE_ENV`): `AGENT_MAX_ACTIONS=1000`, `LLM_ACTION_CONTEXT_FRAMES=4`,
`LLM_ACTION_THINKING=1`, `LLM_CANDIDATE_ARBITER=0`,
`LLM_CONFIDENCE_PROMPT=0`, **`LLM_INCLUDE_FRAME_DESCRIPTOR=0`** (kodda var,
üretimde kapalı — Faz 1a'nın konusu), `LLM_MAX_NEW_TOKENS=3072`,
`LLM_REFLECTION_INTERVAL=10`, `LLM_REFLECTION_MAX_NEW_TOKENS=10000`,
`VLLM_LIMIT_MM_PER_PROMPT={"image":4}`, `VLLM_MAX_MODEL_LEN=32768`.

Kod-çalıştırma sandbox'ı **yok** — model hiçbir zaman Python çalıştırmıyor,
sadece JSON aksiyon planı üretiyor. `_generate_responses` (satır 1075)
içindeki yorum (satır 1122-1130) bilinçli bir mimari karar belgeliyor:
guided-JSON decoding + reasoning aynı anda açılamıyor (vLLM sunucusu
`--reasoning-parser` olmadan başlatılıyor), bu yüzden `thinking` açıkken
`response_format` bırakılıyor ve serbest metinden JSON taranıyor — bu, son
commit (`2258e00`, "Fix two bugs found in Phase 1/2 review:
guided-JSON+thinking conflict") ile düzeltilmiş, aktif korunan bir tasarım
kararı; aşağıdaki hiçbir öneri bunu bozmamalı.

---

## 3. Raporlar arası çelişkiler ve kararlarım

**Çelişki 1 — Agent 3'ün "en umut verici" listesi vs. Agent 4'ün "en yüksek
etki" listesi.** Agent 3, kod-çalıştırmasız/tek-dosya-uyumlu küçük adımları
(duck-harness prompt zenginleştirmesi, Tycho hipotez etiketleme) en üstte
sıralıyor. Agent 4'ün #1 fikri ("aramayı her zaman model içinde yap, gerçek
ortama sadece maksimum-bilgi-kazandıran aksiyonları gönder") daha iddialı,
offline çoklu-hipotez üretimi ima ediyor. **Kararım:** Bunlar çelişmiyor,
farklı ölçekte aynı fikir. Agent 4'ün fikrini **literal olarak** ("N rakip
dünya-modeli programı üret, offline oyla") uygulamak yeni bir çoklu-çağrı
mimarisi ister — tam da ekibin `LLM_CANDIDATE_ARBITER=0` ile bilinçli
kapattığı şey. Onun yerine bu felsefeyi **mevcut tek-çağrı mimarisinin
içine** taşıyorum: zengin durum tanımı (Faz 1a) + yapılandırılmış
hipotez takibi (Faz 1b) + oyunlar-arası mekanizma hafızası (Faz 2a), hepsi
"gerçek aksiyondan önce ucuza anla" ilkesinin aynı çağrı bütçesi içinde
kalan somutlaşmaları. Agent 4'ün literal öngörüsü (offline N-aday üretimi)
Faz 3+ stretch olarak not düşülüyor ama önceliklendirilmiyor.

**Çelişki 2 — Agent 1/2'nin Tycho bulgusu: `trigger` modu (83.07) `orchestrator`
modundan (88.49) daha kötü.** Yani "ne zaman modeli/hipotezi yeniden
kurmalıyım" kararını sert bir kurala (falsifikasyon algılandığında otomatik
tetikle) bağlamak, modele bırakmaktan daha kötü sonuç veriyor. **Kararım:**
Faz 1b'deki hipotez-etiketleme mekanizmasını **hard-gate değil, prompt
düzeyinde bir nudge** olarak tasarlıyorum — `[CONFIRMED]/[HYPOTHESIS]/
[FALSIFIED]` etiketlerini zorunlu kılan bir format kontrolü yok, sadece
istek var; format bozuksa mevcut `_clean_reflection_markdown` (satır 1445)
zaten var olan yumuşak fallback'iyle devam ediyor. Bu, Tycho'nun kendi
bulgusuyla tutarlı ve ek karmaşıklık/risk eklemiyor.

**Çelişki 3 — Agent 3, duck-harness'ın "ASCII/segmentation" fikrini düşük
riskli/yüksek öncelik sayıyor; ama üretim profili
`LLM_INCLUDE_FRAME_DESCRIPTOR=0` ile bunu zaten kapatmış.** Bu bir çelişki
değil ama önemli bir gözlem: kod zaten var (`_frame_descriptor`, satır 1923)
ve bilerek/muhtemelen token-bütçesi için kapatılmış. **Kararım:** Bunu
Faz 1a'da yeniden açmayı **doğrudan üretime almadan önce**
`make verify-local` ile ölçülü biçimde test etmeyi öneriyorum — açık bir
"bunu daha önce biri kapattı, neden kapattığını ölçmeden geri açma" uyarısı
ile.

**Çelişki 4 — Agent 3 "Docker'sız `exec()` sandbox'ı kurmak orta-büyük
efor" derken, Agent 4 kod-çalıştırmayı hiç önceliklendirmiyor (RHAE-uyumlu
"model içi arama" fikri kod çalıştırmadan da yapılabilir).** **Kararım:**
Agent 4 haklı — kod-çalıştırma olmadan da (yapılandırılmış hipotez metni +
BFS'i LLM'in kendi akıl yürütmesine bırakarak) RHAE'nin "ücretsiz reasoning"
ilkesinden faydalanılabilir. Bu yüzden kod-çalıştırma sandbox'ını (Faz 3a)
en sona, açıkça opsiyonel/deneysel olarak koyuyorum — üç rapor da bunun
riskli olduğunda hemfikir (gemma-4-31b-it'in kod kalitesi belirsiz, Windows/
Kaggle'da thread-timeout gerekiyor, Docker yok).

---

## 4. Ortak doğrulama protokolü (her faz için geçerli)

Her faz aşağıdaki adımlardan geçmeden **hiçbir zaman** `make submit`
çağrılmamalı:

1. **`make verify-local`** (ls20 + vc33, 50 adım, ~saniyeler) — değişikliğin
   `MyAgent`'ı hiç bozmadığını, `is_done`/`choose_action`'ın istisna
   fırlatmadığını doğrulayan ilk kapı. Bu yerel makinede gerçek vLLM/31B
   model olmayabileceğinden (bkz. not aşağıda), asıl kontrol ettiği şey
   **kontrol akışı ve fallback zinciri**, model kalitesi değil.
2. **`make play-local GAME=<ilgili oyun>`** — değişiklik belirli bir mekanik
   sınıfını hedefliyorsa (örn. click-tabanlı bir oyun, çok-seviyeli bir
   oyun), o oyunu tek başına, daha fazla adımla (`STEPS=500` gibi) çalıştır.
3. **`make play-local`** (tüm oyunlar, argümansız) — commit öncesi tam
   regresyon: `per_game` özeti ve `Aggregate scorecard score` çıktısını
   değişiklik öncesi/sonrası karşılaştır. **Skor düşüyorsa veya herhangi bir
   oyun daha az seviye tamamlıyorsa, değişiklik reddedilir veya flag arkasına
   alınır.**
4. **Ortam notu:** Yerel geliştirme makinesinde muhtemelen
   `/kaggle/input/models/...` yolu ve RTX 6000 yok — `VLLM_MODEL_PATH`'i
   yerel/küçük bir modele işaret ederek (veya `VLLM_START_SERVER=0` ile
   zaten çalışan harici bir sunucuya) test etmek gerekir. Bu belgenin
   kapsamı dışında bir kurulum detayı, ama her faz için **"local model ile
   verify-local yeşil" ile "31B model ile gerçek kalite" birbirinden ayrı
   iki doğrulama katmanı** olarak ele alınmalı — biri mekanik doğruluk,
   diğeri model kalitesi.
5. **Flag-arkası dağıtım:** Her yeni mekanizma bir ortam değişkeniyle
   açılıp/kapanabilmeli (mevcut desenle tutarlı: `LLM_*`/`VLLM_*` env
   flag'leri). Yeni flag varsayılan olarak **eski davranışı korumalı**;
   `PROFILE_ENV`'e eklenmesi (üretime alınması) ayrı, bilinçli bir adım
   olmalı — kod birleşmesi ile üretime alınma **aynı commit olmamalı.**
6. **Zaman bütçesi regresyon kontrolü:** `LLM_TIMING=1` (varsayılan açık)
   ile `_log_timing` (satır 1606) çıktısındaki `total_choose_action`
   değerlerini değişiklik öncesi/sonrası karşılaştır. Yeni bir mekanizma
   adım başına süreyi anlamlı artırıyorsa, bunun `game_time_remaining_s`
   bütçesine göre haklı olup olmadığı açıkça hesaplanmalı (örn. "adım başı
   +0.3s × 1000 aksiyon = +5 dakika, 8 saatlik bütçenin %0.1'i — kabul
   edilebilir" gibi).

---

## 5. Yol Haritası

### Faz 0 — Ölçüm çıtası (efor: ~1 saat, risk: yok)

Herhangi bir kod değişikliğinden önce: `LLM_INCLUDE_FRAME_DESCRIPTOR=0`
(mevcut üretim) ile `make play-local` tam koşusunu bir kez çalıştırıp
`per_game` tablosunu ve toplam scorecard skorunu kaydet. Bu, Faz 1-3'teki
her değişikliğin karşılaştırılacağı **temel çizgi**. Kod değişikliği yok,
sadece bir referans ölçüm — ama olmadan "iyileşti mi" sorusuna cevap
verilemez.

---

### Faz 1 — Prompt/veri-temsili zenginleştirme (kod-çalıştırma yok, sandbox yok, ikinci model yok)

**Genel gerekçe:** Agent 3'ün hem en yüksek hem en düşük riskli bulduğu
kategori; Agent 4'ün "önce ucuza anla" ilkesinin en doğrudan uygulaması.
Hiçbiri mevcut `choose_action`/`is_done` akışını yeniden yapılandırmıyor,
sadece prompt içeriğini ve reflection formatını değiştiriyor.

#### Faz 1a — Segmentation-benzeri çerçeve tanımını yeniden değerlendir ve zenginleştir

**Kaynak:** duck-harness'ın ham grid'i gizleyip `.segmentation` (bağlı-bileşen,
komşuluk) sunması (Agent 1 §5, Agent 2 §5.5, Agent 3 §5 madde 1). Bizde
kavramsal karşılığı zaten var: `_frame_descriptor` (satır 1923, renk sayımı +
bbox + örnek koordinat) ve `_state_abstraction`'ın flood-fill mantığı
(satır 2021-2064) + `_component_centers` (satır 1852-1882, tıklama hedefi
seçimi için zaten kullanılan bağlı-bileşen algoritması).

**Ne yapılacak:**
1. `_frame_descriptor`'ı, `_state_abstraction`'daki flood-fill'i (kod
   tekrarını önlemek için ortak bir `_connected_components(grid)` yardımcı
   fonksiyonuna çıkararak — bu aynı zamanda küçük bir sadeleştirme) her
   rengin sadece bbox'ını değil, **bağlı bileşen sayısını, her bileşenin
   alanını ve bileşenler arası kabaca komşuluğunu** (örn. bbox'ları N piksel
   içinde kesişen bileşen çiftleri) içerecek şekilde genişlet.
2. `LLM_INCLUDE_FRAME_DESCRIPTOR` varsayılanı zaten `"1"` (satır 752); asıl
   iş, üretim profilinde neden `"0"` olduğunu **ölçerek** anlamak: Faz 0
   temel çizgisine karşı, `LLM_INCLUDE_FRAME_DESCRIPTOR=1` ile
   `make play-local` tam koşusu + `_log_timing` süre karşılaştırması yap.
   Skor artıyor ve süre bütçesi içindeyse, `build_notebook.py`'deki
   `PROFILE_ENV`'de `"LLM_INCLUDE_FRAME_DESCRIPTOR": "1"` yap.

**Efor:** 3-4 saat (yardımcı fonksiyon çıkarma + genişletme + ölçüm).
**Risk:** Düşük — pür veri/prompt değişikliği, mevcut `_include_frame_descriptor()`
flag deseniyle zaten geri alınabilir.
**Fallback:** Flag `"0"`'a çekilirse davranış tam olarak bugünkü üretim
davranışına döner; kod yolu zaten var ve test edilmiş.
**Doğrulama:** Bölüm 4'teki protokol + özellikle adım 6 (zaman bütçesi),
çünkü descriptor JSON'u prompt uzunluğunu artırıyor (token maliyeti).

#### Faz 1b — Yanlışlanabilir hipotez etiketleme (Tycho-lite + schema'nın "reality outranks the model" ilkesi)

**Kaynak:** Tycho'nun öner→doğrula→onayla/reddet döngüsü (Agent 1 §2,
Agent 2 §2.2), astroseger'in `twma` (metinsel dünya modeli, kod
çalıştırmadan) varyantı (Agent 1 §1, Agent 3 §1), ve "[schema]"nın "Reality
outranks the model" ilkesi (Agent 4 §5.1, mimari fikir olarak — sayısal
iddiadan bağımsız, bkz. Bölüm 8).

**Ne yapılacak:** `_build_reflection_prompt` (satır 1412) talimatını
genişlet: `## Rules` altındaki her madde `[CONFIRMED]` (birden fazla
transitionla doğrulanmış), `[HYPOTHESIS]` (tek gözlemden veya çıkarımdan)
ya da `[FALSIFIED]` (son transitionlarla çelişen, artık geçersiz)
etiketiyle işaretlenmeli. Prompt'a şu talimatı ekle: *"Before keeping a
[HYPOTHESIS], check whether the last transitions confirm or contradict it.
If contradicted, mark it [FALSIFIED] and explain what changed."* Bu tamamen
metin-düzeyinde bir talimat; `_clean_reflection_markdown` (satır 1445)
hiçbir yeni zorunlu doğrulama eklemeden aynı şekilde çalışmaya devam eder
(format bozuksa mevcut fallback zaten devrede). İsteğe bağlı, düşük riskli
bir ek: `_run_reflection` içinde (satır 1472) reflection metninde en az bir
`[FALSIFIED]` veya `[CONFIRMED]` etiketi **hiç** geçmiyorsa sadece bir
`logger.info` uyarısı bas (davranışı değiştirme, sadece gözlemlenebilirlik).

**Neden hard-gate değil:** Bölüm 3, Çelişki 2'de açıklandığı gibi Tycho'nun
kendi verisi (`trigger` 83.07 < `orchestrator` 88.49) modele karar verme
özgürlüğü bırakmanın sert kurallardan daha iyi sonuç verdiğini gösteriyor.

**Efor:** 1-2 saat (sadece prompt metni + bir log satırı).
**Risk:** Neredeyse sıfır — mevcut mekanizmanın üstüne, geriye dönük uyumlu;
yeni bir başarısızlık modu yok çünkü format doğrulaması zorunlu değil.
**Fallback:** Model etiketleri hiç kullanmazsa reflection eskisi gibi serbest
metin olarak çalışmaya devam eder — davranışsal fark yok, sadece disiplin
artışı şansı.
**Doğrulama:** `make verify-local` + birkaç çok-seviyeli oyunda (`make
play-local GAME=<çok seviyeli oyun>`) reflection çıktısını `LLM_TRACE_PATH`
JSONL'inden manuel gözden geçir — etiketlerin anlamlı kullanılıp
kullanılmadığını doğrula.

---

### Faz 2 — Hafıza mimarisini genişlet (dosya-tabanlı, kod-çalıştırma yok)

**Genel gerekçe:** Faz 1 tek bir oyun/tek bir reflection penceresi içinde
kalıyor. Faz 2, Agent 4'ün "keşif maliyetini bir kez öde" bulgusunu (Pang'in
538 programlık kütüphanesi, Tycho'nun yeniden kullanılabilir planlayıcı
kütüphanesi — Agent 4 §6 madde 2) ve Agent 3'ün PRO-LONG değerlendirmesini
(sabit pencere yerine programatik/tam geçmiş erişimi — Agent 3 §4) mevcut
dosya-tabanlı reflection altyapısının üstüne taşıyor.

#### Faz 2a — Oyunlar-arası kalıcı "mekanizma sözlüğü"

**Kaynak:** Agent 4'ün #2 önceliği ("spring wall, refuel ring, color
rotator" gibi tekrar eden fizik motiflerini oyunlar arasında biriktir"),
Tycho'nun "yeniden kullanılabilir planlayıcı kütüphanesi" fikri (ama
Tycho'da bu **statik/elle yazılmış**, öğrenilmiş değil — Agent 2 §2.9'un
tespit ettiği zayıflık; biz burada gerçekten **öğrenen/biriken** bir
versiyon öneriyoruz).

**Neden bizim mimarimizde özellikle değerli:** `_reflection_memory_path`
(satır 153) `game_id`'ye göre anahtarlanıyor — yani bir oyunda öğrenilen
hiçbir şey başka bir oyuna **hiç aktarılmıyor**. Ama satır 31-33'teki yorum,
`Swarm`'ın oyun başına bir ajanı **eşzamanlı** çalıştırdığını doğruluyor —
tek bir Kaggle koşusu içinde birden fazla oyun aynı anda, aynı vLLM
sürecine karşı çalışıyor. Bu, "bir oyunda erken öğrenilen genel bir
mekanik, hâlâ çalışmakta olan başka bir oyuna anında fayda sağlayabilir"
demek — ama **garanti değil**, fırsatçı bir kazanç (hangi oyunların ne
zaman başlayıp bittiğine bağlı). Bunu dürüstçe böyle çerçevelemek gerekiyor.

**Ne yapılacak:**
1. Yeni sınıf-seviyeli sabit: `SHARED_MECHANISM_MEMORY_PATH`, varsayılan
   `os.path.join(<LLM_MEMORY_DIR taban dizini>, "_shared_mechanisms.md")` —
   `_reflection_memory_path`'in kullandığı aynı `base_dir` mantığını
   kullanır (satır 154-159), ama `game_id` içermez.
2. Yeni bir sınıf-seviyeli `threading.Lock()` (mevcut `_server_lock`,
   satır 120 ile aynı desen) — birden fazla oyun-thread'i aynı anda bu
   dosyayı yazmaya çalışabileceği için.
3. `__init__` (satır 124) içinde, `_load_reflection_memory` (satır 165) ile
   aynı try/except-OSError-fallback desenini kullanan yeni bir
   `_load_shared_mechanisms()` — dosya yoksa/bozuksa boş placeholder
   metniyle devam eder, **asla exception fırlatmaz**.
4. `_build_prompt` (satır 678) ve `_build_reflection_prompt` (satır 1412)
   içine küçük, açıkça etiketlenmiş bir blok ekle: *"Cross-game notes
   (unverified in this game, treat as hints, not facts):"* + sıkı bir
   karakter tavanı (örn. 500 karakter — `MAX_REFLECTION_CHARS` deseniyle
   tutarlı).
5. `_run_reflection` (satır 1472) tamamlandıktan sonra: reflection
   çıktısında (Faz 1b'nin `[CONFIRMED]` etiketleriyle işaretlenmiş)
   genelleştirilebilir bir mekanik varsa (oyun/level'a özgü isimler
   içermeyen, örn. "touching a moving colored bar reverses controllable
   object direction" gibi), kilit altında oku-birleştir-yaz (temp dosya +
   `os.replace`, `_save_reflection_memory`'nin satır 1460-1470'teki aynı
   atomik-yazma deseni) ile paylaşılan dosyaya ekle. Basit tekrar-önleme:
   yeni satır, dosyadaki mevcut satırların hiçbiriyle yüksek metin
   benzerliği taşımıyorsa eklenir; dosya toplam karakter tavanını
   (örn. 4000 karakter) aşarsa en eski girdiler düşürülür (FIFO).

**Efor:** 1-2 gün (dosya I/O + kilit + prompt entegrasyonu + basit
dedup/tavan mantığı; en zor kısmı "genelleştirilebilir mi" ayrımını
prompt talimatıyla güvenilir kılmak).
**Risk:** Düşük-orta. Ana risk: prompt şişmesi (mitigasyon: sert karakter
tavanı, mevcut `MAX_REFLECTION_CHARS` deseniyle tutarlı) ve eşzamanlı yazma
çakışması (mitigasyon: kilit + atomik `os.replace`, zaten kanıtlanmış
desen). İkinci risk: yanlış genelleme ("bu oyuna özgü bir şeyi genel sanıp"
başka oyunu yanıltmak) — mitigasyon: blok açıkça "hints, not facts" diye
etiketleniyor ve Faz 1b'nin `[HYPOTHESIS]` disipliniyle aynı şekilde
sorgulanabilir tutuluyor, asla doğrudan aksiyon kısıtlayıcı olarak
kullanılmıyor (sadece prompt bağlamı, `_ineffective_actions_for_current_state`
gibi sert bir filtre değil).
**Fallback:** Dosya okunamazsa/bozuksa boş metinle devam; yazma
başarısız olursa (`OSError`) sessizce atlanır (mevcut
`_save_reflection_memory`'nin `except OSError` deseniyle aynı), oyunun
kendi çalışmasını hiçbir şekilde etkilemez.
**Doğrulama:** `make play-local` (tüm oyunlar, eşzamanlı) — özellikle
paylaşılan dosyanın bozulmadığını (JSON/Markdown olarak hâlâ okunabilir
olduğunu) tüm oyunlar bittikten sonra manuel kontrol et; ayrıca kilit
olmadan bir versiyonla (yanlışlıkla) karşılaştırıp gerçekten çakışma
riski olup olmadığını gözlemlemek faydalı olur (dev-only test).

#### Faz 2b — Tek oyun içinde tam-geçmiş farkındalığı (PRO-LONG-lite)

**Kaynak:** Agent 3 §4 (PRO-LONG'un "ham log + grep" felsefesinin
kod-çalıştırmasız yansıması), Agent 1'in RGB-Agent bulgusu ("minimal
tooling... diminishing returns with hand-engineering" — yani karmaşık bir
mekanizma değil, sadece **tam geçmişe erişim** önemli).

**Sorun:** `_run_reflection` (satır 1472) her seferinde sadece
`self.reflection_buffer[-reflection_interval:]`'i (son 10 transition)
görüyor ve işlendikten sonra `del self.reflection_buffer[:reflection_interval]`
(satır 1513) ile siliyor — daha eski transition'lar, önceki reflection
metnine ne kadar sıkıştırılabildiyse o kadar hayatta kalıyor, gerisi
kalıcı olarak kayboluyor. Uzun/çok-seviyeli bir oyunda bu, "levelin
başında öğrenilen kritik bir kural" seviye 5'e gelindiğinde tamamen
unutulabilir demek.

**Ne yapılacak:** `_observe_frame` (satır 1639) zaten her adımda
`changed_pixels`, `levels_delta`, `repeated_state` hesaplıyor
(satır 1682-1717). Bunu bir adım ileri taşı: `levels_delta != 0` (seviye
tamamlandı) veya `state_changed and not repeated_state` (yeni, tekrarsız bir
duruma geçildi) olan adımları ayrı, sınırlı boyutlu bir
`self.significant_events` listesine (örn. son 50 girdi, FIFO) ekle — bu
liste `MAX_HISTORY=12`'nin aksine **tüm oyun boyunca** hayatta kalır çünkü
önemli olaylar ham adımlardan çok daha seyrek. `_build_reflection_prompt`
(satır 1412) bu özet listeyi (kompakt, `_compact_history_item`, satır 1720
formatında) son-N-transition penceresine **ek olarak** dahil eder.

**Efor:** 1 gün (mevcut `_observe_frame` hesaplamalarını yeniden kullanan
küçük bir filtre + liste + prompt entegrasyonu).
**Risk:** Düşük — sadece ekleme, mevcut pencereleri (`MAX_HISTORY`,
`reflection_buffer`) değiştirmiyor; liste sabit tavanlı olduğu için
sınırsız büyüme riski yok.
**Fallback:** Liste boşsa (henüz önemli olay yoksa) prompt bloğu boş/atlanır;
`_compact_history_item` zaten `None`-güvenli.
**Doğrulama:** Çok-seviyeli bir oyunda (`make play-local GAME=<>` ile 500+
adım) reflection çıktısının erken-seviye kurallarını geç seviyede hâlâ
doğru hatırlayıp hatırlamadığını `LLM_TRACE_PATH` JSONL'inden gözden geçir.

#### Faz 2c — Dinamik hesaplama bütçesi politikası

**Kaynak:** Agent 4'ün #4 önceliği — RHAE'nin "reasoning ücretsiz" teşviki
ile Kaggle'ın gerçek 9 saatlik/tek-GPU kısıtını uzlaştırma ihtiyacı; ayrıca
Agent 4'ün Kaggle'ın kendi 500-submission analizi bulgusu (başarısızlıkların
büyük kısmı algoritma kalitesiyle değil mühendislik/bütçe hatalarıyla
ilgili).

**Zaten var olan emsal:** `_action_candidate_count` (satır 897) zaten
`game_time_remaining_s < LLM_CANDIDATE_MIN_SECONDS` (varsayılan 900s)
olduğunda aday sayısını 1'e düşürüyor (satır 902-904) — yani "kalan bütçeye
göre pahalı özellikleri kıs" deseni zaten kabul edilmiş, kanıtlanmış bir
yaklaşım. Faz 2c bunu genelleştiriyor.

**Ne yapılacak:** Yeni bir yardımcı `_adaptive_max_new_tokens()` ekle:
`cls._remaining_global_seconds()` / `GLOBAL_TIME_LIMIT_SECONDS` oranına göre
(örn. kalan bütçenin >%50'si → tam `LLM_MAX_NEW_TOKENS`; <%15'i → tavanın
%60'ı gibi basit, deterministik, doğrusal bir enterpolasyon) `max_new_tokens`
değerini ölçekle. `_generate_responses` (satır 1075) içinde
`token_budget = max_new_tokens or int(os.getenv(...))` satırının (1087-1089)
hemen öncesine, açık bir env flag'le (`LLM_ADAPTIVE_TOKEN_BUDGET`,
varsayılan `"0"` — Bölüm 4 madde 5'teki "yeni flag varsayılan olarak eski
davranışı korur" kuralına uygun) korunan bir çağrı ekle. **Toplam oyun sayısı
bilinmediği için** (her `MyAgent` örneği tek bir oyunu biliyor, `Swarm`'ın
toplam oyun sayısını görmüyor), tasarım kasıtlı olarak **sadece global kalan
zamanı** sinyal olarak kullanıyor — bu, bilinmeyen bir değeri tahmin etmeye
çalışmaktan daha güvenli.

**Güvenlik tavanı/tabanı:** Enterpolasyon asla `REPAIR_MAX_NEW_TOKENS`
(256) değerinin altına inmemeli — aksi halde JSON çıktısı sistematik olarak
kesilir ve `_extract_action_json` sürekli onarım turuna düşer (kendi
kendini yiyen bir döngü riski). Bu alt sınır kodda sabit bir `max(256, ...)`
olarak yazılmalı.

**Efor:** 1 gün (küçük, saf fonksiyon + tek entegrasyon noktası + tavan/taban
sabitleri).
**Risk:** Düşük-orta. Ana risk: token bütçesi kısıldığında model kalitesinin
düşmesi (daha kısa plan/daha az açıklama) — ama bu tam olarak istenen
davranış (bütçe tükeniyorsa tutumlu ol); yanlış hesaplama riski basit,
saf bir fonksiyonla sınırlı, birim-test edilebilir.
**Fallback:** Env değeri parse edilemezse veya hesaplama herhangi bir
istisna fırlatırsa (`except Exception`), sabit `LLM_MAX_NEW_TOKENS`
değerine dön — asla `choose_action`'ı kırmaz.
**Doğrulama:** `make play-local` sırasında `LLM_TIMING` loglarından
`generate_response` sürelerinin ve `finish_reason` alanının (satır
1159-1163'teki `"length"` uyarısı) kalan-zaman azaldıkça beklenen şekilde
davranıp davranmadığını izle; ideal test, kasıtlı olarak düşük bir
`AGENT_GLOBAL_TIME_LIMIT_SECONDS` ile kısa bir koşu yapıp adaptif
davranışı zorlamak.

---

### Faz 3 — Yüksek efor / izole risk, opsiyonel ve dikkatli pilot gerektirir

#### Faz 3a — Tek-turluk, prompt-tetiklemeli "kod round"u (duck-harness-lite)

**Kaynak:** Agent 3'ün 4. sıradaki önerisi; Agent 2'nin duck-harness
sandbox mimarisinin ayrıntılı belgelemesi (§5.3: `subprocess.Popen`
izolasyonu, `SAFE_BUILTINS`/`SAFE_MODULES` allowlist, satır-sınırlı JSON
stdin/stdout protokolü — **gerçek kod, taklit edilebilir bir desen**).

**Neden gerçek vLLM tool-calling değil, prompt-tetiklemeli tek tur:**
duck-harness'ın gerçek mekanizması vLLM native tool-calling'e
(`tool_call_parser: qwen3_coder`) dayanıyor; Agent 3'ün doğruladığı gibi
bunun gemma-4-31b-it'te desteklenip desteklenmediği **bilinmiyor**, ve
bizim kodumuzun kendi yorum satırı (satır 1122-1130) guided-JSON + thinking
+ tool-calling kombinasyonlarının ne kadar kırılgan olduğunu zaten
belgeliyor. Bu yüzden Faz 3a, vLLM'in native tool-calling'ine **hiç
dokunmadan**, tamamen bizim tarafımızda ayrıştırılan bir konvansiyon
öneriyor.

**Tasarım (yüksek seviye, önce spike gerekir, burada kod yazılmıyor):**
- Yeni env flag `LLM_CODE_ROUND` (varsayılan `"0"`).
- Açıksa: modelin JSON yanıtı isteğe bağlı bir `{"request_code": "<kısa
  python>"}` alanı içerebilir. Bu alan varsa, nihai aksiyon planını
  istemeden önce **tek bir** ek tur çalıştırılır.
- Kod, duck-harness'ın kanıtlanmış deseniyle (`subprocess.Popen`,
  `-I -S` yalıtılmış yorumlayıcı, sabit kısa `timeout`, dosya sistemi/ağ
  erişimi yok, allowlist'li `SAFE_BUILTINS`/`SAFE_MODULES` — `os`, `sys`,
  `subprocess`, soket **yok**) izole bir alt süreçte çalıştırılır; girdi
  olarak sadece `_state_abstraction`/`_frame_descriptor` gibi zaten var
  olan, salt-okunur yardımcı verilerin serileştirilmiş bir anlık görüntüsü
  verilir (canlı `self` nesnesi asla değil).
- Zaman aşımı `subprocess.communicate(timeout=...)` ile (platform-bağımsız,
  `signal.alarm` **değil** — Windows/Kaggle'da thread içi sinyal
  çalışmaz).
- **Herhangi bir hata, zaman aşımı, veya bozuk çıktıda** — sessizce mevcut
  salt-JSON yoluna düş; kod turu bir tane ile sınırlı (döngü yok), adım
  başına toplam süre bütçesine eklenen ek süre sabit bir tavanla sınırlı.

**Efor:** 3-5 gün (sandbox'ın kendisi + protokol + geniş test).
**Risk:** Orta-yüksek. Üç bağımsız risk kaynağı: (1) gemma-4-31b-it'in
kod kalitesi belirsiz — kodlamaya özel eğitilmemiş; (2) sandbox'ın kendisi
yeni bir saldırı/hata yüzeyi (kaynak tüketimi, sonsuz döngü) — mitigasyon
yukarıdaki timeout/allowlist; (3) adım başına ek gecikme —
`game_time_remaining_s` bütçesiyle çatışabilir.
**Fallback:** Flag kapalıyken (varsayılan) sıfır davranış değişikliği;
flag açıkken herhangi bir sandbox hatası normal JSON akışına düşer, asla
`choose_action`'ı kesintiye uğratmaz.
**Doğrulama:** Önce **tamamen izole bir dev spike'ı** olarak (bu repodaki
`agent/my_agent.py`'ye commit edilmeden önce) gemma-4-31b-it'in üretebileceği
kısa Python'un kalitesini ayrı bir script'te test et. Kod tabanına
girdikten sonra, `make play-local` ile flag açık/kapalı **tam koşu**
karşılaştırması (skor + toplam süre) yapılmadan üretim profiline
(`PROFILE_ENV`) hiçbir zaman eklenmemeli.

#### Faz 3b — (Spekülatif, taahhüt edilmiyor) Sınırlı yürütülebilir dünya modeli

Faz 3a gemma-4-31b-it'in güvenilir kısa Python üretebildiğini kanıtlarsa,
astroseger/Tycho'nun "tahmin edilen kareyi gerçek kareyle karşılaştır,
uyuşmazlıkta planı iptal et" fikrinin **çok küçültülmüş** bir versiyonu
(tam replay-doğrulaması değil, sadece tek-adım tahmin-vs-gerçek karşılaştırması)
düşünülebilir. Bu belgede **taahhüt edilmiyor** — sadece Faz 3a başarılı
olursa açılan bir kapı olarak not ediliyor. Astroseger'in kendi ikinci
makalesinin bulgusu hatırlanmalı (Agent 1 §1): "executable world model tek
başına değil, doğrulamanın kendisi asıl performans kaynağı" — yani bu
fikrin değeri esas olarak **doğrulama disiplini**nden geliyor, "kod
yazdırma" tek başına sihirli değil.

---

## 6. Özet tablo

| Faz | Değişiklik | Dosya/fonksiyon | Efor | Risk | Beklenti |
|---|---|---|---|---|---|
| 0 | Temel ölçüm | (ölçüm, kod yok) | ~1 saat | yok | — |
| 1a | Segmentation zenginleştirme + yeniden değerlendirme | `_frame_descriptor` (1923), `_state_abstraction` (2021) | 3-4 saat | düşük | mütevazı-olumlu |
| 1b | Yanlışlanabilir hipotez etiketleme | `_build_reflection_prompt` (1412) | 1-2 saat | ~sıfır | mütevazı-olumlu |
| 2a | Oyunlar-arası mekanizma sözlüğü | yeni: `_load_shared_mechanisms`, `__init__` (124) | 1-2 gün | düşük-orta | belirsiz ama düşük maliyetli opsiyonellik |
| 2b | Tam-oyun önemli-olay farkındalığı | `_observe_frame` (1639), `_build_reflection_prompt` (1412) | 1 gün | düşük | uzun oyunlarda hafıza kaybını azaltma |
| 2c | Dinamik hesaplama bütçesi | `_generate_responses` (1075), yeni `_adaptive_max_new_tokens` | 1 gün | düşük-orta | bütçe güvenliği + geç-oyun tutumluluğu |
| 3a | Tek-turluk kod round'u | yeni sandbox modülü, `choose_action` (510) öncesi opsiyonel adım | 3-5 gün | orta-yüksek | belirsiz, spike gerektirir |
| 3b | Küçük yürütülebilir dünya modeli | (spekülatif, taahhüt yok) | — | — | — |

**Önerilen sıra:** 0 → 1a → 1b → 2b → 2c → 2a → (spike sonucu görüldükten
sonra) 3a. 2b/2c'yi 2a'nın önüne aldım çünkü ikisi de tek-oyun kapsamında
kalıyor (daha küçük blast radius), 2a ise eşzamanlı thread'ler arası
paylaşılan durum gerektiriyor (daha fazla dikkat ister). Her faz kendi
başına devreye alınabilir/geri alınabilir; birbirine sıkı sıkıya bağımlı
değiller.

---

## 7. Yapılmaması gerekenler

- **Docker/Docker-in-Docker sandbox (Tycho, RGB-Agent/PRO-LONG'un tam
  sürümü).** Kaggle kernel'i muhtemelen ayrıcalıklı/iç-içe konteyner
  çalıştırmaya izin vermiyor; bunu denemenin başarısızlık modu "sessiz
  submission çöküşü" — tam olarak Agent 4'ün aktardığı Kaggle'ın kendi
  500-submission analizinin en sık gördüğü hata sınıfı. Doğrulamaya
  değmeyecek kadar yüksek riskli/düşük olasılıklı bir yol.
- **Barındırılan/internet gerektiren herhangi bir model çağrısı**
  (Tycho'nun varsayılan Anthropic/OpenAI API'si, symbolica arcgentica'nın
  Agentica SDK + `agentica-server`'ı, RGB-Agent/PRO-LONG'un Claude Code/
  Codex CLI backend'leri). `isInternetEnabled: False`
  (`build_notebook.py:602`) bunu tartışmasız diskalifiye ediyor — puanlanan
  koşuda denenirse ajan başlangıçta veya ilk çağrıda çöker.
- **İkinci, aynı GPU'da servis edilen bir model** (küçük bir "arbiter"
  modeli, ayrı bir "coder" modeli). Tek RTX 6000, tek vLLM süreci zaten
  `--gpu-memory-utilization 0.94`'e yakın çalışıyor
  (`_start_vllm_server`, satır 330); ikinci bir model VRAM yarışına girer
  ve mevcut `_degrade_startup_settings`/`_ensure_vllm_available` yeniden
  deneme mantığının tamamının iki model için ayrı ayrı test edilmesini
  gerektirir — kazanç belirsizken risk yüzeyi ikiye katlanır.
- **Gerçek vLLM native tool-calling / çok-turlu agentic döngü**
  (`--tool-call-parser` üzerinden, duck-harness'ın gerçek mekanizması).
  gemma-4-31b-it için desteklenen bir parser olup olmadığı doğrulanmadı;
  kodun kendi yorum satırı (1122-1130) guided-JSON + thinking + tool-calling
  kombinasyonlarının kırılganlığını zaten belgeliyor. Ayrı, izole bir
  spike olmadan asla denenmemeli, ve o spike bile Faz 3a'nın basit
  prompt-tetiklemeli alternatifinden **sonra** düşünülmeli.
- **Gerçek oyun denemelerinde naif N-yönlü self-consistency**
  (N bağımsız tam-oyun denemesi yapıp en iyisini seçmek). RHAE'nin kareli
  cezası altında bu neredeyse kesin skor kaybı demek (Agent 4 §4.2,
  schema'nın kendi Figure 4'ü: 2.7× fazla aksiyon harcayan bir ajan
  %14'ün altına düşüyor). Tek güvenli self-consistency biçimi, gerçek
  ortama dokunmadan yapılanıdır (Faz 1b/2a zaten bu ilkeyi izliyor).
- **"[schema]"nın %99 sayısını hedef/karşılaştırma noktası olarak
  kullanmak.** Bkz. Bölüm 8 — kendi yazarları tarafından "self-reported,
  ARC Prize tarafından doğrulanmamış" diye işaretlenmiş.
- **Eğitilmiş bir "beklenen bilgi kazancı" değer modeli** (Agent 4 §6
  madde 3'ün tam/eğitilmiş versiyonu). İnternet yokluğu, eğitimin tamamen
  çevrimdışı yapılıp bir Kaggle veri kümesi olarak paketlenmesini
  gerektirir — gerçek altyapı yatırımı, belirsiz getiri. Bunun yerine
  (eğer gerekirse) `_candidate_static_score` (satır 910) tarzı, elle
  yazılmış, in-file bir sezgisel skorlayıcı tercih edilmeli; eğitilmiş bir
  model Faz 4+ stretch olarak bile önerilmiyor.
- **`LLM_ACTION_CANDIDATES`/`LLM_CANDIDATE_ARBITER`'ı ölçmeden üretime geri
  açmak.** Ekip bunu commit `e0ae610` ile bilinçli olarak kapattı (tek
  derin-düşünen çağrı lehine). Bu kararı bozmadan önce, tam bir 25-oyunluk
  `make play-local` koşusunda gerçek zaman maliyetini ölçmeden asla
  `PROFILE_ENV`'de değiştirilmemeli.

---

## 8. RHAE ve "[schema]" hakkında ek not

**RHAE felsefesi** zaten Bölüm 1'de bu belgenin omurgası olarak
işlendi — özetle: reasoning/tool-call/retry resmi olarak "aksiyon"
sayılmıyor, sadece gerçek ortam-etkileşimleri kareyle cezalandırılıyor.
Bu, Faz 1-2'nin tamamının "gerçek aksiyondan önce ucuza anla" ilkesini
neden önceliklendirdiğinin doğrudan gerekçesi. Ama Kaggle'ın 9 saatlik/
tek-GPU bütçesi gerçek ve reasoning'in "bedava" olması bunu değiştirmiyor
— Faz 2c bu iki gerçeği açıkça uzlaştırmak için var.

**"[schema]" (Impossible Research, schema-harness.github.io) hakkında:**
Agent 4'ün ayrıntılı incelemesi (Agent 4 §5) şunu doğruladı: **%98.98 /
%95.35** rakamları yazarların **kendi sayfasında iki kez** "self-reported,
ARC Prize tarafından doğrulanmamış" diye işaretleniyor; aynı grafikte
ARC-Prize-doğrulanmış en iyi sonuç (GPT-5.6 Sol Max, Temmuz 2026) Public'te
**%13.33**, Semi-private'te **%7.78** — yani doğrulanmış rakamla self-reported
rakam arasında ~85 puanlık bir uçurum var. Kod da açık değil (sadece
"skor iddiası yapmayan" bir clean-room yeniden inşası mevcut,
`github.com/Erikiss/Rebuild-schema-harness-by-impossible-research`), ve
Kaggle topluluğu (discussion/727629) aynı şüpheleri (açık kaynak değil,
sadece public sette ölçülmüş, şüpheli "fallback = en iyi modeli tut"
kuralı) zaten dile getirmiş.

**Sonuç:** [schema]'nın sayısal iddiası bu yol haritasında **hiçbir yerde
hedef olarak kullanılmadı** — Faz 1b ve 2a'da atıfta bulunulan tek şey,
**mimari deseni** ("durum-temellendirme + mekanizma-keşfini tek düzenlenebilir
programda birleştir", "tam geçmişe karşı backtest", "reality outranks the
model") — ki bu zaten astroseger/Tycho'nun daha iyi doğrulanmış
versiyonlarıyla örtüşüyor ve bağımsız olarak sağlam bir tasarım fikri.
Kısacası: **fikri aldık, sayıyı almadık.**

---

## 9. Kapanış

Bu yol haritası kasıtlı olarak mütevazı: hiçbir fazı, incelenen 5 projenin
iddia ettiği türden büyük puan sıçramalarını vaat etmiyor, çünkü o
sıçramaların neredeyse tamamı (RGB-Agent/PRO-LONG'un +18 puanı, Tycho'nun
100.00 RHAE'si, symbolica'nın %36.08'i) hosted, çok daha güçlü frontier
modellerle ve/veya internet erişimiyle elde edilmiş — Agent 3'ün en sert ve
en doğru tespiti bu. Gerçekçi hedef, gemma-4-31b-it + tek GPU + 9 saat +
internetsiz kısıtları içinde, **mevcut sağlam altyapıyı (reflection,
etkisiz-aksiyon hafızası, deadline yönetimi, fallback zinciri) bozmadan**
kademeli, ölçülebilir, geri alınabilir iyileştirmeler biriktirmek.

Önerilen ilk somut adım: **Faz 0** (temel ölçüm) ve **Faz 1b** (hipotez
etiketleme) — ikisi de bir günden az sürer, sıfıra yakın risk taşır, ve
Faz 2/3'ün üstüne inşa edileceği ölçüm/disiplin temelini kurar.
