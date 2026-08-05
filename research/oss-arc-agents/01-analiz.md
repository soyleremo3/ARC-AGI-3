# Agent 1 — Açık Kaynak ARC-AGI-3 Ajan Projeleri: Yorumlayıcı Analiz

**Yöntem:** Her repo `git clone --depth 1` ile klonlandı (Symbolica için ayrıca `symbolica/arcgentica` dalı), README'ler tam okundu, dizin yapısı ve kilit kaynak dosyalar (prompt şablonları, ana döngü, "world model"/hypothesis/planner uygulamaları) incelendi. İddialar mümkün olduğunca birincil kaynaklarla (arXiv metadata, resmi blog yazıları, repo içi config/prompt dosyaları) çapraz doğrulandı. Doğrulanamayan noktalar açıkça "doğrulanamadı" olarak işaretlendi.

---

## 1. `astroseger/arc-3-agents-baseline1` (Sergey Rodionov)

### Ne yapıyor
Bu repo aslında bir "ajan" reposu değil, **iki makalenin destekleyici materyal deposu** — kök dizinde sadece `papers/paper01/` ve `papers/paper02/` var, başka kod yok. Ajanın kendisi `papers/paper01/secure_baseline1/` altında.

Çekirdek fikir: **OpenAI Codex CLI'ı** (kod yazan bir ajan olarak) harici bir Python kontrolörüyle (`agent.py`) yönetip, Codex'e her level için şu dört zorunlu teslimatı ürettiriyorlar:
- `world_model_engine.py` — saf `(state, action) -> (new_state, status)` fonksiyonu (oyunun dinamik motoru),
- `world_model_state_io.py` — gözlemden state'e geri kurulum + state'ten ASCII frame'e render,
- `world_model_main_planner.py` — modelin üzerinde arama yapan planlayıcı,
- `world_model.md` — modelin İngilizce ontolojisi + "ad hoc elements inventory" (henüz açıklanamamış her piksel farkının kayıt altına alındığı bir liste — model kendini "hile yapmıyorum" diye denetliyor).

Doğrulama katı: `verify_world_model.py` modeli **geçmişteki tüm attempt'lerin tam ASCII frame replay'i** ile karşılaştırıyor — piksel piksel eşleşmezse "çözüldü" sayılmıyor. `plan_executor.py` her aksiyon adımından sonra tahmin edilen frame'i gerçek frame'le karşılaştırıp uyuşmazlıkta **derhal durduruyor**. Ana prompt (`main_prompt.md`, ~600 satır) "Strict Model-First Protocol (No Exceptions)" başlığı altında level 1'den itibaren executable model kurmayı zorunlu kılıyor — bu, incelediğim 5 proje içinde formalizasyonu **en katı ve en zorunlu** olan yaklaşım.

Subagent kullanımı sınırlı ve spesifik: (1) animasyon-analiz subagent'ı (görsel ipuçlarını çapraz doğrulamak için), (2) "generalization critique" subagent'ı — ana ajanın modelinde level-özel hardcode/hile olup olmadığını **adversarial** şekilde denetleyen ayrı bir eleştirmen. Symbolica'daki gibi görevi bölen bir orkestrasyon değil, ana döngüye ikinci göz katan bir mekanizma.

### İddialar — doğrulama
- **"GPT-5.5 high reasoning, 15/25 tam çözüm, %58.12 ortalama RHAE"** → `papers/paper01/README.md` ve arXiv özeti (2605.05138) ile **birebir doğrulandı**. GPT-5.4 high için 8/25, %41.29 de doğrulandı.
- **"AGI 2026'da kabul edildi"** → arXiv metadata'sında `"Accepted for publication at AGI-2026"` notu ve Springer DOI (`10.1007/978-3-032-33195-3_15`) mevcut — **doğrulandı**.
- **Oyun-özel kod yok** → Repo, "eligible for ARC-AGI-3 main leaderboard" iddiasını bizzat şu argümanla destekliyor: prompt/kontrolör/workspace'te oyun adı, level çözümü veya sabit koda rastlanmadı; aynı prompt tüm oyunlarda kullanılıyor. Makul görünüyor.
- **Şeffaflık notu (repo'nun kendi itirafı, iddia listesinde yoktu ama önemli):** Eski (`old_vulnerable_version`) ajan sürümünde bilgi sızıntısı açığı vardı — konteynerin internete erişimi vardı ve GPT-5.5-medium'un dc22 koşusunda ajanın herkese açık bir scorecard'ı indirmiş olabileceğini kendileri tespit edip raporlamışlar. O koşuları makaleden çıkarmışlar ve konteyneri kapatmışlar (artık internet yok, sadece OpenAI'a giden kısıtlı bir proxy var, oyun adı dosya/env'de yok). Bu, camiada nadir görülen bir dürüstlük seviyesi.

### Beklenenden fazlası / nüans
Task brief'te bahsedilmeyen ama çok önemli bir bulgu: **ikinci makale (`arXiv:2607.15439`) kendi ana tezini sorguluyor.** Rodionov 7 varyantlı bir ablasyon çalışması yapmış (`twma`=metinsel model, `ewma`=executable model, `_s`=+sadeleştirme, `_v`=+doğrulama). Kendi özetinden alıntı: *"Requiring a persistent executable deliverable is not universally beneficial: the textual variant outperforms the flexible-interface executable variant in both gpt-5.5 settings."* Yani "executable world model" tek başına değil, **doğrulamanın (exact-replay verification) kendisi** asıl performans kaynağı — bu nüans, "executable world model = daha iyi" şeklindeki basit pazarlama anlatısını kendi yazarları tarafından düzeltiyor. Aynı ikinci makalede, gelecekteki bir model olan gpt-5.6-sol ile `ewma_sv` varyantı 25 oyunun tamamını çözüp ~%99 RHAE'ye ulaşıyor ve insan aksiyon bütçesinin yarısından azını kullanıyor — ama yazarlar bunu bizzat *"saturation of the public set — not evidence that ARC-AGI-3 has been solved generally"* diye çerçeveliyor. Bilimsel özenin projenin en güçlü yanı olduğunu söyleyebilirim.

---

## 2. `NIMI-research/Tycho`

### Ne yapıyor
GitHub org açıklaması tam olarak şunu söylüyor: *"ARC-AGI-3 Solver using Rendered Deterministic Moore Machines"* — yani hipotez formatı literal bir Moore makinesi: `State`, `init_state`, `transition(state, action) -> state`, `render(state) -> grid`, `outcome(state) -> {ongoing, level_complete, game_over}`. Bu format `world_model.py` içine yazılıyor ve **opsiyonel** — Tycho'nun mimarisi kasıtlı olarak Rodionov'un "zorunlu model" yaklaşımından farklı, dört ayrı politika sunuyor:

| Politika | Ne yapıyor |
|---|---|
| `no_world_model` | Saf akıl yürütme, hiç model yok |
| `single` | Ana aktör kendi `world_model.py`'sini yazabilir |
| `orchestrator` | Aktör gerektiğinde özel bir "builder" subagent'ı çağırır |
| `trigger` | Doğrulayıcı modeli **falsifiye ettiğinde** harness otomatik olarak builder'ı tetikler |

Bu dört politika, Rodionov'un ablasyon ruhuna benzer bir "hangi bileşen işe yarıyor" sorgulamasını Tycho'nun kendi mimarisine gömmüş durumda.

Planlayıcı kütüphanesi (`tycho/workspace/wmlib_template.py`, ~84KB) somut ve etkileyici: `plan_bfs`, `plan_astar`, `plan_subgoals` (hedefe kadar arama), `verify(model)` (kayıtlı geçişler üzerinde simülasyon doğruluğu), `verify_outcome(model)` (outcome() fonksiyonunu level-complete/game-over kanıtlarına karşı falsifiye eder), `diff_text` (kayıpsız kompakt hücre-delta açıklaması), `segment`/`composite` (nesne ayrıştırma ↔ birleştirme). Yani "simülasyon üzerinden planlama" iddiası kod seviyesinde gerçek bir BFS/A* arama olarak doğrulanıyor.

Güvenlik/izolasyon: Ajanın yazdığı Python kodu ağsız, salt-okunur root dosya sistemli, kaynak sınırlı bir Docker/Finch konteynerinde çalışıyor; `PUBLIC_RELEASE_MANIFEST.json` her izlenen dosya için SHA-256 tutuyor ve `make validate` API anahtarı gerektirmeden bütünlük + olası sır sızıntısı taraması yapıyor.

### İddialar — doğrulama
README'deki resmi tablo (arcprize.org scorecard linkleriyle):

| Politika | Model | RHAE |
|---|---|---:|
| no_world_model | Opus 4.8 | 79.07 |
| single | Opus 4.8 | 85.36 |
| orchestrator | Opus 4.8 | **88.49** |
| trigger | Opus 4.8 | 83.07 |
| orchestrator | GPT-5.6 Sol | **100.00** |
| orchestrator | Opus 5 | **100.00** |

Bu tablo doğrudan repo'dan alındı ve her satır bir `arcprize.org/scorecards/...` linkine bağlı — yani self-reported değil, **resmi doğrulanmış scorecard**. İlginç bir bulgu: `trigger` (otomatik tetikleme, 83.07) `orchestrator`'dan (aktörün kendi kararıyla builder çağırması, 88.49) **daha kötü** — yani modele "ne zaman model kurmalıyım" kararını bırakmak, mekanik bir tetikleyiciden daha iyi sonuç veriyor. Bu, Tycho ekibinin kendi tasarımı içinde bulduğu ince bir sonuç.

### Doğrulanan ek iddia: "Living Survey"
Repo içinde "survey" kelimesine hiç rastlanmadı, ancak `CITATION.cff` yazarları (Jens Lehmann, Andrei Aioanei, Sahar Vahdati) ile arXiv'de arama yapınca doğrudan eşleşme bulundu: **arXiv:2603.13372, "The ARC of Progress towards AGI: A Living Survey of Abstraction and Reasoning"** (Vahdati, Aioanei, Suresh, Lehmann — Mart 2026). Özetten: *"the first cross-generation analysis of 82 approaches across three benchmark versions and the ARC Prize 2024-2025 competitions."* ACM Computing Surveys'e gönderilmiş, canlı site: `https://nimi-ai.com/arc-survey/`. **Bu iddia tam olarak doğrulandı** — aynı ekip, Tycho'yu yazmadan önce alanın kapsamlı bir haritasını çıkarmış.

### Nüans
Tycho'nun "advisory" felsefesi (`orchestrator` modunda bile "the actor can keep exploring or reason directly" — ARCHITECTURE.md) baseline1'in "No Exceptions" zorunluluğunun tam tersi bir tasarım kararı. İkisi de RHAE'de güçlü sonuçlar alıyor, bu da "modelin zorunlu mu opsiyonel mi olması gerektiği" sorusunun henüz kapanmadığını gösteriyor.

---

## 3. Symbolica AI — `symbolica-ai/ARC-AGI-3-Agents` (`symbolica/arcgentica` dalı) ve `symbolica-ai/arcgentica`

### Önemli netleştirme: bunlar iki farklı proje
Görev tanımında ikisi birlikte anılmış ama aslında **iki ayrı repo, iki ayrı benchmark**:
- **`symbolica-ai/arcgentica`** (bağımsız repo) — **statik ARC-AGI-2** için: alt-ajanlar örnek giriş/çıkış gridlerini analiz edip Python transform programları yazıp test ediyor (klasik program sentezi). Opus 4.6 ile ARC-AGI-2 public eval'de **%85.28** — bu interaktif oyun değil, klasik ARC bulmacası.
- **`symbolica-ai/ARC-AGI-3-Agents`** reposunun **`symbolica/arcgentica` dalı** — asıl ilgilendiğimiz **ARC-AGI-3 (interaktif oyun)** harness'ı, "Agentica SDK" üzerine kurulu, adı da **"Arcgentica"**.

İkisi aynı "Agentica" orkestrasyon SDK'sını paylaşıyor ama biri statik bulmaca sentezine, diğeri interaktif oyun oynamaya uygulanmış — kardeş projeler, aynı proje değil.

### Ne yapıyor (ARC-AGI-3 harness'ı)
`agents/templates/agentica/IDEA.md` dosyasının kendi ifadesiyle bu bir **"RLM" harness** (`What's particular about the Arcgentica "RLM" harness...`) — yani Symbolica bunu bizzat Recursive Language Model fikriyle ilişkilendiriyor, task brief'teki yorum doğrudan kaynağında teyit edildi.

Mimari: Bir **orkestratör hiç oyuna dokunmuyor** (grid'e bakarsa context'i piksel verisiyle dolup stratejik düşünme kapasitesini kaybediyor) — dört rolde alt-ajan kullanıyor:
- **Explorer** — `submit_action` + frame alır, dürtükler, before/after diff'ler.
- **Theorist** — sadece metin özeti alır, `submit_action` **yok** — kurallar hakkında spekülasyon yapmaya zorlanır (aksiyon harcayamaz).
- **Tester** — dar bütçeli `submit_action` ile hipotezi doğrular/çürütür.
- **Solver** — onaylanmış stratejiyle `submit_action` kullanır.

Ajanlar arası paylaşımlı bir `memories` veritabanı var — doğrulanmış gerçekler ile açıkça "hipotez" etiketli varsayımlar ayrı tutuluyor. Context'i dolan bir ajan emekli edilip yeni bir ajan spawn edildiğinde, yeni ajan `memories`'i sorgulayarak eskisinin bulduklarını miras alıyor — "tek ajan tek context" sınırını aşmanın somut mekanizması tam olarak bu (orkestratörün her şeyi manuel aktarmasına gerek yok). Toplam aksiyon bütçesi ~800 (RESET ücretsiz ama attempt'i sıfırlıyor).

### İddialar — doğrulama
- **%36.08 skor** → Symbolica'nın kendi blog sayfasından (`symbolica.ai/blog`, 25 Mart 2026, "From 0% to 36% on Day 1 of ARC-AGI-3") **doğrudan doğrulandı**: *"Our implementation achieves a score of 36.08% with the Agentica SDK on the ARC-AGI-3 public evaluation set, outperforming base model CoT baselines of 0.2% (Opus 4.6 Max) and 0.3% (GPT 5.4 High)."* — bu aslında task brief'ten daha kesin bir çerçeveleme (~120-180x çıplak CoT taban çizgisine göre).
- **"113/182 level, 7/25 oyun"** ve **"$8,900 Opus vs $1,005 Agentica"** → **doğrulanamadı**. Repo içinde bu rakamlara rastlanmadı; blog yazısının tam metnine (SPA client-side routing nedeniyle) ulaşılamadı, sadece blog listeleme kartındaki özet erişilebildi. %36.08 rakamı doğru olduğuna göre bu alt-detaylar muhtemelen doğru ama **bağımsız olarak teyit edilemedi** — entegrasyon planında bu rakamları "Symbolica kaynaklı, doğrulanmamış" diye işaretlemenizi öneririm.

### Mimari fark
Symbolica'da baseline1/Tycho'daki gibi ayrı, replay-doğrulanmış bir simülatör **yok**. "Planlama" gerçek oyunda bütçe-sınırlı Tester/Solver alt-ajanlarının hipotez-test-onay döngüsü şeklinde — yani gerçek aksiyonlarla test ediliyor, kod-simülasyonuyla değil. Bunun yerine tamamen farklı bir eksende yenilik yapıyorlar: **ajanı** (oyunu değil) rollere bölüp recursive/context-tazeleme yoluyla ölçeklendiriyorlar.

---

## 4. `alexisfox7/RGB-Agent` → şu anda **PRO-LONG**'a dönüşmüş

### Kritik bulgu: proje yeniden adlandırılmış ve genişletilmiş
Repo hâlâ `RGB-Agent` adında ama README'nin ilk satırı: *"# PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning"* ve alt kısımda açıkça: *"This repo was formerly the Read-Grep-Bash (RGB) Agent — see the original blog post on the ARC-AGI-3 preview games."* Yani task brief'in tanımladığı sistem (RGB-Agent, 3 preview oyun, 1.069 aksiyon) artık **eski bir aşama**; repo şu an çok daha geniş kapsamlı, 25-oyunluk tam public sette değerlendirilen ve arXiv'e (2607.20064) çıkmış bir sisteme dönüşmüş.

### Orijinal RGB-Agent (blog: `blog.alexisfox.dev/arcagi3`, "Hill-climbing ARC-AGI-3", 8 Mart 2026, DukeNLP — Alexis Fox, Junlin Wang, Paul Rosu, Bhuwan Dhingra)
ARC-AGI-3'ün henüz sadece 3 önizleme oyunu (vc33, ls20, ft09) varken yayınlanmış. **OpenCode** iskeleti üzerine Claude Opus 4.6 çalıştırıyorlar (task brief'in "OpenCode" iddiası **doğrulandı**), ajana sadece üç araç veriyorlar: `READ()`, `GREP()`, `BASH(*python3)`. Her aksiyon/board/skor `log.txt`'e tek satır olarak ekleniyor (monoton büyüyen, 100k+ satıra çıkabilen ham log) ve ajan bu logu bir kod deposunu gezer gibi grep'liyor.

Kendi ifadeleriyle merkezi tez: *"minimal tooling is sufficient... diminishing (even negative) returns with additional hand-engineering"* — yani hazır bellek modülleri, özetleyiciler, vektör veritabanları **eklemek performansı düşürüyor**; ham logu grep'lemek yeterli ve daha sağlam.

**Sonuç (blog'dan doğrulandı):** 3 oyunu toplam **1.069 aksiyonda** tamamlamışlar. ARC API'nin insan taban çizgisi (ilk kez oynayan ikinci-en-iyi insan) aynı 3 oyun için ~900 aksiyon — yani "insan seviyesine yakın" iddiası doğru ama **insan üstü değil**, ~%19 fazla aksiyon kullanmışlar. Ağustos 2025 önizleme yarışmasındaki en iyi LLM-olmayan keşif ajanları ise levellerin yarısını tamamlamak için ~255.000 aksiyon harcamış — bu karşılaştırmada RGB-Agent gerçekten çarpıcı bir fark yaratıyor. "En düşük kamuya açık aksiyon sayısı" iddiası, kıyaslandıkları referans grubuna göre (Arcgentica dahil, kendi Tablo 1'lerinde) makul görünüyor.

Mimari olarak dikkat çekici nokta: RGB-Agent'ta baseline1/Tycho tarzı ayrı bir `world_model.py`/BFS-planner **yok**. Tek "modelleme" mekanizması: ham gerçek geçmişi bir dosyada tutmak + kodlama ajanının doğal grep/Python yeteneğine güvenmek. Bu, formal executable model fikrine karşı ucuz ve etkili bir alternatif öneriyor.

### Şimdiki hâli: PRO-LONG (arXiv:2607.20064, Temmuz 2026)
Fikir genelleştirilmiş: "programmatic memory" artık iki farklı backend'i (OpenAI Codex CLI, Claude Code CLI) destekliyor, **tam 25 oyunluk public set**'te değerlendiriliyor, ve bellek koşulları sistematik olarak ablate ediliyor (`prolong`=tam log, `lw25`=son 25 aksiyon penceresi, `no-log`=prompt-içi board, `stateless`=her çağrıda workspace sıfırlanıyor). arXiv özetinden doğrulanan rakamlar: ortalama **+18.0 puan** iyileşme (temel kodlama ajanına göre, farklı frontier modellerde), özel harness'lere eşit/üstün (**%76.1 pass@1'e kadar**) ama **4.2-5.8x daha az token**; **"Fable 5"** modeliyle **%97.4 best@2**, toplam **$1.750** maliyetle. ("Fable 5" — kamuya açık kaynaklarda başka doğrulama bulunamayan bir model adı/kod adı; entegrasyon planında bu ismi teyitsiz olarak işaretlemenizi öneririm.)

---

## 5. `Tufalabs/duck-harness` — "The Duck"

### Ne yapıyor
GitHub repo açıklamasında zaten yazıyor: *"The Duck: ARC-AGI-3 inference harness -- winning solution to ARC-AGI-3 Milestone 1."* — **kazanan** iddiası repo meta verisinde doğrulandı. `TAAF` (Tufa ARC-AGI Framework) adlı genel bir `Benchmark`/`GameAPI` çalıştırma katmanının üzerine kurulu; asıl ajan kodu `ARC3-Inference/` altında.

**Model:** `configs/inference.json` içinde doğrudan görülüyor — `"model_name": "vrfai/Qwen3.6-27B-FP8"`, yerel `vLLM` sunucusu üzerinden (`tool_call_parser: qwen3_coder`, `reasoning_parser: qwen3`, thinking açık). Task brief'in **"Qwen 3.6 27B, yerel açık ağırlıklı model"** iddiası **tam doğrulandı** — 5 proje içinde tek yerel/açık-ağırlıklı model kullanan bu.

**Mimari:** Ajana verilen tek araç `python` — her çağrı **tamamen taze bir yorumlayıcı** ile başlıyor (önceki çağrılardan hiçbir değişken kalmıyor, her seferinde yeniden import/tanım gerekiyor). Ham sayısal grid bilinçli olarak **gizlenmiş**; onun yerine `current_frame.segmentation` veriliyor — 4-bağlantılı nesne ayrıştırması (id, renk, çapraz-frame kimliği için hash, piksel sayısı, sınır, içerdiği çocuk nesneler) + `adjacency_list`. Bu, ekibin bilinçli olarak koruduğu **tek** hazır soyutlama; onun dışında BFS/DFS/flood-fill gibi arama algoritmalarını modelin kendi Python kodunda **ad hoc** yazması bekleniyor — harness'ın verdiği doğrulanmış bir planlayıcı yok.

Kendi blogları (`tufalabs.ai/research/duck-harness/`) bunu Kaggle GPU bütçesi kısıtına bilinçli bir yanıt olarak çerçeveliyor ve **doğrudan baseline1 ile kıyaslıyor**: *"the duck harness is an order of magnitude cheaper on each game"* (GPT-5.4 tabanlı executable-world-model ajanına göre) — *"both approaches solve a similar set of games."* 25 public oyunda ortalama skor **"1.6002 ± 0.4475"** olarak raporlanmış — bu muhtemelen RHAE-yüzdesi değil, harness'a özgü bir ölçek (ör. oyun başına ortalama tamamlanan level) olduğundan **dikkatle, etiketlenerek** kullanılmalı.

### İddialar — doğrulama
- **Qwen 3.6 27B, yerel** → **doğrulandı** (config dosyasında).
- **Kaggle Milestone 1 kazananı** → **doğrulandı** (repo açıklaması + README'de Kaggle discussion, blog post, ve bir Machine Learning Street Talk bölümü linkleri var — üçü de dış meşruiyet sinyali).
- **"El yapımı araçlar improvizasyonu engelledi, bu yüzden harness'ı hafif/genel tuttular"** → Blog fetch'i bu cümleyi birebir veremedi (sayfa JS ile render ediliyor, sadece kısmi özet çekilebildi), ama **tasarımın kendisi bu iddiayı güçlü şekilde destekliyor**: tek genel araç (`python`), tek genel soyutlama (segmentation), oyun-özel hiçbir yardımcı fonksiyon yok, prompt dosyasında (`prompts.py`) modele *"maintain a compact working world model"* dendiği hâlde bunun için **hiçbir hazır iskelet/şablon verilmiyor** — model bunu kendi ephemeral Python koduyla her seferinde yeniden inşa ediyor. Sonuç olarak iddiayı tasarımdan **dolaylı olarak doğruladım**, ama blog'daki tam cümleyi teyit edemedim — bu ayrımı burada açıkça belirtiyorum.

### Mimari fark
Duck-harness'ta da (Symbolica ve RGB-Agent gibi) baseline1/Tycho tarzı ayrı, replay-doğrulanmış bir simülatör dosyası **yok**. Zorunlu tuttukları tek şey nesne segmentasyonu; geri kalan her şey (world model, planlama) modelin anlık, kalıcı olmayan Python kodunda "gömülü" — en "hafif" ve en az formalize edilmiş yaklaşım, ama en zayıf/ucuz modelle (yerel 27B) çalışıyor olması bunu haklı çıkarıyor gibi duruyor.

---

## Ekosistem omurgası (bağlam)

- **`arcprize/ARC-AGI-3-Agents`** — resmi iskelet. İnce bir katman: `FrameData`/`GameAction` üzerinden barındırılan ARC-AGI-3 API'sine bağlanma, `random` referans ajanı, `.env` tabanlı API anahtarı, opsiyonel AgentOps gözlemlenebilirlik entegrasyonu. Symbolica'nın iki reposu da bunun fork'u/dalı; muhtemelen diğer projeler de benzer `main.py --agent=... --game=...` CLI kalıbını miras alıyor.
- **`arcprize/ARC-AGI-Community-Leaderboard`** — bilinçli olarak **bir sıralama değil**, bir vitrin. Kabul şartı: genel-amaçlı olmak, kodun (sadece çıktının değil) açık olması, yeni bir katkı sunmak. Self-reported skorlar **gösterilmiyor**; sadece resmi "ARC Prize Verified" scorecard'lar rozet alıyor. Bu, yukarıdaki 5 projenin neden bu kadar sık `arcprize.org/scorecards/...` linklerine veya arXiv preprint'lerine referans verdiğini açıklıyor — bu ekosistemde self-reported sayının epistemik ağırlığı düşük.

---

## Ortak Desen ve Ayrım Eksenleri

**Ortak olan:** Beşi de oyun-özel hardcode kullanmıyor (hepsi bunu açıkça iddia ediyor ve makul kanıtlarla destekliyor); hepsi RHAE'nin aksiyon sayısını karesel cezalandırdığının bilincinde ve "gerçek aksiyon harcamadan önce ucuza doğrula/simüle et" ilkesini bir şekilde uyguluyor (ister kod-simülasyonuyla, ister bütçe-sınırlı test alt-ajanlarıyla, ister grep ile); hepsi çalıştırma kanıtlarını (replay görüntüleyici, hash'lenmiş artifact'lar, tam log'lar) kamuya açık yayınlıyor — bu şeffaflık kültürü, tipik ML liderlik tablosu yarışından belirgin şekilde farklı bir norm.

**Eksen 1 — Hipotez formalizasyonunun zorunluluğu/otomasyonu:**
`baseline1` (level 1'den itibaren zorunlu, "No Exceptions") → `Tycho` (4 politika, hiç-yoktan-otomatik-tetiklemeye kadar, hep "advisory") → `duck-harness` (tek zorunlu soyutlama = nesne segmentasyonu, gerisi anlık/kalıcı-olmayan) → `RGB-Agent/PRO-LONG` (formal modeli bilinçli olarak reddediyor, "ham log + grep yeter" tezi). `Symbolica` bu eksenin tamamen dışında: oyunu değil, **ajanın kendisini** rollere bölüyor (Explorer/Theorist/Tester/Solver) ve hipotezleri kod-simülasyonuyla değil paylaşımlı doğal-dil `memories` ile ve gerçek aksiyonlarla test ediyor.

**Eksen 2 — Tekli-context/ajan vs. recursive/çoklu-ajan:**
Symbolica burada belirgin bir aykırı değer — orkestratör + 4 rol + paylaşımlı bellek + context dolunca ajanı "emekli edip" yeniden doğurma mekanizması, kendi ifadeleriyle bir "RLM" (Recursive Language Model) harness'ı. `baseline1` subagent'ları sadece dar çapraz-doğrulama için kullanıyor (döngüyü bölmüyor). `Tycho`'nun `orchestrator`/`trigger` modları tek bir "builder" subagent'ı çağırıyor, sürü değil. `duck-harness` ve `RGB-Agent/PRO-LONG` baştan sona tek-ajan tek-context; PRO-LONG'un tüm amacı zaten bu sınırı **ajan çoğaltmadan**, bellek yönetimiyle aşmak.

**Bonus eksen — model bağımlılığı:** `baseline1` ve `Tycho` en güçlü sonuçları frontier/tescilli modellerle (Codex/GPT-5.x, Opus 4.x/5) alıyor ve her iki ekip de kendi ablasyonlarında asıl kazancın model gücü + doğrulama disiplininden geldiğini gösteriyor (ikisi de gelecekteki daha güçlü modellerle "saturation" buluyor). `duck-harness` burada net bir aykırı değer: yerel, açık ağırlıklı, göreceli mütevazı bir 27B modelle rekabetçi sonuç alıyor — harness'ın hafifliği, zayıf temel modeli telafi ediyor. Bu muhtemelen Kaggle'ın kaynak-kısıtlı ortamında kazanmalarının asıl sebebi.
