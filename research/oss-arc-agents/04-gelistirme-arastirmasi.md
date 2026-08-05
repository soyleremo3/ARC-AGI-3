# Ötesi Ne? — 5 Projenin Kapsamadığı Boşluklar ve Gelecek Fikirler

**Agent 4 raporu.** Diğer ajanlar incelenen 5 projeyi (astroseger/baseline1, NIMI-research/Tycho, symbolica-ai/arcgentica, alexisfox7/RGB-Agent, Tufalabs/duck-harness) derinlemesine analiz ediyor. Bu raporun amacı farklı: bu 5 projenin **kapsamadığı** teknikleri, ARC Prize'ın kendi dokümantasyonunu, Kaggle tartışma forumunu, akademik bir "living survey"i ve komşu alanlardaki teknikleri tarayıp, bir Kaggle gönderisini gerçekten ileri taşıyabilecek fikirleri önceliklendirmek.

**Metodoloji notu (önemli):** Bu araştırma web arama sonuçlarını küçük bir özetleyici model üzerinden işleyen bir araç (WebFetch) ve JS render eden bir tarayıcı (Kaggle, schema-harness.github.io için) kullanılarak yapıldı. Arama motoru sonuç sayfalarının WebFetch özetleri **ikinci elden** kabul edilmeli; bunlarda halüsinasyon riski var. Bu yüzden mümkün olan her yerde birincil kaynağa (arxiv sayfası, docs.arcprize.org ham markdown, schema-harness.github.io'nun tarayıcıyla okunan ham metni, Kaggle tartışma sayfasının erişilebilirlik ağacından çekilen gerçek yorumlar) geçtim ve aşağıda hangi bulgunun birincil kaynaktan doğrulandığını, hangisinin sadece bir arama-sonucu özetine dayandığını ayrı ayrı belirttim. Tarih bağlamı: bugün 2026-08-04; ARC-AGI-3 Kaggle yarışması Mart 2026'da başladı, ~3 ay sonra bitiyor.

---

## 1. ARC Prize resmi dokümantasyonu ve Kaggle tartışma forumu (doğrulanmış, birincil kaynak)

### 1.1 RHAE'nin tam formülü ve son değişiklikler

`docs.arcprize.org/methodology` (ham markdown olarak doğrudan okundu) ve bunu teyit eden `docs.arcprize.org/changelog`:

- **Formül:** `level_score = (human_baseline_actions / agent_actions)²`, üst sınır (cap) **1.15×** insan temeliyle sınırlı.
- **14 Nisan 2026 değişikliği (changelog'da doğrulandı):** İnsan temeli önceden "ikinci en iyi performans gösteren kişi" iken artık **"medyan ilk-kez-oynayan insan"** kullanılıyor; per-level cap **1.0×'tan 1.15×'e** yükseltildi; scorecard'lar artık en fazla 24 saat açık kalabiliyor; 15 oyun versiyonu güncellendi. Bu, metodolojinin hâlâ aktif olarak ayarlandığını gösteriyor — 5 projenin analiz ettiği bazı sonuçlar eski cap/baseline kuralına göre üretilmiş olabilir.
- **Oyun-içi ağırlıklandırma:** Seviyeler 1'den n'e artan ağırlık alıyor (geç seviyeler daha değerli), ve bir oyun %100 alabilmesi için **tüm seviyelerin** tamamlanması gerekiyor (tamamlama-tabanlı bir üst sınır var).
- **Aksiyon tanımı:** "İç işlemler (tool call, reasoning adımı, retry) aksiyon sayılmaz — sadece ortamı değiştiren gerçek etkileşimler sayılır." **Bu kritik bir nokta:** RHAE'de LLM'in ne kadar "düşündüğü" resmi olarak ücretsiz; tek kısıtlayıcı gerçek ortam-aksiyonu sayısı. (Bkz. Bölüm 4.5 — bunun test-time-compute stratejisi için sonucu var.)
- Resmi teknik rapor (`arcprize.org/media/ARC_AGI_3_Technical_Report.pdf`, ayrıca arxiv 2603.24621 "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence") dört-eksenli bir çerçeve tanımlıyor: **exploration, modeling, goal inference, planning** — 5 projenin hiçbiri bu dört ekseni açıkça bu şekilde ayrıştırıp ayrı ayrı ölçmüyor (hepsi örtük olarak dördünü de yapıyor ama hiçbiri "hangi eksende zayıfız" diye kendini teşhis etmiyor).

### 1.2 Kaggle forumundaki doğrulanmış/gerçek skorlar — "self-reported" ile "verified" arasındaki uçurum

`kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion` sayfasını tarayıcıyla (erişilebilirlik ağacı üzerinden, gerçek yorumlar dahil) okudum. Öne çıkan bulgular:

- **"Is 100% Accuracy Realistic With the Available Compute?"** (discussion/728278) başlıklı tartışmada katılımcılar, Kaggle'ın verdiği donanımla (**tek RTX Pro 6000 Blackwell, 96 GB VRAM, ~9 saat çalışma süresi, internet kapalı**) gerçekçi skorların şu an **%1 civarında** olduğunu, %10'a çıkmanın bile "büyük başarı" sayılacağını yazıyor. Bir katılımcı "Kimi K3" ile kamuya açık verilerde %70 aldığını iddia ediyor — ama bu **doğrulanmamış, tek kişilik, ikinci elden bir forum yorumu**, schema-harness'ten bile daha düşük güvenilirlikte; ciddiye alınmamalı, sadece "böyle iddialar dolaşıyor" diye not düşülüyor.
- **Tufa Labs'in resmi Duck yazısında** (discussion/717133, yazarlar: Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Dries Smit, Michal Tešnar, Stefano Viel — bu 5 projeden biri, derinlemesine analiz başka bir ajanda) kendi bildirdikleri **gerçek Kaggle leaderboard skoru %1.21** (bir ara %1.30 gösterip Kaggle tarafından geri alınmış), aynı submission'ın **standart sapması %0.4'e varan varyansla %0.77'ye kadar düşebildiği** açıkça yazılıyor. Bu, RHAE gibi kareli-ceza içeren bir metrikte tekil bir skorun ne kadar gürültülü olabileceğinin doğrudan kanıtı.
- **"500 Submissions Analyzed — Common Errors"** (discussion/727119, ARC Prize'ın kendisi, Greg Kamradt): 500 başarısız submission incelenmiş. **~%33'ü** notebook içinde iz sürülemeyen sessiz takılmalar (mantık hatası, sonsuz döngü, async deadlock); **~%20'si** GPU ayarlanmadan GPU gerektiren kod göndermek; kalanı (<%5'er) eksik dataset, eksik bağımlılık, `float`in `len()`i olmaması gibi mantık hataları, CUDA OOM, yasak `three.arcprize.org` API'sine istek atmak, salt-okunur `/kaggle/input`'a yazmaya çalışmak. **Bu, 5 projenin hiçbirinin ele almadığı ama gerçek skoru doğrudan etkileyen bir "boşluk":** en gelişmiş algoritma bile bu mühendislik hatalarından biriyle sıfır puan alabilir. Diğer bir tartışmada (discussion/729985) resmi olarak doğrulandı: private skor **submission anında hesaplanıyor, tekrar çalıştırılmıyor**; her koşu hem public hem semi-private/private oyunları oynuyor; wall-clock limiti kesin olarak **9 saat**.
- **Düşük kaliteli / şüpheli iddialara karşı topluluk tepkisi:** "Active Neuro-Symbolic Search Engine via Minimum Description Length" (discussion/730225) başlıklı, gösterişli matematiksel notasyonla (MDL, Kolmogorov complexity) süslenmiş ama gerçek sonuç tablosunda sadece "3 execution steps / SUCCESS" gibi anlamsız/doğrulanamaz veriler içeren bir gönderi **topluluktan -5 oy** almış — yani forum, akademik jargonla süslenmiş ama somut kanıtı olmayan iddialara karşı zaten şüpheci. Bu, "[schema]" iddiasını değerlendirirken de akılda tutulmalı (bkz. Bölüm 5).

**Sonuç:** Kaggle'ın gerçek submission ortamı (tek GPU, internet yok, yerel açık-ağırlıklı model, 9 saat) ile "self-reported" blog yazılarında görülen sonuçlar (genelde hosted frontier model + internet + sınırsız zaman) arasında **devasa bir fark** var. 5 projeden 3'ü (astroseger'in en güçlü varyantı, Tycho, arcgentica) hosted API'lere dayanıyor — Kaggle'daki gerçek skor potansiyeli çok daha düşük. Bu rapor boyunca önerilen fikirleri bu gerçeklikle tartmak gerekiyor.

---

## 2. Tycho'nun "Living Survey"i — 82 yaklaşımın taksonomisi (doğrulanmış, birincil kaynak)

**Kaynak:** "The ARC of Progress towards AGI: A Living Survey of Abstraction and Reasoning", yazarlar Sahar Vahdati, Andrei Aioanei, Haridhra Suresh, Jens Lehmann; arxiv 2603.13372, Mart 2026 (arxiv HTML sürümünden doğrudan okundu). Bu, görev tanımında bahsedilen "Tycho ekibinin living survey'i" ile aynı kaynak.

**Not edilmesi gereken kısıt:** Bu makale çok yeni bir preprint (Mart 2026), henüz geniş çapta hakemli/doğrulanmış literatüre girmemiş olabilir. Sadece bir fetch aracı üzerinden filtrelenmiş bir özet aldım, tablonun tamamını (82 yaklaşımın hepsini) göremedim — aşağıdaki bulgular güvenilir ama muhtemelen eksiksiz değil.

**Taksonomi:** Program sentezi/indüksiyon (DSL arama), transdüktif/nöral (örüntü eşleştirme), test-zamanı adaptasyonu (TTA), hibrit (nöral algı + sembolik doğrulama), ensemble/refinement (çoklu hipotez + oylama/seçim).

**Ne işe yarıyor, ne yaramıyor (Tablo 4/5'ten alıntılar):**

- **"Guided search" (öğrenilmiş sezgisel, nöral ağ veya LLM ile aday sıralama) belirleyici:** "Unguided search rarely exceeds 30%; guided systems reach 72–80%." Yani rehbersiz arama ~%30 tavanına çarpıyor, rehberli arama %72-80'e çıkıyor.
- **Test-zamanı adaptasyonu (TTA) evrensel bir başarı faktörü:** "Present in *every* >70% system; ∼20–30 pt gain over frozen baselines." %70 üzerini geçen **her** sistemde TTA var; donmuş (fine-tune edilmemiş) taban çizgilere göre 20-30 puan kazandırıyor.
- **Refinement loop'lar merkezi:** En iyi sistemler "aday üret → doğrula (feedback sinyaliyle) → tekrarla" döngüsü kuruyor — veri-üretim refinement'ı, çıkarım-zamanı refinement'ı, ensemble refinement'ı olarak üçe ayrılıyor.
- **Kütüphane-tabanlı transfer (araştırma konusu #4 ile doğrudan ilgili):** Pang'in yaklaşımı 1.000 eğitim görevinden **538 programlık kalıcı bir kütüphane** biriktiriyor, bu da görev başına LLM çağrısını 36'dan 10'a düşürüyor — yani **görevler arası çapraz bilgi transferi** doğrudan aksiyon/çağrı verimliliğine dönüşüyor. Bu, ARC-AGI-1/2'nin statik (etkileşimsiz) ortamında yapılmış; ARC-AGI-3'ün etkileşimli oyunlarına doğrudan taşınmamış.
- **Başarısızlık örüntüleri:** "Pure pattern matching" (~%40 tavanı, daha zor kompozisyonlarda çöküyor), "No TTA" (~%30-40 platosu, adaptif sistemlerin 20-30 puan gerisinde).
- **Nesiller arası performans uçurumu (en çarpıcı bulgu):** "all systems, regardless of scale, cost regime, or architectural paradigm, [maintains its] ARC-AGI-1 performance on ARC-AGI-2" — cümlenin can alıcı kısmı şu: **hiçbir** sistem korumuyor, hepsi 2.5-3× performans kaybediyor (örnek: Berman'ın program-sentez yaklaşımı %79.6→%29.4; Wang'ın nöral yaklaşımı %40.3→%5). Aynı uçurumun ARC-AGI-2→3 arasında da (muhtemelen daha da sert biçimde) tekrarladığı görülüyor — survey özetinde ARC-AGI-1 %93.0, ARC-AGI-2 %68.8, ARC-AGI-3 %13 gibi rakamlar geçiyor (bu son rakam grubu ikinci elden bir özetten geliyor, birebir tablo görülemedi — temkinli okunmalı).
- **MCTS/tree-of-thought:** Survey'nin taranan kısımlarında **hiç geçmiyor** — bu, gerçek bir boşruk olabilir ya da sadece özetleyicinin gözden kaçırdığı bir detay olabilir; kesin değil.
- **Self-consistency/çoğunluk oylaması:** Var ("NVARC'ın multi-component voting'i", "AIRV"), ama ARC-AGI 2025 yarışması bağlamında (statik görevler) — ARC-AGI-3'ün "her ek deneme gerçek aksiyon = kareli ceza" ortamına uyarlanmamış.

---

## 3. RHAE-optimal strateji: keşif/sömürü dengesi üzerine ne var?

### 3.1 Bağımsız akademik bulgu: "Explore Before You Solve" (doğrulanmış, birincil kaynak)

**Kaynak:** Liew Keong Han, "Explore Before You Solve: The Speed–Depth Trade-off in Epistemic Agents for ARC-AGI-3", arxiv 2605.25931, 25 Mayıs 2026 (arxiv PDF'den doğrudan okundu — 5 projenin hiçbirinde adı geçmeyen, bağımsız bir akademik çalışma).

- **AERA (Adaptive Epistemic Reasoning Agent):** üç fazlı mimari — Exploration (ortamı sistematik keşfet) → Analysis (epistemik muhakeme ile hipotez kur) → Exploitation (keşfedilen kalıbı uygula). Kavramsal olarak astroseger/baseline1 ve schema-harness'in "önce modelle, sonra planla" mimarisiyle aynı aileden, ama küçük/açık-ağırlıklı bir modelle (**Qwen2.5-0.5B** — 5 projedeki en küçük modelden bile çok daha küçük) test edilmiş.
- **Sonuç:** 25 kamuya açık oyunda **RHAE=0.2116 (4/25 çözüldü)**. Bu düşük bir sayı ama önemli olan şu: yazar, ARC-AGI-3'ün bazı bölümlerinin "zeki olmayan" (pattern-matching, kaba kuvvet) stratejilerle çözülebildiğini iddia ediyor ve bunu bir **benchmark eleştirisi** olarak sunuyor — yani bazı oyunlarda yüksek RHAE, gerçek "dünya modeli inşa etme" becerisinden değil, oyunun kendisinin basit olmasından kaynaklanabilir.
- **"Speed-Depth trade-off framework":** Makalenin matematiksel çerçevesi, hızlı-sığ ile yavaş-derin keşif arasındaki dengeyi formalize etmeye çalışıyor. Fetch aracı üzerinden tam formülü çıkaramadım (PDF'in matematik kısmı düzgün ayrıştırılamadı) — bu yüzden **formülün kendisini burada doğru aktaramıyorum, sadece varlığını ve genel iddiasını** raporluyorum. İlgilenen biri PDF'i doğrudan okumalı: `arxiv.org/pdf/2605.25931v1`.

### 3.2 Schema-harness'in kendi RHAE-optimal iddiası (birincil kaynak, ama sonucun kendisi doğrulanmamış — bkz. Bölüm 5)

Schema'nın blog yazısı, RHAE'nin kareli ceza yapısından **açık bir strateji** türetiyor (bu kısım metodoloji açıklaması, doğrulanabilirlik sorunu olan asıl skor iddiasından ayrı değerlendirilmeli):

> "This targeted experimentation is also efficient under the official metric, which applies a squared penalty to excess actions: the best experiment is the one that resolves the most uncertainty with the fewest real interactions."

Yani: birden fazla aday-kural (hipotez) hâlâ kayıtlı geçmişle tutarlıysa, ajan bu adaylardan **farklı tahmin ürettikleri** bir aksiyonu arıyor (bilgi kazancını maksimize eden deney) ve sadece onu gerçek ortamda deniyor. Model bir kez "sertifiye" edildikten (kayıtlı tüm geçişleri tekrar oynatarak doğrulandıktan) sonra, planlama tamamen **model içinde, gerçek aksiyon harcamadan** (BFS ile 10³-10⁴ durum) yapılıyor — sadece nihai plan gerçek ortamda yürütülüyor.

### 3.3 Kaggle forumunun örtük teorisi (doğrulanmış)

`docs.arcprize.org` GitHub aynasından bir alıntı (arama sonucu özetinden, ikincil kaynak ama docs.arcprize.org/methodology ile tutarlı): "Test takers must use actions in two ways: exploration (learning the rules and building a strategy) and execution (carry out a strategy)." Bu, ARC Prize'ın kendisinin de aynı ikili çerçeveyi (keşif vs. yürütme) kullandığını gösteriyor, ama **resmi dokümantasyon bu ikisi arasındaki optimal oranı sayısal olarak formalize etmiyor** — sadece kavramsal olarak ayırıyor.

### 3.4 Sentez — gerçek bir "teori" var mı?

Dürüst değerlendirme: **Hayır, RHAE-optimal keşif/sömürü dengesi için genel kabul görmüş, matematiksel olarak kanıtlanmış bir teori bulamadım.** Bulunan üç parça (Liew Keong Han'ın "speed-depth" çerçevesi, schema'nın "en çok belirsizliği en az aksiyonla çöz" sezgisel kuralı, ARC Prize'ın "keşif vs yürütme" kavramsal ayrımı) birbirini tamamlıyor ama hiçbiri formel bir optimal politika kanıtı sunmuyor. Kendi sentezim (bu bir **öneri**, doğrulanmış bir bulgu değil): metrik `(human/agent)²` olduğu ve iç muhakeme (reasoning token) ücretsiz sayıldığı için, doğru RHAE-optimal ilke **Bayesçi deneysel tasarım / beklenen bilgi kazancı** çerçevesinden türetilebilir — her gerçek aksiyon, rakip hipotezler arasında en çok ayrım yaratacak şekilde seçilmeli, ve model "sertifiye" olur olmaz kalan tüm arama iç simülatöre taşınmalı. Bu noktayı Bölüm 6'daki öncelikli fikir listesinde somutlaştırıyorum.

---

## 4. Komşu alanlardan teknikler — 5 proje ne kapsamıyor?

### 4.1 Klasik DSL / program sentezi (ARC-AGI-1 dönemi) — kısmen zaten emilmiş

- **Michael Hodel'in `arc-dsl`'i** (arama sonucu üzerinden doğrulandı, ikincil kaynak ama iyi bilinen bir proje): elle yazılmış, az sayıda ama genel primitiflerden oluşan bir DSL + bu DSL üzerinde program arama. 400 eğitim görevinin hepsi için çözücü program yazılmış; sonraki birçok çalışmanın (doğrulayıcılar dahil) temeli olmuş.
- **Ryan Greenblatt'in yaklaşımı** (arama sonucu üzerinden doğrulandı): GPT-4o'ya görev başına **~8.000 Python programı** ürettirip örnekler üzerinde doğru olanları seçerek 2024'te ARC-AGI-1 public set'te %50 (o dönem SOTA) almış.
- **Değerlendirme:** Bu teknikler ARC-AGI-3'e **doğrudan** taşınmıyor çünkü ARC-AGI-1 statik "girdi-ızgara → çıktı-ızgara" görevleri için tasarlandı; ARC-AGI-3 etkileşimli, çok adımlı bir ortam. Ama **fikrin kendisi** (dünya/dönüşüm kuralını yürütülebilir bir program olarak temsil et) zaten astroseger/baseline1 ve schema-harness'in temelinde — yani bu araştırma konusu "kapsanmayan bir boşluk" değil, "zaten absorbe edilmiş bir köken fikir." Asıl boşluk, "8.000 aday üret ve seç" tarzı **kaba-kuvvet çoklu-örnekleme**nin ARC-AGI-3'e hiç taşınmamış olması — çünkü orada her "deneme" gerçek bir aksiyon değil, offline bir kod üretimi olurdu, dolayısıyla RHAE'yi hiç etkilemez. **Bu gerçek bir fırsat:** bir ajan, gerçek ortama dokunmadan, aynı gözlem geçmişine karşı **birden fazla rakip dünya-modeli programı** üretip bunları offline'da (Greenblatt tarzı) birbirine karşı test edip en tutarlısını seçebilir — bu tamamen ücretsizdir (reasoning/tool-call sayılmaz) ve hiçbir 5 proje bunu bu şekilde açıkça yapmıyor (Tycho'nun `orchestrator`/`trigger` politikaları buna yakın ama "N rakip modeli offline üret ve oyla" değil, "bir builder'ı tetikle" mantığında).

### 4.2 Self-consistency / çoğunluk oylaması — ARC-AGI-3'e taşınırsa tehlikeli

Survey'de (Bölüm 2) doğrulandığı gibi ARC-AGI-1/2'de yaygın ama kritik bir uyarı: ARC-AGI-3'te naif self-consistency (N bağımsız tam-oyun denemesi yapıp en iyisini seç) **RHAE'yi N kat gerçek aksiyonla çarpar** — kareli ceza altında bu neredeyse kesin skor kaybı demektir (schema-harness'in Figure 4'ü tam olarak bunu gösteriyor: 2.7× fazla aksiyon harcayan bir ajan %14'ün altına düşüyor). **Tek güvenli self-consistency biçimi**, Bölüm 4.1'de önerilen gibi, oylamanın **gerçek ortama dokunmadan, dünya-modeli programları arasında** yapılmasıdır — bu ayrım hiçbir 5 projenin açıklamasında net biçimde vurgulanmıyor.

### 4.3 MCTS / ağaç arama — kısmen kapsanmış, ince bir boşluk var

Schema-harness zaten "sertifiye model içinde BFS ile 10³-10⁴ durum ara" diyor — yani düz ağaç/graf araması zaten kullanılıyor (deterministik, tam-doğrulanmış modellerde BFS yeterli ve optimal). Gerçek boşluk, model **belirsiz veya kısmen doğrulanmış** olduğunda: hangi rakip hipotezi test etmek için hangi deneyi (aksiyonu) seçeceğine karar vermek bir **keşif problemi** (multi-armed bandit / MCTS'e yakın), düz BFS'in çözdüğü bir "biliniyor, en kısa yolu bul" problemi değil. 5 projenin hiçbiri bu ayrımı (planlama-arama vs. deney-seçme-araması) açıkça iki farklı algoritma olarak ele almıyor — hepsi bunu LLM'in örtük muhakemesine bırakıyor.

### 4.4 Öğrenilmiş değer fonksiyonları / önceliklendirme modelleri — gerçek bir boşluk

Survey'de bulunan "evaluator-based ranking" (Bölüm 2), ARC-AGI-1/2'de **üretilmiş çözüm adaylarını** sıralamak için kullanılıyor — yani sentezden **sonra**. ARC-AGI-3'te asıl ihtiyaç, sentezden **önce**, "hangi aksiyonu denersem en çok öğrenirim" sorusuna cevap veren bir modeldir (aktif öğrenme / Bayesçi deneysel tasarım literatüründeki "expected information gain" / BALD tarzı yaklaşımlar). Hiçbir 5 proje bunun için **eğitilmiş** (fine-tune edilmiş veya klasik ML ile eğitilmiş, sadece prompt edilmiş değil) hafif bir model kullanmıyor — hepsi bu kararı LLM'in kendi (prompt-edilmiş) yargısına bırakıyor. Bu, ucuz (küçük özellik seti: hangi hücreler değişti, kaç rakip hipotez hâlâ tutarlı, vs.) ve potansiyel olarak yüksek etkili bir boşluk.

### 4.5 Test-zamanı hesaplama ölçeklendirmesi — RHAE'nin kendine özgü bükümü

Standart "test-time compute scaling" literatürü (o1/o3 tarzı) "daha fazla düşünme tokeni = daha iyi doğruluk" der. ARC-AGI-3'te bu **kısmen farklı** çalışıyor çünkü (Bölüm 1.1'de doğrulandığı gibi) **reasoning token'ları RHAE'ye hiç girmiyor** — sadece gerçek ortam-aksiyonları sayılıyor. Teorik olarak bu, "gerçek aksiyon almadan önce sınırsız düşün" stratejisini resmi metrik açısından bedava kılıyor. Ama **Kaggle'ın pratik kısıtı** (tek GPU, 9 saat, ~110-300 oyun) bunu gerçekte bedavasız yapıyor — her "düşünme" turu gerçek GPU-saati tüketiyor. **Bu gerilim (resmi metrik teşviki vs. Kaggle'ın pratik zaman/donanım bütçesi) 5 projenin hiçbirinin açıklamasında açıkça ele alınmıyor** — hepsi "ne kadar düşünsek iyi olur" sorusuna kendi sabit bütçe ayarlarıyla (örn. `LLM_ACTION_THINKING`, reasoning-effort parametreleri) cevap veriyor, ama bunun RHAE'nin resmi ücretsizlik kuralından **kaynaklanan bilinçli bir strateji** olduğunu hiçbiri açıkça gerekçelendirmiyor.

### 4.6 Kütüphane/kural transferi oyunlar arası — Tycho kısmen kapsıyor, agresifleştirilebilir

Survey'deki Pang örneği (538 programlık kütüphane) ARC-AGI-1'in statik görevleri arasında. Tycho'nun "yeniden kullanılabilir planlayıcı beceri kütüphanesi" (görev tanımında bahsedilen) ARC-AGI-3'e en yakın örnek. Ama schema-harness'in kendi metninde geçen "spring wall, refuel ring, color rotator" gibi tekrar eden **mekanizma motifleri** — yani farklı oyunların paylaştığı düşük-seviye fizik kuralları — hiçbir 5 projede **oyunlar arası açıkça aranıp yeniden kullanılan** bir "mekanizma sözlüğü" olarak ele alınmıyor. Bu Bölüm 6'da öncelikli fikir olarak öneriliyor.

---

## 5. "[schema]" harness söylentisi — detaylı inceleme (KISMEN DOĞRULANMIŞ, SKOR İDDİASI DOĞRULANMAMIŞ)

Bu bölümü özellikle dikkatli okuyun: aşağıdaki bulguların bir kısmı **birincil kaynaktan doğrudan doğrulandı** (sayfanın kendisini tarayıcıyla okudum), bir kısmı **bağımsız ikinci bir kaynaktan çapraz kontrol edildi**, ama **iddia edilen %99 skor, ARC Prize tarafından doğrulanmamış ve kendi yazarları bunu açıkça kabul ediyor.**

### 5.1 Ne buldum, doğrudan kaynaktan (schema-harness.github.io, tarayıcıyla tam metin okundu)

- Sayfa başlığı: **"Frontier Models with Our Harness Achieve ~99% on ARC-AGI-3 Public — Schema"**. İddia edilen skorlar: **Claude Opus 4.8 & Fable 5 ile %98.98**, **GPT-5.6 Sol ile %95.35** — ama bunlar **sadece 25 kamuya açık oyunda (Public set)**, semi-private set'te hiç ölçülmemiş.
- Sayfanın kendi metninde, resmin (Figure 1) altyazısında **birebir şu ifade var**: *"Both Schema results are self-reported and have not been verified by ARC Prize."* Ve raporun sonunda tekrar: *"The 98.98% and 95.35% scores are self-reported results on the Public set... Neither score has been independently verified by ARC Prize."* — yani bu bir "söylenti" değil, **yazarların kendi açık itirafı**.
- Aynı grafikte, **ARC Prize tarafından doğrulanmış (verified)** en iyi sonuç olarak **GPT-5.6 Sol Max ile Temmuz 2026 itibarıyla Public'te %13.33, Semi-private'te %7.78** gösteriliyor (Mart'taki lansmanda %0.51'den başlayarak). Yani doğrulanmış rakamlarla self-reported %99 arasında **~85 puanlık bir uçurum** var.
- **Metodoloji (skor iddiasından bağımsız olarak değerlendirilmeye değer):** "state grounding" (piksellerden nesne/değişken çıkarma) ve "mechanism discovery" (geçiş kuralını program olarak yazma) **tek, düzenlenebilir bir programda birlikte** çözülüyor. Döngü: observe → deliberate (teori yaz, `run_backtest` ile TÜM kayıtlı geçmişe karşı doğrula, `run_bfs` ile planla) → execute (her adımda tahminle gerçek karşılaştır, uyuşmazlıkta planı iptal et) → record (append-only "Timeline"). Bu, astroseger/baseline1'in "yürütülebilir dünya modeli + replay doğrulama" fikriyle kavramsal olarak neredeyse aynı, ama daha titiz belgelenmiş.
- **Kaynakça (BibTeX, sayfada birebir yer alıyor) yazarları:** Guanning Zeng, Jiani Wang, Wenjie Ma, Shaofeng Yin, Chenyang Wang, Shichen Liu, **Angjoo Kanazawa**, Wode Ni, Xiuyu Li, **Andrea Zanette**, Haiwen Feng. Yayıncı: **"Impossible Research"**, 2026.

### 5.2 Ekibin gerçekliği hakkında çapraz doğrulama (dolaylı ama önemli)

- **Angjoo Kanazawa gerçek, tanınmış bir UC Berkeley öğretim üyesi** (bilgisayarla görü alanında). Bu, "Impossible Research"in tamamen uydurma/anonim bir grup olmadığını, en azından bazı üyelerinin gerçek akademisyenler olduğunu gösteriyor.
- Ayrıca aynı isimlerden üçü (**Guanning Zeng, Haiwen Feng, Andrea Zanette**) 2026'da yayımlanmış ayrı, ilgisiz bir makalede (**"Maximum Likelihood Reinforcement Learning" / MaxRL**, arxiv 2602.02710) **Ruslan Salakhutdinov ve Jeff Schneider** (ikisi de tanınmış **CMU** öğretim üyeleri) ile birlikte ortak yazar olarak geçiyor. Bu, görev tanımındaki "Berkeley/CMU team" ifadesini **dolaylı olarak destekliyor** — ekip gerçekten Berkeley ve CMU'ya bağlı araştırmacılardan oluşuyor gibi görünüyor. **Ama bu, sadece ekibin var olduğunu ve yetkin olduğunu gösterir; %99 skorunun doğru/temsili olduğunu kanıtlamaz.**
- `github.com/impossible-research` adında bir GitHub organizasyonu var ama tek deposu ("common-workflows", Aralık 2024'te son güncellenmiş, Schema/VIGA ile ilgisiz görünüyor) — **Schema'nın orijinal kodu bu organizasyonda yok**, ya da en azından herkese açık değil. Bu, görev tanımındaki "harness'in kendisi tam olarak açık kaynak değil" varsayımını **doğruluyor**.

### 5.3 "Açık kaynak" iddiası — sadece bir "clean-room" yeniden inşası var

`github.com/Erikiss/Rebuild-schema-harness-by-impossible-research` adında bir depo buldum — bu **resmi/orijinal uygulama DEĞİL**, README'de birebir şöyle deniyor: *"A clean-room reproduction of the conceptual agentic structure of the Schema harness by Impossible Research"* ve *"This repo makes no ARC-AGI-3 score claim."* README, 6 tane **yeniden üretilemeyen** unsuru açıkça listeliyor: kesin sistem promptları, kesin model/API versiyonları, Fable-5 fallback mantığı, sandbox/zaman/token limitleri, Impossible Research'ün iç değerlendirme altyapısı, ve modellerin deterministik-olmayan yanıtları. Yani **görev tanımındaki şüphe tamamen doğrulanıyor:** ortada resmi, çalıştırılabilir, tekrar üretilebilir bir açık kaynak yok — sadece bir tanıtım yazısı + toplulukça yapılmış, "skor iddiası yapmayan" bir iskelet var.

### 5.4 Kaggle topluluğunun tepkisi (doğrulanmış, birincil kaynak — discussion/727629)

Kaggle'da `"https://schema-harness.github.io/ pub 99%"` başlıklı bir tartışma var (CreateAMind tarafından açılmış, 5 oy, 3 yorum — tam metnini okudum):

- **ktyser:** *"It looks promising, but it does not seem to be open source. I would also like to see how it performs on the semi private set, since it may be overfit to the public games. The fallback rule also bothers me. It reruns weak games with stronger models and keeps the better score."* — yani topluluk üç somut metodolojik endişe dile getiriyor: (1) açık kaynak değil, (2) sadece public set'te ölçülmüş (overfitting riski), (3) "zayıf oyunu daha güçlü modelle tekrar dene, en iyisini tut" kuralı, tek bir tutarlı politikayı yansıtmıyor olabilir (bu aslında schema'nın kendi yazısında da itiraf edilen bir seçim).
- **Doruk Doğrular:** *"is this over-fitting on seen games?"*
- **Vladimir Yakunin:** İlgisiz görünen ama gerçek bir makaleye (arxiv 2602.02710, yukarıda bahsedilen MaxRL) atıfta bulunup şüpheci bir ton kullanıyor: *"The only question is whether they're implementing it and whether they're simply mistaking wishful thinking for reality."*

### 5.5 Genel değerlendirmem

**%99 rakamına güvenilmemeli** — bu, kendi yazarlarının da açıkça belirttiği gibi, tek bir public-set koşusundan, model-başına-en-iyi-sonucu-tutan bir fallback kuralıyla elde edilmiş, bağımsız doğrulaması olmayan bir sayı. Doğrulanmış en iyi sonucun (~%13, Temmuz 2026) bunun onda biri bile olmaması, ve Kaggle'daki gerçek submission skorlarının (%1 civarı, Bölüm 1.2) bununla kıyaslanamayacak kadar düşük olması, sağlıklı bir şüphecilik gerektiriyor. **Ama** yazının kendisi (ekibin CMU/Berkeley bağlantılı gerçek araştırmacılardan oluşması, metodolojinin ayrıntılı ve iç tutarlı biçimde belgelenmiş olması, sınırlamaların dürüstçe itiraf edilmesi) bunu tamamen bir "hype" gönderisi olarak görmemi de engelliyor. **En dengeli okuma:** mimari fikir (durum-temellendirme + mekanizma-keşfini tek programda birleştirmek, tam-geçmişe-karşı-backtest, hedefli ayrıştırıcı deneyler, model-içi ücretsiz planlama) muhtemelen gerçek ve astroseger/baseline1 + Tycho'nun doğal bir evrimi; **ama mutlak %99 sayısı, semi-private/private sette veya Kaggle'ın gerçek donanım kısıtları altında tekrarlanabileceğinin garantisi olmadan ciddiye alınmamalı.**

---

## 6. Önceliklendirilmiş liste — en umut verici 5 fikir

### 1. Aramayı/self-consistency'yi/ağaç aramasını her zaman "model içinde" yap, gerçek ortama sadece maksimum-bilgi-kazandıran aksiyonları gönder

**Gerekçe:** Bu, Bölüm 1.1 (reasoning ücretsiz), 3.2 (schema'nın kareli-ceza sezgisi), 4.1 (offline çoklu-aday üretimi) ve 4.2'nin (naif self-consistency'nin RHAE'yi N katına çıkarma tehlikesi) doğal birleşimi. RHAE'nin matematiği bunu neredeyse zorunlu kılıyor: `(human/agent)²` formülünde fazladan her gerçek aksiyon karesel olarak cezalandırılırken, iç muhakeme (Bölüm 1.1'de doğrulandığı gibi) resmi olarak bedava. Astroseger/baseline1 ve schema bunu kısmen zaten yapıyor (model-içi BFS planlama); asıl eksik olan, bunu **çoklu-hipotez üretimi + offline oylama** (Greenblatt'in 8.000-örnek fikrinin ARC-AGI-3'e uyarlanmış hâli) ile birleştirip, gerçek ortama sadece rakip hipotezleri ayrıştıran deneyleri göndermektir. Bu fikir doğrudan skorlanan metriği hedeflediği ve 5 projenin ikisinin zaten üzerine inşa edilebileceği kısmi bir temeli olduğu için en yüksek beklenen etkiye sahip.

### 2. Oyunlar-arası kalıcı "mekanizma sözlüğü" (spring wall, refuel ring, carry-state gibi tekrar eden fizik motifleri) inşa et ve gizli oyunlarda geri kullan

**Gerekçe:** Survey'deki Pang örneği (538 programlık kütüphane, LLM çağrısını 36'dan 10'a düşürüyor, Bölüm 2) ve Tycho'nun "yeniden kullanılabilir planlayıcı becerisi kütüphanesi" bunun ARC-AGI-1/statik ve kısmen-ARC-AGI-3 versiyonlarının zaten işe yaradığını gösteriyor. Schema'nın kendi metninde geçen tekrar eden mekanizma isimleri (spring wall, refuel ring, color rotator) ARC-AGI-3 oyunlarının **tamamen birbirinden bağımsız olmadığını**, paylaşılan düşük-seviye "fizik" birimleri olabileceğini ima ediyor. RHAE "keşif maliyetini bir kez öde, sonra modelden ücretsiz planla" mantığı üzerine kurulu olduğu için (schema'nın Observation 1'i tam olarak bunu gösteriyor: M0R0 seviye 4'te 42 aksiyon vs insanın 500'ü), bu keşif maliyetini **oyunlar arasında da bir kez ödemek** — yani 25 kamuya açık oyunda öğrenilen motifleri gizli/semi-private oyunlarda sıfırdan keşfetmek yerine ilk aday olarak denemek — potansiyel olarak en yüksek kaldıraçlı ama hiçbir 5 projede bu netlikte açıkça yapılmayan fikirlerden biri.

### 3. Eğitilmiş (sadece prompt-edilmiş değil), ucuz bir "beklenen bilgi kazancı" değer fonksiyonu ile hangi deneyin/aksiyonun rakip hipotezleri en iyi ayrıştıracağını seç

**Gerekçe:** Bölüm 4.4'te tespit edilen net bir boşluk: survey'nin bulduğu "evaluator-based ranking" sentez-sonrası aday-seçimi için (ARC-AGI-1/2), 5 projenin hepsi ise aksiyon-seçimini LLM'in örtük, prompt-edilmiş yargısına bırakıyor. Aktif öğrenme / Bayesçi deneysel tasarım literatüründeki (BALD tarzı) "beklenen bilgi kazancı" fikri iyi kurulmuş ama ARC-AGI-3 bağlamında hiçbir projede **eğitilmiş bir model** olarak uygulanmıyor. Küçük bir özellik seti (kaç rakip hipotez hâlâ tutarlı, hangi hücreler son aksiyonlarda değişti, hangi aksiyon türü henüz hiç denenmedi) üzerinde eğitilecek hafif bir sıralayıcı (gradient-boosted trees kadar basit olabilir), LLM'in her adımda "hangi deneyi yapayım" diye yeniden düşünmesinin (Bölüm 3.2'deki schema örneğinde Opus'un Fable'a göre ~250 fazla aksiyon harcamasına neden olan gecikmiş temsil-değişikliği gibi) israf ettiği aksiyonları azaltabilir. Riski düşük (mevcut mimariye ek bir sıralama katmanı olarak eklenebilir), ama etkisi doğrulanmamış — bu yüzden 3. sırada.

### 4. Kaggle'ın gerçek donanım/zaman bütçesini (tek GPU, 9 saat, internet yok) RHAE'nin "ücretsiz muhakeme" teşvikiyle açıkça uzlaştıran bir "düşünme bütçesi" politikası + mühendislik-hatalarına karşı sağlamlaştırma

**Gerekçe:** Bölüm 1.2 ve 4.5'te belgelendiği gibi, iki gerçek var: (a) resmi metrik sınırsız iç-muhakemeyi teşvik ediyor, (b) Kaggle'ın kendi 500-submission analizi başarısızlıkların ~%53'ünün (sessiz takılma + GPU ayar hatası) **algoritma kalitesiyle hiç ilgisi olmayan mühendislik hataları** olduğunu gösteriyor. Bu fikir en az "havalı" olanı ama muhtemelen **en yüksek garanti-edilmiş getiriye** sahip: hiçbir sofistike dünya-modeli veya bilgi-kazancı stratejisi, submission sessizce takılıp sıfır puan alırsa işe yaramaz. Somut olarak: her oyun/adım için sert bir wall-clock/adım bütçesi + watchdog, GPU ayarının submission öncesi otomatik doğrulanması, ve "düşünme derinliği" (reasoning-effort, candidate sayısı gibi) parametrelerinin, kalan oyun sayısı ve kalan wall-clock'a göre **dinamik olarak** ayarlanması (erken oyunlarda cömert keşif, tükenmekte olan bütçede agresif tasarruf) — 5 projenin hiçbiri bunu açıkça dinamik bir politika olarak tarif etmiyor, hepsi sabit parametrelerle çalışıyor.

### 5. (Düşük güven / spekülatif) Schema'nın "durum-temellendirme + mekanizma-keşfi tek düzenlenebilir programda, gerçeklik-modeli-geçersiz-kılar" mimari desenini — %99 iddiasından bağımsız olarak — somut bir mimari referans olarak benimse

**Gerekçe:** Bölüm 5'te ayrıntılı tartışıldığı gibi, sayısal iddia (%99) doğrulanmamış ve muhtemelen temsili değil, ama **mimari açıklamanın kendisi** (Bölüm 5.1) astroseger/baseline1'in halihazırda yaptığının daha titiz, daha iyi belgelenmiş bir versiyonu ve bu araştırma turunda bulunan en somut "sıradaki mimari" fikri. Özellikle "tek bir tahmin uyuşmazlığında planı tamamen iptal et ve önce modeli düzelt" kuralı (schema'nın "Reality outranks the model" ilkesi) ve "durum temsilini değil sadece kuralı değil, ikisini birlikte sorgulayabilme" (WA30 örneğindeki "carry-state" temsil-değişikliği) kavramsal olarak net ve bağımsız biçimde test edilebilir öneriler. 5. sırada çünkü bu fikrin kaynağı (schema-harness) bu raporun en az güvenilir kaynağı — mimariyi benimsemek, sayısal iddiayı benimsemekten tamamen ayrı tutulmalı.

---

## Kaynak listesi (bu raporda atıfta bulunulan)

- `docs.arcprize.org/methodology`, `docs.arcprize.org/changelog`, `docs.arcprize.org/llms.txt`, `docs.arcprize.org/swarms`, `docs.arcprize.org/llm_agents`, `docs.arcprize.org/benchmarking-agent` — birincil kaynak, doğrudan okundu.
- `arcprize.org/media/ARC_AGI_3_Technical_Report.pdf` / arxiv 2603.24621 "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence" — ikincil kaynak (emergentmind.com özeti üzerinden), doğrudan PDF okunmadı.
- Kaggle `arc-prize-2026-arc-agi-3/discussion` — discussion/727629 (schema-harness tartışması), /717133 (Tufa Labs Duck), /728278 (donanım/compute tartışması), /727119 (500 submission analizi), /729985 (skorlama netleştirmeleri), /730225 (düşük-oy alan MDL gönderisi) — birincil kaynak, tarayıcıyla gerçek yorumlar okundu.
- arxiv 2603.13372 "The ARC of Progress towards AGI: A Living Survey of Abstraction and Reasoning" (Vahdati, Aioanei, Suresh, Lehmann, Mart 2026) — birincil kaynak (HTML tam metin), kısmi/filtrelenmiş okuma.
- arxiv 2605.25931 "Explore Before You Solve: The Speed–Depth Trade-off in Epistemic Agents for ARC-AGI-3" (Liew Keong Han, Mayıs 2026) — birincil kaynak (PDF), kısmi okuma (matematik bölümü tam ayrıştırılamadı).
- `schema-harness.github.io` "[schema]: Frontier Models with the Right Harness Achieve ~99% on ARC-AGI-3 Public" (Zeng, Wang, Ma, Yin, Wang, Liu, Kanazawa, Ni, Li, Zanette, Feng; Impossible Research, 2026) — birincil kaynak, tarayıcıyla tam metin okundu. **Skor iddiası doğrulanmamış — sayfanın kendisi bunu itiraf ediyor.**
- `github.com/Erikiss/Rebuild-schema-harness-by-impossible-research` — birincil kaynak (README), gayrı-resmi "clean-room" yeniden inşası.
- `github.com/impossible-research` — birincil kaynak, tek depo, Schema ile görünür bağlantı yok.
- arxiv 2602.02710 "Maximum Likelihood Reinforcement Learning" (Tajwar, Zeng, Zhou, Song, Arora, Jiang, Schneider, Salakhutdinov, Feng, Zanette) — ikincil kaynak (özet üzerinden), Schema yazarlarının CMU/Berkeley bağlantısını çapraz doğrulamak için kullanıldı.
- Michael Hodel `arc-dsl`, Ryan Greenblatt'in GPT-4o program-sentezi yaklaşımı — ikincil kaynak (arama özetleri), genel bilinen sonuçlarla tutarlı.
