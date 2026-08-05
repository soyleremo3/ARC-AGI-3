Bu bağlamda en başarılı ve öne çıkan açık kaynak projeler şunlar — hepsinde ortak nokta, LLM'i doğrudan cevap üreten bir kutu olarak değil, kod yazıp test eden bir "bilim insanı" gibi kullanmaları:

1. Sergey Rodionov / SingularityNET — Executable World Models
📎 https://github.com/astroseger/arc-3-agents-baseline1
Bu, sonraki tüm "kod-tabanlı dünya modeli" yaklaşımlarının temelini oluşturan çalışma. Ajan çalıştırılabilir bir Python dünya modeli tutuyor, bunu önceki gözlemlere karşı doğruluyor, MDL benzeri bir sadelik önyargısının pratik bir vekili olarak daha basit soyutlamalara doğru yeniden düzenliyor ve harekete geçmeden önce bu model üzerinden planlama yapıyor. GPT-5.5 ile yüksek muhakeme çabasıyla ajan 15 oyunu tam olarak çözdü ve ortalama oyun başına %58.12 RHAE elde etti. Farklılaştırıcı özelliği: oyuna özel hiçbir kod içermiyor, yani genel-amaçlı bir taban çizgisi (baseline) olarak hizmet ediyor — ve AGI 2026'da kabul edildi. 
arXiv
Springer

2. NIMI — Tycho
📎 https://github.com/NIMI-research/Tycho
Tycho, her oyun ortamı hakkında yanlışlanabilir hipotezler olarak Python kodu yazıyor, tahminleri gözlemlere karşı doğruluyor ve harekete geçmeden önce simülasyon yoluyla planlama yapıyor. Rodionov'un fikrini genişletiyor: hipotez yönetimine sistematik bir yaklaşım, daha yapılandırılmış aktif keşif stratejisi ve yeniden kullanılabilir bir planlayıcı beceri kütüphanesi ekliyor. Ayrıca ekip, 3 ARC-AGI kuşağı boyunca 82 yaklaşımı analiz eden bir "living survey" yayınlamış olduğundan, metodolojik olarak güvenilir bir grup — bu da başka araştırmacıların üzerine inşa edebileceği, tam olarak yeniden üretilebilir bir sistem olmasını sağlıyor. 
Tech Times

3. Symbolica AI — Agentica SDK / Arcgentica
📎 https://github.com/symbolica-ai/ARC-AGI-3-Agents (branch: symbolica/arcgentica) ve https://github.com/symbolica-ai/arcgentica
Bu proje hızıyla dikkat çekti: ARC-AGI-3'ün genel kullanıma açılmasının ilk günü, 182 seviyeden 113'ünü geçerek ve 25 oyundan 7'sini tamamlayarak %36.08 skor elde etti. Ayrıca Opus 4.6'nın %0.25 skor için 8.900 dolar harcamasına karşı, Agentica aynı işi 1.005 dolara yaptı. Farklılaştırıcı özelliği: modelin kendini özyinelemeli (recursive) olarak çağırmasına izin veren bir orkestrasyon sistemi kullanması — Recursive Language Models'e benzer şekilde, bir oyunu çözmek için alt-ajanlara bölünüyor, yani klasik "tek ajan tek context" sınırını aşıyor. 
Symbolica

4. RGB Agent (OpenCode tabanlı)
📎 https://github.com/alexisfox7/RGB-Agent
Bu ajan, 3 önizleme oyununda insan-düzeyi aksiyon verimliliğine yakın performans gösteren ilk sistem oldu ve üç önizleme oyununu, halka açık şekilde bildirilen en düşük aksiyon sayısıyla (1.069 aksiyon) tamamladı. Genel amaçlı bir kodlama ajanı iskeletini (OpenCode) ARC'a özel bir harness'a sarmak yerine kullanması, onu diğer ARC-özel çözümlerden ayırıyor — yani "domaine özel bir araç seti kurmadan da genel bir coding-agent ile ciddi verimlilik elde edilebilir" tezini kanıtlıyor. 
Tufa Labs
GitHub

5. Tufa Labs — Duck Harness
📎 https://github.com/Tufalabs/duck-harness
Kaggle'daki ARC Prize 2026'nın ilk milestone ödülünü kazanan çözüm. Diğerlerinden en büyük farkı: dev, kapalı bir frontier modele değil, yerelde çalıştırılan açık ağırlıklı Qwen 3.6 27B modeline dayanıyor ve kazanan üç çözümden "ajan kendi kodunu yazıyor" yaklaşımını kullanan tek proje. İlginç bir bulgu da paylaşmışlar: ekip, el yapımı (hand-crafted) araçların modelin doğaçlama becerilerini engellediğini fark etti — yani harness'ı bilinçli olarak hafif ve jenerik tuttular, modelin sürmesine izin verdiler. 
ARC Prize

Ekosistemin omurgası (bunlar tek başına "başarılı bir çözüm" değil ama diğer projelerin hepsi bunun üzerine inşa ediliyor):

ARC Prize'ın resmi ajan iskeleti: https://github.com/arcprize/ARC-AGI-3-Agents
Topluluk vitrini (kendi ajanını göstermek isteyenler için): https://github.com/arcprize/ARC-AGI-Community-Leaderboard — bu bir sıralama değil, sadece genel-amaçlı, yeni ve açık yöntemleri öne çıkaran küratörlü bir vitrin. 
GitHub

Bir not: "[schema]" harness'ı (Impossible Research, Berkeley/CMU ekibi) kamuya açık test setinde kendi bildirdiği rakamla %99'a kadar çıktığını iddia ediyor, ancak bu skor ARC Prize tarafından doğrulanmadı ve harness'ın kendisi (sadece traces/veri seti değil) tam olarak açık kaynak yayınlanmış görünmüyor — bu yüzden yukarıdaki listeye dahil etmedim, ama merak ederseniz detaylarına girebilirim.

Ortak başarı deseni: Yukarıdaki projelerin hepsinin paylaştığı temel fikir aynı — LLM'in kendisini "oyunu bilen" bir kara kutu olarak kullanmak yerine, LLM'e çalıştırılabilir, test edilebilir bir kod modeli yazdırmak, bunu gerçek gözlemlere karşı doğrulamak (verifier), ve gerçek aksiyon harcamadan bu model üzerinde plan aramak (simülasyon/BFS). RHAE'nin aksiyon sayısını karesiyle cezalandırması, bu "önce anla, sonra verimli hareket et" stratejisini doğrudan ödüllendiriyor — ki bu da ham model ölçeğinin neden tek başına yetmediğini açıklıyor.