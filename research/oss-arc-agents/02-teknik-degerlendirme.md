# ARC-AGI-3 Açık Kaynak Agent Projeleri — Teknik Değerlendirme (Agent 2)

Bu rapor, 5 ARC-AGI-3 (ARC Prize 2026) agent projesinin kaynak kodu düzeyinde derinlemesine incelemesidir. Agent 1'in düz-dilde yorumundan farklı olarak burada mimari, algoritma detayı, kod kalitesi ve sınırlamalar ele alınıyor; okuyucunun projelerin ne iddia ettiğini zaten bildiği, nasıl inşa edildiğini merak ettiği varsayılıyor.

Tüm repolar `git clone --depth 1` (gerekli durumlarda ilgili branch ile) kullanılarak `C:\Users\EMRULL~1\AppData\Local\Temp\claude\oss-arc-research-tech\` altına klonlandı ve doğrudan kaynak kod okunarak incelendi. Beş repo da başarıyla klonlanabildi; hiçbiri erişilemez değildi. Aşağıdaki her bölümde gerçek dosya yolları ve fonksiyon/sınıf adları referans veriliyor.

Ortak paradigma hatırlatması: LLM, gerçek gözlemlere karşı doğrulanabilir, çalıştırılabilir bir Python "world model" / hipotez yazıyor, gerçek oyun aksiyonu harcamadan önce bu model üzerinde planlama/arama yapıyor.

---

## 1. astroseger/arc-3-agents-baseline1 (Rodionov / SingularityNET)

### 1.1 Repo yapısı ve hangi varyantın incelendiği

Repo aslında bir "proje sayfası": `papers/paper01/` ve `papers/paper02/` altında iki makaleye ait çok sayıda near-duplicate agent varyantı barındırıyor (`ewma_s_v1.2`, `ewma_sv_v1.2`, `ewma_sv_v1.5`, `ewma_sv_v1.6`, `ewma_v1.2`, `twma_v1.2`, `twma_v1.6` vb., `papers/paper02/agents/README.md`'deki ablasyon merdiveninde: `twma_v1.2`→`ewma_v1.2`→`ewma_s_v1.2`→`ewma_sv_v1.2`→`ewma_sv_v1.5`→{`twma_v1.6`, `ewma_sv_v1.6`}). Kök `README.md`, en güçlü sonuçları veren `papers/paper02/agents/ewma_sv_v1.6` ("executable world model") ile daha ucuz alternatif `papers/paper02/agents/twma_v1.6`'yı ("textual world model, replay-verification yok") önerdiği için inceleme bu ikisine odaklandı.

### 1.2 Mimari (ewma_sv_v1.6)

**Orkestrasyon katmanı** (Python, framework tarafından yazılmış, LLM'in dokunmadığı kısım):
- `run_controller.py` / `run_funs.py` — Docker ağları/konteynerleri kurar, her oyunu rastgele bir UUID takma ada eşler (`build_game_id_mapping`), Codex hesabı başına bir agent konteyneri başlatır.
- `src/agent/agent_runner.py` — konteyner içi süpervizör: `agent.py`'yi master modda başlatır, beklenmeyen çıkış veya 30 dakikalık log durgunluğunda (`RECOVERY_LOG_STALE_SECONDS`) `--recovery` modunda yeniden başlatır (en fazla `MAX_RECOVERY_ATTEMPTS = 10` kez).
- `src/agent/agent.py` (sınıf `Agent`) — asıl kontrol döngüsü. `iteration_loop`, disk üzerindeki oturum durumunu (`session_inspector.py::SessionInspection`) her turda yeniden okuyup dört "protokol"den birini seçiyor: `normal_continuation_protocol`, `normal_reset_protocol`, `trouble_protocol1`, `stuck_protocol`.
- `src/agent/codex_runner.py` (`CodexRunner`) — OpenAI Codex CLI'yi subprocess olarak sarar (`codex -m <model> -c model_reasoning_effort=<effort> --dangerously-bypass-approvals-and-sandbox exec ...`, devam turlarında `codex ... resume <thread_id>`).

**Oyunu-çözme katmanı** (çalışma zamanında Codex tarafından `agent_run/` içine yazılan/düzenlenen kısım, `workspace_init/`'ten tohumlanır):
- `world_model_engine.py::world_model_engine(state, action) -> (new_state, status)` — **boş placeholder**, Codex tarafından doldurulur.
- `world_model_state_io.py::initial_state_reconstruction`, `state_renderer`, `apply_render_overrides` — render/state-reconstruction stub'ları.
- `world_model_main_planner.py::planner(state) -> list[dict] | None` — planlayıcı stub'ı (tek satır: `return None`).
- `verify_world_model.py`, `verify_main_planner.py`, `plan_executor.py`, `state_reconstruction_tools.py`, `mismatch_artifacts.py`, `session_tools.py` — **framework tarafından hazır verilen** (LLM'in yazmadığı) doğrulama/yardımcı kod.

Bu ayrım kritik: "world model" bir framework veri yapısı değil, tamamen LLM'in çalışma zamanında yazdığı, tek zorunlu anahtarı `"level"` olan bir Python `dict`'tir. Framework yalnızca fonksiyon imzalarını sabitler (`AGENT.md`'deki "fixed-interface" iddiasının karşılığı budur — arayüz sabit, implementasyon %100 LLM üretimi).

### 1.3 Çekirdek algoritma

**Güncelleme döngüsü:** Her Codex turu tek bir `codex exec` çağrısıdır. `Agent.iteration_loop`, prompt seçimini disk durumuna göre yapar; dünya modelinin dosyaları Codex'in kendi shell/dosya araç çağrılarıyla mutasyona uğrar — Python orkestratör `world_model_engine.py`'yi hiç doğrudan düzenlemez.

**Tam-replay doğrulama** (`verify_world_model.py`): Her seviye için kayıtlı gerçek denemeler (`session_tools.read_all_attempts_for_level`) tek tek yeniden oynatılır: `_replay_attempt`, `initial_state_reconstruction` ile başlangıç durumunu kurar, `state_renderer` ile render eder ve `np.array_equal` ile kayıtlı `initial_frame`'e karşı **piksel-hassasiyetinde** karşılaştırır; ardından her aksiyonu `world_model_engine` üzerinden tahmin edip durum (`RUNNING`/`LEVEL_COMPLETED`/`GAME_OVER`) ve render'ı gerçek kayıtla karşılaştırır. Herhangi bir uyuşmazlık `VerificationMismatchError` fırlatır ve `mismatch_artifacts.py` diagnostik PNG/ASCII fark görselleri üretir. Tolerans yoktur — tam piksel eşitliği aranır. `plan_executor.py` ayrıca gerçek zamanlı yürütmede de aynı çapraz kontrolü yapar: önerilen bir aksiyon dizisini gerçek client üzerinde çalıştırırken her adımda world model tahminiyle gerçek gözlemi karşılaştırır, ilk sapmada `AssertionError` ile durur.

**Planlama/arama algoritması:** Framework içinde **hiçbir arama algoritması yoktur** (BFS/DFS/A* implementasyonu yok). `world_model_main_planner.py` bir satırlık stub'dır; `prompts/world_model_planner.txt` yalnızca düzyazıyla "çoğu durumda basit bir arama/planlama algoritması yeterli olmalı" diye talimat verir. Derinlik, dallanma faktörü, algoritma seçimi tamamen Codex'in o çalıştırmada ne yazdığına bağlıdır ve bu repoda denetlenebilir/sabit değildir.

### 1.4 MDL/basitlik-önyargısı sorusu: tamamen prompted, kodda metrik yok

`ewma_sv_v1.6` içinde `MDL|description.length|complexity|simplicity|occam|minimum.description` için yapılan repo-geneli grep, **kod düzeyinde sıfır sonuç** verdi — yalnızca prompt dosyalarında düzyazı bulundu. Hiçbir AST-boyut kontrolü, cyclomatic-complexity eşiği, satır-sayısı limiti veya sıkıştırma-oranı hesaplaması yok. Basitleştirme mekanizması, `Agent.send_simplification_prompts` (`agent.py:225-234`) tarafından zamanlanmış doğal-dil promptlarıdır: seviye 1 / erken seviye 2'de `light_simplification_level1.txt`, aksi halde dört promptluk grup (`hard_refactoring_pgroup` = `world_model_simplification_step1/2/3.txt` + `world_model_planner.txt`). Temsili metin: *"Your job is to compress the ontology and dynamics, not preserve them—to find the simplest possible world model that explains the previous observations"* (`world_model_simplification_step1.txt:4`); *"Assume the real mechanics are simpler than your current explanation unless the evidence clearly forces complexity"* (`world_model_simplification_step2.txt:13`). Yani basitlik baskısı tamamen talimat-düzeyinde, LLM'in kendi öznel yargısıyla değerlendiriliyor; tek nesnel kontrol, (muhtemelen) basitleştirilmiş modelin hâlâ tam-replay doğrulamasını geçmesi. "MDL-benzeri basitlik önyargısı" iddiası için hesaplanan bir metrik yoktur — bu önemli bir sınırlamadır.

### 1.5 "Oyuna özgü kod yok" iddiası: koda dayalı olarak doğrulandı

- `run_config.yaml`'daki 25 kamu oyunu ID'si (`ar25, bp35, cd82, ...`) için `ewma_sv_v1.6/src` içinde grep **sıfır eşleşme** verdi.
- `run_funs.py::build_game_id_mapping` agent konteynerine yalnızca rastgele bir UUID takma ad verir; `server.py::start_game` gerçek `game_id`'yi yalnızca sunucu tarafında çözer ve tekrar kullanılmasın diye mapping'den siler.
- `Dockerfile.proxy` — Squid proxy yalnızca `.chatgpt.com .openai.com`'a izin verir (port 443), bu yapısal olarak agent'ın internete erişip oyun bilgisi sızdırmasını/çekmesini engeller.
- Tüm prompt dosyaları jenerik ARC-AGI-3 diliyle yazılmış, oyun ismi/seviye ipucu içermiyor.
- Tek istisna: `apply_render_overrides` — "verification-only escape hatch", prompt'ta "geçici, dar tutulmalı ve gerçek mekaniğin hâlâ eksik olduğunun kanıtı" olarak işaretlenmiş; `verify_world_model.py::_render_frame` her seviyede ilk kullanımda uyarı basar ama kodda bunu engelleyen bir kontrol yok — dürüstlük mekanizması zorunlu değil, kendiliğinden bildirimli.

### 1.6 twma_v1.6 karşılaştırması

`twma_v1.6/src/agent/workspace_init/` içinde yalnızca `client/` var — `world_model_engine.py`, `verify_world_model.py`, `plan_executor.py` vb. **hiç yok**. `agent.py`'de `hard_refactoring_pgroup` kurulumu comment-out edilmiş, `send_simplification_prompts` `pass`'e indirgenmiş. `twma_v1.6/AGENT.md`: *"Scheduled simplification and exact replay verification are disabled."* İki varyant aynı Codex CLI sürümü, aynı model (`gpt-5.6-sol`), aynı reasoning effort (`max`) ve aynı oyun listesini kullanıyor — yani "çalıştırılabilir model + replay doğrulaması" tek yapısal fark olacak şekilde izole edilmiş bir ablasyon tasarımı.

### 1.7 Kod kalitesi

- Orkestrasyon kodu (`agent.py`, `session_inspector.py`, `codex_runner.py`) için **otomatik birim testi yok**; `workspace_init/test_timeout.py` bir pytest suite değil, çalışma-zamanı smoke-check'i (ve LLM'in kendi çalışma alanı için, framework CI'ının değil).
- Config düz bir YAML (`run_config.yaml`: `codex_accounts`, `games`, `tag`, `reasoning_effort`, `model`); şema doğrulaması yok.
- Sabit sayılar koda gömülü: `MAX_LEVEL_ACTIONS = 1500` (`client.py`, `agent.py::stop_condition` içinde tekrarlanmış), `RECOVERY_LOG_STALE_SECONDS = 30*60`, `MAX_RECOVERY_ATTEMPTS = 10`, doğrulama `TIMEOUT_SECONDS = 180` (üç ayrı dosyada).
- Savunmacı tasarım örneği: `client.py::client_lock` içinde `fcntl.flock` ile oturum yazımı korunuyor; `repair_if_needed` mekanizması önceki bir `client.py` çağrısının aksiyon gönderip artefakt yazamadan çöktüğü durumları algılayıp sunucunun `/game/last-step` uç noktasından onarıyor.

### 1.8 Güçlü/zayıf yönler ve sınırlamalar

**Güçlü:** Tam-replay doğrulayıcı gerçek, oynanamaz bir doğruluk kapısı sağlıyor; UUID-takma-ad + proxy-allowlist tasarımı oyun-genelliğini politika değil yapısal olarak zorluyor; supervisor/recovery katmanı (`agent_runner.py`) durum tamamen `session/` dosyalarında tutulduğu için Codex/konteyner çökmelerine dayanıklı.

**Zayıf:** Planlayıcı sınırsız — derinlik/dallanma faktörü kodda hiçbir yerde zorlanmıyor; "basitlik önyargısı" öznel/prompted, azaltımın gerçekten gerçekleştiğine dair kod kontrolü yok; `apply_render_overrides` kabul edilmiş bir açık kapı.

**Sınırlamalar:** Oyun başına tek deneme değerlendirmesi (yeniden başlatma/geri dönme yok); README'ye göre tam 25-oyunluk `v1.6` koşusu için "kabaca 4-5 ChatGPT Pro aboneliği" gerekiyor — yüksek maliyet; framework, üretilen planlayıcının verimli olup olmadığını kendi başına teyit edemez (yalnızca dış Zenodo arşivinden incelenebilir).

### 1.9 Compute/bağımlılık/çalışma-zamanı varsayımları

LLM: OpenAI Codex CLI, sabitlenmiş sürüm `0.144.1`; varsayılan model `gpt-5.6-sol`, `reasoning_effort: max`. Codex `--dangerously-bypass-approvals-and-sandbox` ile **sandbox'sız** çalıştırılıyor — izolasyon tamamen dış Docker konteyneri + ağ proxy allowlist'i tarafından sağlanıyor. Kimlik doğrulama: önceden kimlik doğrulanmış Codex hesap dizinleri veya `OPENAI_API_KEY` (ikincisi README'de önbellek isabet oranı düşük olduğu için çok daha pahalı diye açıkça uyarılmış). Üç-konteynerli Docker topolojisi: agent + ayrı sunucu (arc_agi/arcengine) + Squid egress-proxy. Boyutlandırma önerisi: oyun başına ~1-2 çekirdek / ~3 GB RAM; tam 25-oyunluk paralel koşu için ~48 çekirdek / 64 GB RAM.

---

## 2. NIMI-research/Tycho

### 2.1 Mimari

Tycho, `docs/ARCHITECTURE.md`'nin kendi tanımıyla "skor etkileyen politikayı model taşımasından ve çalıştırma operasyonlarından ayırır." Paket yapısı:

- `tycho/agent/agent.py` (2492 satır) — `TychoAgent`: "aktör"; oyun başına büyüyen tek bir tool-use konuşması (`ls`/`read_file`/`write_file`/`edit_file`/`run_python`/`take_action` gibi doğal LLM araçları üzerinden); seviye geçişleri, geçmiş tahliyesi, bütçe kontrolleri ve mod dispatch'inin sahibi.
- `tycho/agent/builder.py` — `WorldModelBuilder`: yalnızca `world_model.py`'yi yazma/iyileştirme görevi olan, sınırlı ve ayrı-konuşmalı bir subagent; danışma-metni raporu döndürür.
- `tycho/agent/modes.py` — `ModeSpec`/`MODES` kaydı: dört politikanın (`no_world_model`, `single`, `orchestrator`, `trigger`) tek doğruluk kaynağı.
- `tycho/agent/dispatcher.py` — `TriggerDispatcher`: `trigger` modunda harness'ın builder'ı ne zaman otomatik ateşleyeceğine karar verir.
- `tycho/agent/wm_signal.py` — `wm_signal()`: sandbox'lanmış, ucuz bir prob; `dynamics_inaccurate`/`outcome_divergence` döndürür.
- `tycho/prompts/*.j2` + `render.py` — Jinja2 şablonları; `actor.system.j2`, `wm_variant`'a göre `partials/wm_{single,orch,trigger,none}.j2`'den birini `{% include %}` eder.
- `tycho/workspace/workspace.py` (`GameWorkspace`), `agent_tools.py` (`ToolExecutor`/`tool_specs()`), `sandbox.py` (`PythonSandbox`), `wmlib_template.py` (1774 satır, `wmlib.py` olarak her workspace'e tohumlanan sabit araç kutusu), `version_store.py` (`WorkspaceVersionStore`, SHA-256 içerik-adresli tam snapshot).
- `tycho/harness/harness.py`, `inference_budget.py`, `model_replay.py`, `planner_follow_diagnostics.py`, `resume.py`, `run.py`, `run_parallel.py`, `scoring.py`, `submission_replay.py` — ARC motoruyla etkileşim, bütçe zorlaması, tam-resume, çok-süreçli orkestrasyon, scorecard replay.
- `tycho/serving/llm_client.py`, `public_backends.py`, `pricing.py` — sağlayıcı-nötr LLM istemcisi.
- `tycho/viewer/*` — yerel replay/inceleme sunucusu; skorlamayı **etkilemez**.

### 2.2 Hipotez yönetimi: veri yapısı, yaşam döngüsü

Ayrı bir "Hypothesis" sınıfı veya durum makinesi **yoktur**. Hipotezin kendisi, `tycho/workspace/templates/seed_world_model.py.tmpl`'den tohumlanan, sabit bir sözleşmeye sahip gerçek Python modülü `world_model.py`'dir: `@dataclass State`, `init_state(grid0, level)`, `transition(state, action)`, `render(state)`, `outcome(state)` (kesinlikle `"ongoing"|"level_complete"|"game_over"` döndürmeli, `wmlib.outcome()` tarafından zorlanır), artı isteğe bağlı `actions`, `subgoals`, `heuristic`, `observation_variants`, `planner_key`.

- **Öner (propose):** Aktör (`single` modda) veya `WorldModelBuilder.build()` (`orchestrator`/`trigger`), `write_file`/`edit_file`/`edit_function` ile `world_model.py`'yi yazar/düzenler.
- **Test et (yanlışla):** `wmlib.verify(model)` (`tycho/workspace/wmlib_template.py:567`) her gözlemlenen geçişi `init_state`→`transition`→`render` ile tekrar oynatıp `simulation_accuracy`, `known_cell_accuracy`, `prediction_coverage`, `first_divergence` puanlar. `wmlib.verify_outcome(model)` (satır 1071) ayrıca `outcome()`'u gözlenen level-complete/GAME_OVER terminalleri ve bir render-köprüsü kontrolüyle (modellenen kazanma durumu, gözlenen kazanma karesini render etmeli) yanlışlar. `ToolExecutor._wm_feedback()` (`agent_tools.py:324`) bunu her `world_model.py` düzenlemesinden sonra otomatik çalıştırır.
- **Onayla/reddet sinyali:** `tycho/agent/wm_signal.py::wm_signal()` sandbox'lanmış birleşik bir prob çalıştırır, `{dynamics_inaccurate, outcome_divergence}` döndürür; `TriggerDispatcher.should_fire()` (`dispatcher.py`) bu koşulu bir modeli reddedip yeniden inşayı zorlamak için kullanır — **rate-limit yok**; modülün kendi docstring'i: "the natural circuit breaker is the game's overall LLM budget."
- **Versiyonlama:** `WorkspaceVersionStore.capture()` (`version_store.py`) her turda tüm "nedensel" workspace dosyalarının (world_model.py, notlar, yardımcı modüller) SHA-256 içerik-adresli tam anlık görüntüsünü alır (`SNAPSHOT_SCHEMA = 2`) — içerik-adresli bir denetim günlüğü, ama bir yaşam-döngüsü/durum-makinesi nesnesi değil (`Hypothesis.status` enum'u yok).

### 2.3 Aktif keşif stratejisi

Ayrı bir keşif algoritması (UCB/novelty modülü) **yok**. "Yapılandırılmış aktif keşif", yanlışlama araçlarıyla desteklenmiş prompt-düzeyinde bir politikadır: `actor.system.j2`/mod partial'ları, "güvenilir bir model/plan yoksa, rekabet eden mekanik/hedef hipotezlerini ayırt edecek aksiyonları seç" diye talimat verir (`wm_orch.j2`, `wm_trigger.j2`, `wm_single.j2`, `wm_none.j2` hepsi bunu tekrarlar). Builder'ın rapor formatı bir `recommended_action` + `subgoal` ister; doğrulama-durumu probu (`_VERIFY_STATE_PROBE`, `builder.py`) her builder geçişine tam sapan geçişin diff'lenmiş kanıtını ve PNG'lerini otomatik ekler. Ayrı bir arama/bandit süreci **yoktur** — "yapılandırılmış" ifadesi algoritmik bir keşif politikasına değil, yanlışlama-güdümlü prompt iskeletine ve otomatik enjekte edilen sapma kanıtına atıfta bulunuyor.

### 2.4 "Yeniden kullanılabilir planlayıcı beceri kütüphanesi"

Repo genelinde `skill`/`reusable`/`library of` için grep, böyle adlandırılmış bir kavram bulmuyor. Buna en yakın gerçek mekanizma, her oyun workspace'ine `wmlib.py` olarak materyalize edilen (`tycho/workspace/wm_templates.py::_load_template` üzerinden) **sabit, elle yazılmış** bir araç kutusudur (`tycho/workspace/wmlib_template.py`, 1774 satır); `world_model.py`/`verify.py`/`plan.py` tarafından import edilir:
- Planlama primitifleri: `plan_bfs`/`plan_bfs_with_diagnostics` (satır 1543-1587, BFS, `max_nodes=20000` varsayılan, `prune_game_over=True`), `plan_astar`/`plan_astar_with_diagnostics` (satır 1589-1640, sezgi fonksiyonuyla A*, varsayılan sezgi `lambda _s: 0` = BFS'ye eşdeğer), `plan_subgoals`, `validate_plan` (kanonik-replay kabul kapısı).
- Algılama primitifleri: `segment`/`segment_summary` (4-bağlantılı bileşenler, kompakt row-run şekil kodlaması), `diff_text` (kayıpsız run-length hücre-delta özeti).
- Kanıt erişimcileri: `frames()`, `transitions()`, `attempts()`, `death_events()`, `terminal_events()`, `animation_index()`/`animation_grids()`.
- Yanlışlama: `verify()`, `verify_outcome()`, `game_over_action_report()`.

Ayrıca `tycho/workspace/agent_tools.py`, aktöre `write_file` ile "yeniden kullanılabilir Python yardımcı modülleri" oluşturup sonraki çağrılarda import edebileceğini açıkça söylüyor — ama **oyunlar arası kalıcılık mekanizması yoktur**; her oyun taze bir workspace ve aynı statik `wmlib.py`'nin taze bir kopyasıyla başlar. Yani "yeniden kullanım" (a) her oyunda aynı jenerik kütüphane ve (b) oyun-içi yardımcı dosyalar anlamına gelir — çalıştırmalar arasında büyüyen, küratörlü bir beceri kütüphanesi değildir.

### 2.5 "82 yaklaşımın canlı taraması"

Bu repoda **bulunamadı**. `README.md`, `docs/ARCHITECTURE.md`, `docs/PAPER_RESULTS.md`, `docs/REPRODUCING.md`, `CITATION.cff` kontrol edildi; hiçbiri böyle bir taramadan bahsetmiyor, repo-geneli "82"/"survey" grep'i de ilgisiz bir eşleşme (`cd82` oyun ID'si alt dizesi) dışında bir şey vermedi. Bu iddia karşılaştırma setindeki başka bir repoya ait olmalı, Tycho'ya değil.

### 2.6 Dört operasyon modu

Tek doğruluk kaynağı `tycho/agent/modes.py::MODES`. Dört kağıt politikası (`configs/paper/opus48_{no_world_model,single,orchestrator,trigger}.yaml`) neredeyse byte-özdeş konfigürasyonlardır (`max_llm_calls=3500`, `max_inference_cost_per_game=750`), yalnızca `orchestration.mode` farklıdır:

| Mod | Prompt partial | `world_model.py`'yi kim yazar | Builder tetikleyicisi |
|---|---|---|---|
| `no_world_model` | `wm_none.j2` | kimse — doğrudan akıl yürütme | yok |
| `single` | `wm_single.j2` | aktörün kendisi | yok |
| `orchestrator` | `wm_orch.j2` | `WorldModelBuilder`, aktör `invoke_builder` aracıyla çeker | aktör karar verir |
| `trigger` | `wm_trigger.j2` | `WorldModelBuilder`, harness iter | `TriggerDispatcher.should_fire()` — `wm_signal()` sapmasında, seviye başlangıcında ve GAME_OVER resetinden sonra otomatik ateşler |

Sonuçlar (RHAE, `docs/PAPER_RESULTS.md`): no-model 79.07, single 85.36, orchestrator 88.49, trigger 83.07 (Opus 4.8); orchestrator GPT-5.6-Sol ve Opus 5 ile 100.00'e ulaşıyor.

### 2.7 Sandbox / izolasyon

`tycho/workspace/sandbox.py::PythonSandbox` — agent tarafından üretilen tüm Python, Docker veya Finch üzerinden taze bir konteynerde çalışır: `--network none`, `--read-only` kök dosya sistemi, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--pids-limit 64`, `--memory 1g --cpus 1`, ulimit sınırları. `PythonSandbox.check()`, izolasyonun gerçekten uygulandığını (`cap_eff=0000000000000000`, `no_new_privs=1`, host dosyası okunamaz, ağ yok) çalışma zamanında doğrulayan bir "doctor" komutu içerir — yalnızca bayraklara güvenmek yerine.

### 2.8 Kod kalitesi

- **Test kapsamı:** 35 dosya, ~5 alt paket. Planlayıcı sözleşme davranışı (`tests/workspace/test_wmlib_planning.py`), segmentasyon şekil kodlaması (`test_wmlib_segmentation.py`), tam içerik-adresli snapshot/restore (`test_workspace_versioning.py`), planlayıcı-takip diagnostiği (`test_planner_follow_diagnostics.py`), fiyatlandırma (`test_pricing.py`), prompt-sözleşme testleri. `wm_signal` probu için test yalnızca geçerli Python olarak derlendiğini kontrol ediyor (`test_wm_signal.py`) — davranışsal bir test değil, sığ bir smoke test.
- **CI:** `.github/workflows/tests.yml`, `make validate`'i (test suite + config çözümleme + bütünlük hash'leri + secret-leakage taraması + wheel build) push/PR'da çalıştırır, API çağrısı yapmaz.
- **Hata yönetimi:** Güven sınırlarında (agent-yazımı kod yürütme, tahmin/doğrulama probları) bilinçli `except Exception: # noqa: BLE001` kullanımı, her seferinde yorum içinde gerekçelendirilmiş ("never crash a planner-prep helper").
- **Fiyatlandırma tablosu** (`tycho/serving/pricing.py`) versiyonlanmış manuel bir sabit (`PRICE_SCHEDULE = "public-list-2026-07-13"`) — gelecekteki modeller için manuel güncellenmezse eskiyecek; bilinmeyen model için dolar bütçesi zarifçe değil sert şekilde başarısız olur.

### 2.9 Güçlü/zayıf yönler

**Güçlü:** Yanlışlama gerçekten yük taşıyor (dekoratif değil) — dinamik ve terminal/outcome doğruluğu iki bağımsız kanalda, UNKNOWN-hücre semantiği ve coverage-vacuity korumalarıyla puanlanıyor. Sandbox izolasyonu gerçek ve savunma-derinlikli. Bütçe zorlaması katmanlı: RHAE aksiyon bütçesi (motor düzeyi, `5×insan taban çizgisi`), oyun/seviye başına LLM-çağrı tavanı, isteğe bağlı dolar-cinsinden çıkarım bütçesi.

**Zayıf:** "Yeniden kullanılabilir beceri kütüphanesi" statik ve elle yazılmış, öğrenilmiş/biriktirilmiş değil — çözülmüş bir oyunun world-model içgörüsünü yeni bir oyuna aktaran mekanizma yok. `trigger` modunun sapma probu tasarım gereği rate-limit'siz, tüm bütçeyi yakınsamayan bir modeli yeniden inşa etmeye harcayabilir. Ajan çekirdeği (`agent.py`, ~2500 satır) tek büyük bir modül — paketin geri kalanının aksine merkezi.

### 2.10 Compute/bağımlılık/çalışma-zamanı varsayımları

Üç yerleşik protokol taşıyıcısı: Anthropic Messages, OpenAI Responses, OpenAI-uyumlu Chat Completions; `TYCHO_LLM_PLUGIN` ile genişletilebilir. Sandbox: Docker veya Finch (macOS'ta Finch tercih edilir), `TYCHO_SANDBOX_RUNTIME=host` yalnızca güvenilir yerel geliştirme için (benchmark çalıştırmasında reddedilir). Konteyner imajı `python:3.12.11-slim-bookworm` + sabitlenmiş numpy/pillow/scipy/networkx. RHAE aksiyon bütçesi = seviye başına insan taban çizgisinin 5 katı; varsayılan oyun-başı LLM-çağrı tavanı 1100 (kağıt konfigürasyonlarında 3500'e çıkarılmış); kağıt konfigürasyonları `effort: xhigh`, `max_tokens: 24000`, `max_inference_cost_per_game: 750` USD ile çalışır — README açıkça "uzun süren, stokastik ve pahalı; smoke test olarak kullanmayın" diyor.

---

## 3. symbolica-ai — ARC-AGI-3-Agents (branch `symbolica/arcgentica`) ve arcgentica

### 3.1 İki reponun ilişkisi

**`arcgentica`** (symbolica-ai/arcgentica) klasik ARC-AGI'ye yöneliktir (`data/arc-prize-2024`/`2025`, interaktif ARC-AGI-3 oyun ortamı değil). README: *"Agentica achieves 85.28% on ARC-AGI-2 with Opus 4.6 (120k) High at $6.94/task."* Bu, özyinelemeli-kendi-kendine-çağrı fikrinin **orijinal, minimal implementasyonu**: tek bir `Agent` sınıfı (`arc_agent/agent.py`), LLM'in bir `transform(grid)` Python fonksiyonu yazdığı ve `call_agent(...)` ile hipotezleri paralel keşfetmek üzere alt-agent'lar özyinelemeli olarak çağırabildiği.

**`ARC-AGI-3-Agents-symbolica`** (symbolica-ai/ARC-AGI-3-Agents, branch `symbolica/arcgentica`) resmi arcprize `ARC-AGI-3-Agents` harness'inin symbolica-ai fork'u; `agents/templates/agentica/` altında "Arcgentica" adlı bir agent şablonu ekliyor — aynı fikrin **interaktif-oyun portu**: tek bir `transform` fonksiyonu yazmak yerine, bir orkestratör LLM rol-uzmanlaşmış alt-agent'lar (explorer/theorist/tester/solver) üreterek gerçek zamanlı bir ARC-AGI-3 oyununu oynuyor.

Her iki repo da `from agentica import spawn` şeklinde import edilen dış bir SDK üzerine kurulu (PyPI paketi `symbolica-agentica`, `pyproject.toml`'da `"symbolica-agentica>=0.4.1"`). **Asıl LLM-çağırma primitifi (`spawn()`, `agent.call()`) her iki denetlenen repo dışında yaşıyor**, `symbolica-ai/agentica-server`/`agentica-python-sdk` içinde — ne repo "yeni bir agent modelle nasıl konuşur" mantığını kendisi implemente etmiyor. Bu, teknik değerlendirmenin önemli bir sınırlaması: özyinelemeli çağrının düşük-seviye mekaniği (context penceresi kesme, konuşma devamlılığı) klonlanan repolarda gözlemlenemez.

### 3.2 Mimari

**Repo #1 (`ARC-AGI-3-Agents-symbolica`):**
- `agents/agent.py` — temel `Agent` ABC (upstream, değiştirilmemiş oyun döngüsü).
- `agents/swarm.py` — `Swarm` sınıfı: birden çok `Agent` örneğini (oyun başına bir tane) OS **thread'leri** ile paralel çalıştırır (`Thread(target=a.main)`); bu özyinelemeyle **ilgisizdir** — oyun-düzeyinde paralellik, alt-agent üretimi değil.
- `agents/templates/agentica/agent.py` (467 satır) — `Arcgentica(Agent)` sınıfı; **özyineleme burada yaşıyor**.
- `agents/templates/agentica/model.py` — `ModelConfig` ön ayarları (`OPUS_4_6`: `anthropic/claude-opus-4-6`, `GPT_5_2`: `openai/gpt-5.2`).
- `agents/templates/agentica/prompts.py` — `GAME_REFERENCE` (oyun-bağımsız mekanik rehberi) ve `premise()` (Explore→Hypothesize→Test→Iterate→Solve fazlarını tanımlayan orkestratör sistem promptu).
- `agents/templates/agentica/scope/memories.py` — `Memories`/`Memory`: agent'lar arası paylaşılan bilgi deposu (`add`, `summaries`, `get`, `evict`, LLM-tabanlı `query`).
- `agents/templates/agentica/logging/tracker.py` — `UsageTracker`: yalnızca ham token sayıları (input/output/cached/reasoning), $ dönüşümü yok.

**Repo #2 (`arcgentica`):** `arc_agent/agent.py` — `call_agent()` içeren `Agent` sınıfı, özyineleme primitifinin en saf hali; `arc_agent/prompts.py`, `common.py`, `solve.py` (sandbox'ta `transform` kodu çalıştırma), `score.py`/`submit.py`/`summary.py` (pass@2 skorlama, Kaggle gönderim formatı, **her iki repodaki tek maliyet-muhasebesi kodu**).

### 3.3 Özyinelemeli kendi-kendine-çağrı mekanizması (yeniden implemente edilebilir detay)

**Üretme mekanizması.** Her iki repoda da üretim `await spawn(...)` — asenkron bir çağrı, subprocess veya thread değil. Kanıt: README ayrı bir `agentica-server` süreci çalıştırmayı ve agent'ı `S_M_BASE_URL=http://localhost:2345` ile ona yönlendirmeyi öğretiyor; akış chunk-tabanlı (`agentica.common.Chunk`, `WsLogger.on_chunk`'ta tüketiliyor) — bir SSE/streaming LLM API çağrısıyla tutarlı. Eşzamanlılık `asyncio` iledir (`asyncio.gather(call_agent(...), ...)`), OS thread'leriyle değil.

Repo #1'in tam fonksiyonu — `Arcgentica.spawn_agent` (`agents/templates/agentica/agent.py:340-364`):
```python
async def spawn_agent(self, system_prompt: str | None = None):
    return await spawn(
        model=self.model.subagent_model,
        premise=system_prompt,
        reasoning_effort=self.model.reasoning_effort,
        listener=self._make_listener(),
        scope={
            "spawn_agent": self.spawn_agent,   # özyineleme: aynı bağlı metod tekrar enjekte ediliyor
            "numpy": np, "np": np,
            "Memories": Memories, "Memory": Memory,
        },
    )
```
Orkestratörün kendisi de `_run()` içinde (satır 403) aynı şekilde, kendi kapsamında `spawn_agent` ile oluşturuluyor. Döndürülen "agent handle" `await agent.call(return_type, task, **objects)` ile tekrar tekrar çağrılabiliyor. LLM'in kendi ürettiği Python kodu, verilen REPL içinden `spawn_agent(...)`'i tekrar çağırıyor — yani **özyineleme derinliği kodda sınırsız**, tamamen LLM'in tercihine bağlı.

Repo #2'nin tam fonksiyonu — `Agent.call_agent` (`arc_agent/agent.py:60-130`), aynı desen ama toplam üretilen agent sayısına açık sabit bir tavan var (`self.max_num_agents`, varsayılan 10, CLI `--max-num-agents`), tüm özyineleme ağacında **düz** paylaşılıyor (dal-başına değil).

**Bağlam/durum aktarımı.** Hiçbir şey örtük olarak aktarılmıyor — bir alt-agent'ın ihtiyaç duyduğu her nesne (frame'ler, `submit_action`/`bounded_submit_action`, `memories`, `GAME_REFERENCE`, geçmiş) `.call()`'a açık `**objects` kwarg'ları olarak veriliyor. IDEA.md: *"A new subagent knows NOTHING about the game. Its only context are the prompts you gave it."* Tek örtük kanal paylaşılan `Memories` nesnesi — referansla geçirilen mutable bir Python nesnesi, tüm agent'lar arasında yeniden iletim gerektirmeden görünür.

**Sonuç birleştirme.** Açık bir "merge" adımı **yok** — `await agent.call(return_type, task, ...)` yapılandırılmış, tipli bir nesneyi (repo #1'de `FinishStatus`, repo #2'de `FinalSolution`) doğrudan çağırana döndürür, bu sıradan Python dönüş-değeri yayılımıdır. Orkestratör/ebeveyn bir alt-agent'ın raporuyla ne yapacağına tamamen prompt talimatları üzerinden (`premise()` fazları) karar verir, algoritmik bir toplama kodu yoktur.

**Özyineleme tetikleyicisi.** **Her iki repoda da kodlanmış bir derinlik limiti, karmaşıklık sezgisi veya başarısızlık-sayacı tetikleyicisi yoktur** — alt-agent üretme kararı tamamen prompt metnine devredilmiştir. IDEA.md (satır 22-25): *"Reuse vs. fresh... The orchestrator picks: feed new info into the existing agent, or spawn a clean one."* `arc_agent/prompts.py` INITIAL_PROMPT satır 34: *"Be judicious—spawning agents has a cost. Only delegate when it genuinely helps"* — yine prompt-düzeyinde, zorlanmayan bir talimat. Kod-zorlamalı tek limitler güvenlik valfleridir: repo #2'nin düz `max_num_agents` sayacı (`ValueError` fırlatır) ve repo #1'in alt-agent-başına aksiyon bütçesi (`bounded_submit_action`, `agent.py:266-330`). Not: `Arcgentica.MAX_ACTIONS = 10_000` (satır 46-48) **ölü koddur** — `Arcgentica` `main()`/`_run()`'ı tamamen override ediyor ve bu sabiti kontrol eden temel `Agent.main()` döngüsünü hiç çağırmıyor. `prompts.py:231-243`'te bağlam-penceresi tabanlı bir özyineleme tetikleyicisi taslak halinde ama açıkça kapalı: `if False  # disabled until this is added in the SDK`.

### 3.4 $1.005 vs $8.900 maliyet iddiası: kod kanıtı yok

"$1,005"/"1005" veya "$8,900"/"8900" için her iki repoda **tam-repo grep hiçbir eşleşme bulamadı** (yalnızca `uv.lock` hash'leri ve `visualizer_logs/*.jsonl`'de ilgisiz rastlantısal sayı eşleşmeleri). Repo #1'de dolar-maliyet muhasebesi **hiç yok** — `UsageTracker` yalnızca ham token sayılarını biriktiriyor, hiçbir $ sabiti veya para birimi alanı yok. Repo #2'nin (farklı bir kıyaslama olan klasik ARC-AGI-2 için) bir maliyet metodolojisi **var**: `summary.py:16-20`, Anthropic token-başı fiyatlarını sabit koduyor (`INPUT=5, CACHED=0.5, ...`, yorum: "as of 10th February 2026, must be updated") ve `Cost per task: $6.9442` hesaplıyor — README'nin "$6.94/task" başlığıyla eşleşiyor, ama bu ARC-AGI-2 için görev-başına ortalama, ARC-AGI-3 için toplam bir $1.005/$8.900 rakamı değil. **Sonuç: $1.005 vs $8.900 iddiası her iki denetlenen repodaki hiçbir kodla desteklenmiyor** — dış kaynaklı (blog, tweet, leaderboard karşılaştırması) bir rakam olarak değerlendirilmeli, bu kod tabanından hesaplanabilir/doğrulanabilir değil.

### 3.5 Kod kalitesi

- **Bayat/bozuk testler (repo #1):** `tests/unit/test_swarm.py` ve `tests/unit/test_core.py`, repoda **var olmayan** `agents.structs`'tan import ediyor; ayrıca artık kullanılmayan `agents.swarm.requests.Session.post`'u mock'luyor — gerçek `Swarm` sınıfı `arc_agi.Arcade()` kullanıyor ve `_session` özniteliği yok. Bu testler `ModuleNotFoundError` ile collection aşamasında başarısız olur ve zaten Arcgentica özyineleme mekanizmasını değil, upstream temel-agent kodunu test ediyorlardı.
- Özyinelemeyi destekleyen altyapı için tek gerçek/geçen test kapsamı `tests/test_smoke.py` — `spawn_agent`, orkestratör döngüsü veya `bounded_submit_action`'ın kendisini test eden **hiçbir test yok** (dış `agentica` SDK'sının `spawn()`'ını mock'lamayı gerektirirdi).
- Repo #2'de **sıfır test dosyası var** ("CI" rozeti yeşil olsa da `.github/workflows/ci.yml` yalnızca `ruff format --check` + `ruff check` çalıştırıyor — linting, test değil).
- Her iki repo da davranışsal testler yerine `mypy strict = true`'ya güveniyor.

### 3.6 Güçlü/zayıf yönler

**Güçlü:** Özyineleme deseni gerçekten basit ve tekdüze — kendi kapsamına yerleştirilmiş, özyinelemeli tek bir bağlı metod — iki çok farklı alanda (statik grid dönüşümleri vs. gerçek-zamanlı oyun) yeniden kullanılabiliyor. Rol-tabanlı bağlam hijyeni (IDEA.md'de belgelenmiş): explorer'lar aksiyon alır ama orkestratörün tüm bilgi yükünü taşımaz; theorist'ler yalnızca özet alır, ham piksel değil.

**Zayıf:** Özyineleme derinliği/genişliği tamamen prompt mühendisliğiyle yönetiliyor, kodla değil — repo #1'de kaçak üretime karşı algoritmik bir koruma yok (yalnızca aksiyon bütçesi *oyun aksiyonlarını* sınırlıyor, *LLM çağrılarını*/*agent sayısını* değil). Repo #2'nin düz `max_num_agents` tavanı ağacın tamamına eşit uygulanıyor — bir açgözlü dal kardeşlerini aç bırakabilir. Repo #1'deki bayat testler swarm/core katmanının regresyon korumasının olmadığı anlamına geliyor.

**Yeniden implementasyon için sınırlama:** Asıl LLM-çağırma semantiği (context penceresi kesme, aynı handle üzerinde tekrarlanan `.call()`'lar arası "konuşma devamlılığı" nasıl implemente ediliyor) **opak** — ayrı, denetlenmemiş `agentica-server`/`symbolica-agentica` paketinde yaşıyor.

### 3.7 Compute/bağımlılık/çalışma-zamanı varsayımları

LLM API: Anthropic (`claude-opus-4-5`/`claude-opus-4-6`), OpenAI (`gpt-5.2`) veya OpenRouter — `agentica-server`'a geçilen `--inference-endpoint`/`--inference-token` ile seçiliyor, veya barındırılan "Agentica platform" (`AGENTICA_API_KEY`). Sandbox: repo #2 LLM-üretimi `transform` kodunu gerçek bir OS subprocess'inde (`asyncio.create_subprocess_exec`, `PYTHONHASHSEED=0`, duvar-saati timeout) çalıştırıyor; her iki README'nin hızlı-başlangıç komutları `agentica-server`'ı `--sandbox-mode='no_sandbox'` ile başlatıyor — yani belgelenen kurulum yolu, LLM-üretimi kodu **izole bir sandbox olmadan** çalıştırıyor. Eşzamanlılık/bütçe: repo #2 README'si `max-concurrent-invocations`'ın en az `max-concurrent * num-attempts * max-num-agents` olması gerektiğini belirtiyor (örnek: `60 * 2 * 10 = 1200`). ARC-AGI-3 için aksiyon bütçesi IDEA.md'ye göre ~800 toplam, alt-agent başına `bounded_submit_action` ile zorlanıyor (kodda global bir tavan yok).

---

## 4. alexisfox7/RGB-Agent

### 4.1 Görev tanımıyla ilgili kritik düzeltme

Görev talimatı, bu projenin OpenCode'u (genel bir coding-agent iskeleti) sardığını varsayıyordu; **bu önerme bu repo için geçerli değil**. Kanıt: `.gitmodules` dosyası var ama **0 byte** — hiçbir submodule tanımlı değil, yani eksik bir OpenCode kaynak ağacı değil, sadece boş bir stub dosyası. `*.py`/`*.md`/`*.toml`/`Dockerfile*` genelinde "opencode" için tam-metin arama sıfır sonuç verdi. README, `pyproject.toml`, `docker/*/Dockerfile` ve tüm agent sınıfları tutarlı biçimde yalnızca iki backend adlandırıyor: **Claude Code CLI** (`@anthropic-ai/claude-code`, `claude` komutu olarak çağrılıyor) ve **OpenAI Codex CLI** (`@openai/codex`, `codex` olarak çağrılıyor). Projenin eski adı "Read-Grep-Bash (RGB) Agent"; `pyproject.toml`'un açıklaması: *"An agent for ARC-AGI-3 that uses Read, Grep, and Bash to solve puzzles."* Bu bölüm, gerçekte burada olanı (Claude Code ve Codex — OpenCode değil) tanımlıyor.

### 4.2 Mimari

- `prolong_agent/agent/base.py` (190 satır) — `BaseAgent` ABC: paylaşılan oturum-durumu kalıcılığı, log-penceresi kırpma, `actions.json` ayrıştırma/doğrulama.
- `prolong_agent/agent/claude_code_agent.py` (833 satır) — `ClaudeCodeAgent`: `claude -p`'yi oyun-başına kalıcı bir Docker konteyneri içinde başlatır (`_ContainerPool`); `stream-json` olaylarını ayrıştırır.
- `prolong_agent/agent/codex_agent.py` (1049 satır) — `CodexAgent`: `codex exec`'i çağrı-başına taze (`--rm`) bir konteynerde başlatır; Codex'in ND-JSON olay akışını ayrıştırır.
- `prolong_agent/agent/action_queue.py` (95 satır) — `ActionQueue`: çok-aksiyonlu bir planın FIFO tahliyesi, skor değişince flush edilir.
- `prolong_agent/agent/game_state.py` (87 satır) — `GameState`: gözlemden "yerleşmiş" grid'i çıkarır, ASCII/hex tahta metni render eder.
- `prolong_agent/agent/prompts.py` (146 satır) — tüm prompt şablonları.
- `prolong_agent/agent/swarm.py` (538 satır) — `Swarm` + CLI `main()`: bir scorecard'a karşı N oyunu paralel thread'lerde çalıştırır.
- `prolong_agent/environment/runner.py` (563 satır) — `GameRunner`: `GameState` + `ActionQueue` + analyzer'ı bağlayan asıl oyun-başı döngü.
- `prolong_agent/utils/sandbox_net.py` (46 satır) — `--internal` bir Docker ağı + egress proxy oluşturur.
- **Hiçbir test suite yok** (`**/*test*` glob'u eşleşme vermiyor).

### 4.3 Coding-agent → ARC aksiyon uzayı arayüzü

Arayüz tamamen **dosya-tabanlı, bir Docker bind-mount üzerinden aracılık edilen** bir arayüz — CLI'ye özel bir araç/fonksiyon şeması verilmeden, CLI'nin stok Read/Write/Edit/Bash/Grep/Glob araçları kullanılıyor. Analyzer çağrısı başına (`ClaudeCodeAgent.analyze`/`CodexAgent.analyze`):
1. Harness, `logs.txt`'yi (her aksiyonun, tahtanın, önceki planın düz-metin dökümü) bind-mount'lu sandbox dizinine yazar/günceller.
2. Sistem promptunu `CLAUDE.md` (Claude) veya `AGENTS.md` (Codex) dosyasına yazar — CLI'ların kendi proje-talimatı sözleşmeleri, framework değişikliği gerektirmiyor.
3. CLI'yi subprocess olarak çağırır: `docker exec ... claude -p - --model ... --permission-mode bypassPermissions --disallowedTools Agent,Task,TodoWrite,...` veya `docker run --rm ... codex exec ...`.
4. Coding agent normal döngüsünü çalıştırır — log'u Read/Grep eder, grid diff'leri veya bağlı-bileşenler hesaplamak için Bash ile keyfi Python/shell çalıştırır, notlar yazar — ve `/workspace/actions.json` yazarak biter, örn. `{"actions": ["ACTION6(30,40)", "ACTION1", "RESET"]}`.
5. Harness bu dosyayı geri okur (`_read_actions_json`), `BaseAgent._parse_actions_json_text`/`_parse_action_entry` ile doğrular (`VALID_ACTIONS = {ACTION1..7, RESET}`), `ACTION6(x,y)` koordinat sözdizimini regex ile ayrıştırır.
6. Ayrıştırılan aksiyonlar `ActionQueue`'ya yüklenir ve `GameRunner._next_action` tarafından **başka LLM çağrısı olmadan** tek tek tahliye edilir; `ActionQueue.check_score` bir skor farkı algılarsa kalan kuyruğu flush eder ve analyzer'ın hemen yeniden ateşlenmesini zorlar.

Yani serbest-biçimli coding-agent çıktısı ile ARC'ın ayrık aksiyon uzayı arasındaki "çeviri katmanı", özel bir araç şeması değil, tek bir JSON dosya sözleşmesi + ~15 satırlık ayrıştırma/doğrulama kodudur.

### 4.4 ARC'a özgü iskelet ne kadar

Neredeyse tamamen prompt-düzeyinde ve minimal: `SYSTEM_PROMPT`/`SYSTEM_PROMPT_INPROMPT` çekirdek metni ~30 satır (`prompts.py:49-111`, "board-in-prompt" ablasyonu için iki varyantla); hedefi, log format işaretçilerini, `actions.json` çıktı sözleşmesini ve hex/ASCII renk lejantını belirtir. `game_state.py`'nin tahta render'ı ve `grid_utils.py` (37 satır) tek ARC-alanına-özgü veri-dönüşüm kodu. Geri kalanı (oturum yönetimi, konteyner havuzu, olay-akışı ayrıştırma, retry/backoff, kota algılama) jenerik harness altyapısıdır, oyun-özel mantık değil. Kabaca oran: ~150 satır gerçekten ARC-özgü kod/prompt'a karşı ~2700 satır backend-bağımsız süreç/oturum/harness makinesi. Altta yatan CLI'ye özel araçlar/fonksiyon şemaları verilmiyor — yalnızca `--disallowedTools` ile ilgisiz stok araçlar (Task/TodoWrite/WebSearch/vb.) çıkarılıyor, ARC'a özel araç eklenmiyor.

### 4.5 Aksiyon-verimliliği mekanizması: açık mı, ortaya çıkan mı?

Her ikisi de var, farklı işler görüyor. **Açık mühendislik:** `ActionQueue`, çağrı-başına bir kerede birden çok aksiyonu (varsayılan tavan 15) sıraya koyup çalıştırıyor — bu, oyun-başına LLM çağrı sayısını azaltıyor, aksiyon-başına-çözüm sayısını değil. **Ortaya çıkan/prompt-güdümlü:** Gerçek aksiyon tutumluluğu (bir seviyeyi çözmek için toplam daha az ACTION çağrısı) kodda hiçbir arama/planlama/doğrulama algoritmasıyla zorlanmıyor — tek kaldıraç prompt metninin kendisi: *"Your secondary objective is to minimize total cumulative actions used"*, *"Prefer short lists (1-2 actions) when testing a new hypothesis... scale up toward {action_cap} for proven sequences"* (`prompts.py:52,74`). README'nin verimlilik iddiaları (yayılan bir token sayısı iyileştirmesi) log dosyasının kendisinin harici bellek olarak işlev görmesine atfediliyor, açık bir aksiyon-arama mekanizmasına değil — "coding agent'ın doğal olarak tutumlu olması" hipotezini destekliyor.

### 4.6 Sandbox/Docker mimarisi

Dört imaj, iki amaç: `docker/claude-sandbox`/`docker/codex-sandbox` — sırasıyla CLI'yı `npm install -g` ile kuran ince `python:3.12-slim` imajları, kök-olmayan kullanıcı; `docker/anthropic-proxy`/`docker/openai-proxy` — alan-adı-allowlist'li (`.anthropic.com .claude.ai` / `.openai.com .chatgpt.com`) Alpine+Squid ileri-proxy'ler, `access_log none`. İzolasyon: konteynerler `--internal` bir Docker ağında (`rgb-internal`) çalışıyor, tek çıkış yolu Squid'e yönlendirilmiş `HTTP(S)_PROXY`; `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--memory=8g --cpus=4`, yalnızca `/workspace` bind-mount'lu. Proxy'ler maliyet/sızıntı kontrolü amaçlı allowlist güvenlik duvarları, günlük-tutan maliyet-izleme proxy'leri değil — asıl maliyet muhasebesi her CLI'nin kendi JSON olay akışından `total_cost_usd`/token alanları ayrıştırılarak host tarafında yapılıyor.

### 4.7 Kod kalitesi

- Hiçbir yerde otomatik test yok (`pytest`/`ruff` `pyproject.toml`'da dev-bağımlılık olarak listelense de `tests/` dizini yok).
- `claude_code_agent.py` ile `codex_agent.py` arasında ciddi kod tekrarı — `base.py`'nin "concentrated" iddiasına rağmen `CodexAgent` aslında `BaseAgent`'ı subclass **etmiyor**; her şeyi yerel olarak yeniden tanımlıyor (örn. `_parse_actions_json_text` her iki dosyada neredeyse birebir).
- `swarm.py`'de geniş bir CLI yüzeyi (`--session-mode`, `--log-window`, `--clear-every` vb.) — makale için ablasyon/araştırma altyapısı, temiz bir üretim config dosyası değil.
- Makul savunmacı mühendislik: durgunluk tespiti için heartbeat thread'leri, kota-tükenmesi algılama ve retry, atomik checkpoint yazımı (`tmp.replace()`), tüm ARC API çağrılarını saran retry/backoff.

### 4.8 Güçlü/zayıf yönler ve sınırlamalar

**Güçlü:** "Harness" (kuyruk, runner, metrikler) ile "backend" (Codex/Claude Code arasında runner değişikliği gerektirmeden geçiş) arasında temiz ayrım; sandbox gerçekten katmanlı savunma (ağ + cap-drop + kaynak limitleri); yeniden-sürdürülebilirlik (`--resume`) `logs.txt`'yi deterministik biçimde tekrar oynatıyor.

**Zayıf:** İki farklı harici CLI'yi shell-out edip akışlarını ayrıştıran bir harness için sıfır test kapsamı — üst akım bir CLI bayrak/format değişikliği (örn. Claude Code'un stream-json şemasını değiştirmesi) sessizce çalışma-zamanında başarısız olur; iki agent sınıfı arasındaki ~250 satırlık tekrar bir bakım riski; tüm çok-aksiyon verimlilik hikayesi prompt ifadesine dayanıyor, gözlemlenen verimsizliğe dayalı kod-düzeyinde bir ceza/tavan-ayarlama yok.

**Sınırlamalar:** Canlı bir Docker daemon'ı ve iki npm-kurulu tescilli CLI (`@anthropic-ai/claude-code`, `@openai/codex`) gerektiriyor — sıfırdan bir agent implementasyonu değil, tamamen Anthropic'in ve OpenAI'ın coding-agent ürünlerinin CLI kararlılığına bağımlı.

### 4.9 Compute/bağımlılık/çalışma-zamanı varsayımları

Varsayılan model dizeleri: Claude Code için `claude-opus-4-6` (`--effort high`), Codex için `gpt-5.4` (`--reasoning-effort none`) — her ikisi de `swarm.py` içinde sabit kodlu. Docker zorunlu, kod içinde Docker-dışı bir yürütme yolu yok. Varsayılan `--max-actions 500`/oyun; subprocess timeout varsayılan 2400s (Claude). README'nin belgelediği maliyet rakamı (25 oyun için) repoda hesaplanan/uygulanan bir sınır değil — `total_estimated_cost` izleniyor/loglanıyor ama bir koşuyu durdurmak için asla kullanılmıyor. `uv.lock`'ta ne `anthropic` ne `opencode` Python paketi var — her iki backend için de CLI-subprocess (SDK değil) entegrasyon yaklaşımını doğruluyor.

---

## 5. Tufalabs/duck-harness

### 5.1 Mimari — üç ayrı katman

- **`ARC3-Inference/`** ("The Duck") — harness'in kendisi:
  - `inference/agent/tool_agent.py` (2064 satır) — `ToolAgent`: çekirdek LLM-sürüş döngüsü, mesaj/geçmiş yönetimi, araç dispatch'i, bağlam-penceresi kırpma.
  - `inference/agent/python_tool_sandbox.py` (577 satır) — tek Python aracı için subprocess-izole yürütme.
  - `inference/agent/prompts.py` (114 satır) — sistem-promptu parçaları.
  - `inference/framework/solver.py` (1278 satır) — `HarnessSolver(taaf.solver.Solver)`: TAAF adaptörü; oyun-başı oynama döngüsü, gerçek motora karşı aksiyon yürütme, yerel vLLM sunucu yaşam döngüsü.
  - `inference/tools/{eval,significance,eval_plot,traces,vllm_runtime_lora_guard}.py` — skorlama, istatistiksel karşılaştırma, iz dışa aktarma.
- **`tufa-arc-agi-framework/` (TAAF)** — Duck'ın üzerine kurulu olduğu **ayrı, yeniden kullanılabilir** bir orkestrasyon kütüphanesi, oyun/çözücüden bağımsız:
  - `src/taaf/game.py` (689 satır) — `Game`, `GameState`, `GameRun`; ARC-AGI-3 skor formülünü sahipleniyor (`GameRun._compute_final_score`).
  - `src/taaf/solver.py` (83 satır) — soyut `Solver` temel sınıfı, `HarnessSolver` bunu implemente ediyor.
  - `src/taaf/benchmark.py` (480 satır) — `Benchmark`: N oyun × N geçiş'in asenkron orkestrasyonu, periyodik/atomik JSON+pickle kaydı.
  - `src/taaf/deploy_inline.py`/`deploy_slurm.py`/`deploy_kaggle.py` — takılabilir `DeploymentTarget` implementasyonları.
  - `src/taaf/diagnostics.py` (2987 satır) — HTML/plot üretimi + `_paired_score_test`/`_paired_permutation_test` istatistiksel test fonksiyonları.
  - Bağımlılık yönü: `ARC3-Inference/pyproject.toml`, `tufa-arc-agi-framework`'ü bağımlılık olarak bildiriyor — TAAF alt-seviye kütüphane, Duck onun üzerine kurulu uygulama.
- **`example-run/`** — tam bir kayıtlı yarışma koşusu (25 resmi oyun × 20 geçiş): ham model-çağrısı log'ları, render edilmiş transkriptler, HTML analiz — "agent kendi kodunu yazıyor" iddiasının somut birincil kanıtı.

### 5.2 Yerel Qwen 3.6 27B servisi

`configs/inference.json`'dan doğrulanan model kimliği: `"model_name": "vrfai/Qwen3.6-27B-FP8"` — Kaggle veri kümesi `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`'tan (önceden-FP8-kuantize edilmiş, herkese açık HF hub yolu değil, üretilebilirlik açısından bir boşluk) kaynaklanan, özel barındırılan bir kaynak.

**Servis yığını vLLM'dir**, birden çok şekilde doğrulanmış: `Makefile`'ın `server:` hedefi doğrudan `vllm serve <model>` çağırıyor (`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`); `pyproject.toml`'un `server` extra'sı belirli bir vLLM wheel'ini pinliyor (`0.17.2rc1.dev150...`); `inference/tools/vllm_runtime_lora_guard.py`, sessiz bir vLLM LoRA-yükleme uyarısını sert bir `RuntimeError`'a çeviriyor; `openai_compat.py::normalize_provider`, herhangi bir OpenAI-uyumlu sağlayıcıyı varsayılan olarak `"vllm"`'e çeviriyor.

Sunucu konfigürasyonu (`configs/inference.json`): host `127.0.0.1:1234`, `tensor_parallel_size: 1`, `gpu_memory_utilization: 0.92`, `context_window: 32768`, `tool_call_parser: "qwen3_coder"`, `reasoning_parser: "qwen3"`, `chat_template_kwargs.preserve_thinking: true`. Gerçek bir Slurm koşusu kaydı (`example-run/run_in_worker.py`) `local_server_count=2` (B200 GPU başına bir vLLM sunucusu), `concurrency=64`, `analyzer_timeout=120.0` gösteriyor.

Çağrılar `ToolAgent._chat_completion`'da düz bir `requests.post`'la `{base_url}/chat/completions`'a gidiyor — SDK istemcisi yok, streaming yok (`"stream": False`). **Prompt yapısı:** sistem promptu `_build_system_prompt`'ta sabit parçaların (oyun genel bakışı → çalışma-zamanı durum belgeleri → görsel-oyun rehberliği → Python-aracı rehberliği → araç-oturumu kuralları) birleştirilmesiyle oluşuyor. **Tam olarak bir** araç şeması kayıtlı: `python` adında, tek zorunlu string parametresi `code` olan bir fonksiyon. Araç-çağrı formatı Qwen3-coder'ın XML-benzeri işaretlemesini takip ediyor (`<tool_call><function=python><parameter=code>...`), hem regex-tabanlı kurtarma yolunda hem de canlı bir transkriptte doğrulandı. Örnekleme parametreleri: `temperature=0.6, top_p=0.95, top_k=20, thinking=true`.

### 5.3 Kod-yürütme mekanizması

Her `python` araç çağrısı, **taze bir `subprocess.Popen`-üretimi Python yorumlayıcısında** yürütülüyor, süreç-içi `exec()` değil: `run_sandboxed_python`, `[sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP]` başlatıyor (`-I`: yalıtılmış mod, `-S`: `site` import'u yok), minimal sabit-kodlu env, taze geçici dizin, kendi süreç grubunda (`start_new_session=True`, timeout'ta `SIGKILL` ile öldürülebilir). POSIX'te `resource.setrlimit` CPU süresini, dosya boyutunu, açık-dosya sayısını sınırlıyor — bu süreç/OS-düzeyi izolasyon (subprocess + rlimit'ler), **Docker/gVisor/seccomp değil**; kazara kaynak tüketimine karşı yeterli ama kararlı bir düşmana karşı güçlü bir güvenlik sandbox'ı değil.

Çocuk süreç içinde, `exec(compiled, runtime_globals, runtime_globals)` kısıtlı bir globals sözlüğüyle çalışıyor: `__builtins__` açık bir `SAFE_BUILTINS` allowlist'inden yeniden kuruluyor (~45 isim — `open`, `eval`, `exec`, doğrudan `__import__` yok), `__import__` yalnızca sabit bir `SAFE_MODULES` kümesine (`bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string` — `os`, `sys`, `subprocess`, soket yok) izin veren `_safe_import` ile değiştiriliyor. Tahtanın **ham sayısal grid'i kasıtlı olarak gizli** — yalnızca `.ascii` (harf-kodlu) ve `.segmentation` (bağlı-bileşen nesneleri) görünümleri `FrameView` üzerinden sunuluyor.

Host↔sandbox iletişimi stdin/stdout üzerinden **satır-sınırlı JSON protokolü**: sandbox içindeki `action(actions)` fonksiyonu stdout'a `{"type":"action","actions":[...]}` yazıp stdin'den cevap bekliyor; host'un `run_sandboxed_python` döngüsü bu isteği okuyup gerçek `action_handler` callback'ini çağırıyor (`ToolAgent._run_python_tool → _handle_action` → `HarnessSolver`'ın `step_env`'i → `taaf.game.Game.execute_action` — gerçek ARC-AGI-3 motoru), sonra taze durumu `{"type":"action_result", ...}` olarak geri gönderiyor. Bu, tek bir Python parçasının bir döngü içinde `action()`'ı birden çok kez, her seferinde taze durumla çağırmasına izin veriyor. Tamamlanınca çocuk `{"type":"final", ...}` veya `{"type":"error", ...}` gönderiyor; istisnalar host dosya yollarını gizlemek için traceback-temizleniyor. Bu, "agent kendi kodunu yazıyor" için mekanik geri-besleme döngüsü: `ToolAgent.analyze`, sınırlı bir while döngüsü çalıştırıyor (chat-completion → tool_calls ayrıştır → `_dispatch_tool` → `run_sandboxed_python` → sonucu ekle → sonraki chat-completion), ta ki `action()` gerçekten bir oyun hamlesi yürütene kadar.

### 5.4 "Elle-yapılmış araçlar doğaçlamayı engelledi" iddiasının kanıtı

**Bu repoda bu ifadeye dair hiçbir dil yok.** `prompts.py`, tüm `README.md` dosyaları ve her `.py`/`.md`/`.json` dosyası `hand-craft`, `improvis`, `minimal`, `scaffold` için büyük/küçük harf duyarsız taranmış — tek eşleşmeler ilgisiz "minimal" kullanımları. `git log --oneline -30` (ve `--all`) kök dizinde **yalnızca bir commit** gösteriyor: `7652836 "Refactor README for clarity and formatting"`. `ARC3-Inference/` ve `tufa-arc-agi-framework/`'ün kendi `.git`'i yok — bütün paket, geliştirme geçmişi olmayan tek bir sıkıştırılmış "sunum" görüntüsü, dolayısıyla aranacak bir commit-mesajı izi de yok. Kök `README.md`, tasarım tartışmasını harici bir Kaggle yazısına ve bir blog gönderisine (`tufalabs.ai/research/duck-harness/`) atfediyor, ikisi de klonlanan repoya dahil değil. **Bu spesifik iddia kodda veya geçmişte doğrulanamıyor** — repo dışından kaynaklandığı varsayılmalı, bu kod tabanının belgelediği bir şey olarak değil. Yapısal minimalizmin kendisi (aşağıda) gerçek ve doğrudan gözlemlenebilir; belirtilen *gerekçe* burada mevcut değil.

### 5.5 Bilinçli olarak minimal tutulanlar

- **Tam olarak bir LLM-yüzü araç**: `python`, tek parametre (`code: string`). `move_up()`, `click(x,y)` gibi primitif-başına araç yok; tüm oyun etkileşimi tek bir jenerik `action(actions)` fonksiyonundan geçiyor.
- **Önceden-inşa edilmiş görsel/sezgisel araç yok**, tek istisna segmentasyon yardımcı programı — object-tracker, pathfinder, OCR yok; prompt açıkça modele BFS/DFS/beam-search'ü **kendisinin yazmasını** söylüyor.
- **Kalıcı kod/oturum durumu yok**: "Her `python` araç çağrısı taze başlar" — dosya sistemi yok, modül önbelleği yok; model her turda yardımcı mantığı yeniden türetmeli/yapıştırmalı.
- **Dar stdlib allowlist'i**: 13 modül, sandbox içinde numpy/scipy yok (host tarafında ikisi de bağımlılık olsa da).
- **Ham piksel grid gizli**: model 0-15 renk dizisini asla doğrudan görmüyor, yalnızca türetilmiş `.ascii`/`.segmentation` görünümlerini.

Ağır-iskeletli bir tasarımla karşılaştırma: nesne-tipi-başına araç kütüphanesi yok, araç olarak sunulan sabit-kodlu bir BFS/pathfinder fonksiyonu yok, kalıcı bir "hafıza" aracı/veritabanı yok — hafızaya en yakın şey, modelin kendi akıl yürütmesinde yaymasının istendiği, sezgisel olarak ayrıştırılan bir metin-kalıbı sözleşmesi (`World model:`/`Goal model:`/`Plan:` önekleri).

### 5.6 Kod kalitesi

- **Yapı:** TAAF (motor-bağımsız orkestrasyon/skorlama) ile ARC3-Inference (LLM agent + sandbox + prompt'lar) arasında temiz ayrım. Birçok fonksiyon `R2.x`/`R11.x`/`R12.x` gereksinim-ID'leriyle etiketlenmiş docstring'lere sahip — bir spec belgesinin (bu repoya dahil değil) tasarımı yönlendirdiğini gösteren gerçek bir mühendislik-disiplini sinyali.
- **Testler:** Her iki `pyproject.toml` da `testpaths = ["tests"]` bildiriyor ve README'nin repo haritası bir `tests/` dizini iddia ediyor ("unit coverage for config, duck runtime, TAAF runner, viewer, scoring, significance, and traces") — ama **bu klonlanan repoda gerçek bir `tests/` dizini yok** (`find -iname "*test*"` `example-run/` dışında sıfır sonuç). Bu, belgelenen repo haritası ile bu genel "sunum" kopyasında gerçekten teslim edilen arasında gerçek bir fark — testler muhtemelen iç repoda var ama buraya dahil edilmemiş.
- **Değerlendirme titizliği — gerçek bir güçlü yön:** `inference/tools/significance.py`, eşleştirilmiş oyun-başı skor farkları üzerinde normal yaklaşımla tek-taraflı Bayesçi bir test implemente ediyor (`P(true_delta > 0 | results) ≥ 0.90`), **artı** 10.000 örneklemli bootstrap %90 güven aralığı, **artı** TAAF'ın kendi frekansçı eşleştirilmiş t-testine ve eşleştirilmiş işaret-çevirme permütasyon testine (`taaf/diagnostics.py:1443,1487`) karşı çapraz kontrol. İki koşuyu karşılaştırmadan önce uyumluluk kontrolleri de zorluyor (aynı çalışma-zamanı bütçesi, aynı GPU donanımı, aynı veri kümesi/etiketler, aynı deneme sayıları) — gerçekten dikkatli bir deneysel-metodoloji mühendisliği.

### 5.7 Güçlü/zayıf yönler ve sınırlamalar

**Güçlü:** Duck'ın ötesinde yeniden kullanılabilir, temiz, araç-bağımsız TAAF çekirdeği; uzun-süreli Slurm işleri için disiplinli pickle/deepcopy sözleşmeleri; gerçek subprocess-düzeyi sandboxlama; istatistiksel olarak titiz A/B karşılaştırma araçları; kapsamlı çalıştırma-başı gözlemlenebilirlik.

**Zayıf/sınırlama:** Sandbox rlimit'li OS-süreç izolasyonu, sertleştirilmiş bir konteyner/seccomp sandbox'ı değil — tek bir güvenilir yerel model için kabul edilebilir ama katmanlı savunma değil; model kimliği (`vrfai/Qwen3.6-27B-FP8`) kanonik bir genel HF checkpoint'i değil, özel barındırılan kuantize bir Kaggle veri kümesi anlık görüntüsü — tam üretilebilirliği zedeliyor; bu genel repodaki `tests/` dizini ve `docs/requirements.md` referans veriliyor ama yok, dolayısıyla belgelenen birçok "R##" sözleşmesi bağımsız olarak kontrol edilemiyor; TAAF'ın README'si açıkça "çevrimdışı yerel-veri kümesi/üreteç kod yolları burada paketlenmedi" diyor — bu, daha büyük bir iç monorepo'nun küratörlü bir alt kümesi (üçüncü, özel bir repo olan `re-arc-3`'e de referans veriyor).

### 5.8 Compute/bağımlılık/çalışma-zamanı varsayımları

**GPU:** Slurm varsayılan konfigürasyonu `gpu: B200, gpu_count: 2`, `time: 13:00:00`; her GPU için bir vLLM sunucusu, `tensor_parallel_size: 1`. 27B FP8 model `tensor_parallel_size=1`'de, her sunucunun tek bir yüksek-bellekli GPU'ya sığdığını ima ediyor (FP8 ≈ BF16'nın yaklaşık yarısı bellek). **Bağımlılıklar:** Python tam olarak `3.12.12`'ye sabitlenmiş; `uv` ile ortam yönetimi; vLLM belirli bir ön-sürüm wheel URL'sine sabitlenmiş; `arcengine>=0.9.3` ve `arc_agi>=0.9.8` (özel ARC-AGI-3 yarışma paketleri, muhtemelen PyPI'da değil). **Slurm dağıtımı:** `ARC3-Inference` + `tufa-arc-agi-framework` kaynaklarını iş dizinine paketliyor, bir worker giriş noktası (`run_in_worker.py`) üretiyor. **Kaggle dağıtımı:** `make kaggle-duck`, kaynakları özel bir Kaggle veri kümesine paketliyor ve özel bir Kaggle notebook'u push ediyor; varsayılan yarışma-koşusu şekli: 25 resmi oyun, `model=local`, 16 eşzamanlı oyun, oyun-başı 75 dakika, toplam 90 dakika Kaggle çalışma-zamanı. **GPU'suz yol:** viewer (`make view`) ve paketlenmiş `example-run/` yalnızca temel Python bağımlılıkları gerektiriyor — model-servis yığınını yeniden üretmeye gerek kalmadan tam bir kayıtlı koşuyu incelemeyi mümkün kılıyor.

---

## 6. Karşılaştırmalı sentez

| Proje | "World model" temsili | Doğrulama mekanizması | Planlama/arama | Çok-agent orkestrasyon | Sandbox |
|---|---|---|---|---|---|
| **baseline1** (`ewma_sv_v1.6`) | LLM-yazımı `dict` + sabit fonksiyon imzaları | Tam piksel-hassasiyetinde replay (`np.array_equal`) | Kod içinde yok — LLM'in yazdığı, denetlenemez | Yok (tek Codex konuşması, protokol dispatch'i Python'da) | Docker + proxy-allowlist, Codex CLI sandbox'sız |
| **Tycho** | LLM-yazımı `State`/`transition`/`render`/`outcome` sözleşmesi | `wmlib.verify()`/`verify_outcome()` — çift kanal (dinamik + terminal), UNKNOWN-hücre semantiği | `wmlib.plan_bfs`/`plan_astar`/`plan_subgoals` — **framework-verilmiş**, LLM'in çağırdığı hazır kütüphane | 4 deklaratif mod (`no_world_model`/`single`/`orchestrator`/`trigger`), builder tek-seviye subagent | Docker/Finch, `--network none`, doğrulanmış izolasyon |
| **symbolica arcgentica** | Yok (world-model kavramı bu projede merkezi değil) | Yok | Yok — kod-yazma yerine agent-yazma odaklı | Gerçek özyinelemeli `spawn_agent`/`call_agent`, sınırsız derinlik, prompt-güdümlü tetikleyici | Repo #2'de subprocess, dokümante `no_sandbox` varsayılan |
| **RGB-Agent** | Yok (world model değil, doğrudan coding-agent akıl yürütmesi) | Yok | Yok — Claude Code/Codex'in kendi genel planlama tarzı | Yok (tek analyzer çağrısı + aksiyon kuyruğu) | Docker + proxy-allowlist, iki farklı CLI |
| **duck-harness** | Yok (yapılandırılmış world-model sözleşmesi değil, serbest-biçim "working world model" notu) | Yok | LLM'in tek `python` aracı içinde kendi yazdığı arama (BFS vb. prompt'ta önerilir) | Yok | Subprocess + rlimit + builtin/import allowlist, container değil |

Bu tablo, beş projenin ortak "LLM kod yazar, gözleme karşı doğrular, harekete geçmeden önce planlar" sloganının altında aslında çok farklı mimari taahhütler taşıdığını gösteriyor: **baseline1** ve **Tycho** kelimenin tam anlamıyla çalıştırılabilir, doğrulanabilir bir world-model sözleşmesi zorluyor (Tycho'nunki daha zengin — çift-kanal yanlışlama, hazır BFS/A* kütüphanesi, dört deklaratif mod); **symbolica arcgentica** world-model fikrini tamamen terk edip özyinelemeli çoklu-agent ayrıştırmasına odaklanıyor; **RGB-Agent** ve **duck-harness** ise "dünya modeli" kavramını hiç zorlamıyor — ikisi de genel bir coding-agent'ın (sırasıyla Claude Code/Codex CLI ve özel bir tek-araçlı Qwen3.6 döngüsü) kendi doğal akıl yürütmesine güveniyor, minimal bir dosya/JSON sözleşmesiyle ARC aksiyon uzayına bağlanıyor.

Maliyet/verimlilik iddiaları açısından ortak bir örüntü: **hiçbir proje kodda uçtan uca dolar-maliyet hesaplaması içermiyor** (Tycho token-fiyat tablosu tutuyor ama harici bir sabit liste; symbolica arcgentica'nın ana ARC-AGI-3 şablonunda hiç $ muhasebesi yok; RGB-Agent ve duck-harness maliyeti izliyor ama bir bütçeyi zorlamak için kullanmıyor). $1.005 vs $8.900 gibi başlık rakamları, incelenen kod tabanlarının dışında (blog/leaderboard) üretilmiş görünüyor.

---

## 7. Metodolojik not

Bu rapor, beş reponun her biri için bağımsız, odaklanmış bir kod incelemesi (arka planda çalışan araştırma ajanları + elle doğrulama) sonucunda derlendi. Tüm alıntılanan dosya yolları ve fonksiyon/sınıf adları klonlanan kaynak koddan doğrudan okunmuştur; hiçbir repo erişilemez olmadığı için "klonlanamadı" notu gerekmemiştir. `symbolica-ai` bölümünde belirtildiği gibi tek önemli erişim sınırlaması, özyinelemeli agent-çağırma SDK'sının (`agentica-server`/`agentica-python-sdk`) ayrı bir repoda yaşaması ve bu incelemenin kapsamı dışında kalmasıdır — bu, "spawn nasıl gerçek bir LLM API çağrısına dönüşür" sorusunun yalnızca dışarıdan (README/config kanıtlarıyla) çıkarsanabildiği, doğrudan kod okunarak doğrulanamadığı anlamına gelir.
