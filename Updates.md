2025-12-22 16:08:04 新建 Datasets/Questions/medical_questions_100_balanced.json：从 medical_questions.json 按 question_type 均衡抽取 100 条（每类 25）。
2025-12-22 16:41:30 修改 Examples/run_clearrag.py：medical 子集 questions 路径改为 medical_questions_100_balanced.json。
[2026-02-23 12:39:22] 新增 AGENTS.md（Repository Guidelines），补充仓库结构、运行/评测命令、代码风格、测试与PR规范。
[2026-03-02 21:14:38] 深度分析仓库结构与模块划分：完成顶层目录、核心功能模块、框架子模块、数据-推理-评测流程梳理。
[2026-03-02 21:42:45] 审查 Examples 运行脚本代码质量：定位可读性、冗余、不一致与可执行性问题，并形成按严重度排序的整改建议。
[2026-03-02 22:10:57] 优化 Examples 运行脚本（方案1）：统一异常输出结构、移除硬编码CUDA设备、修复run_lightrag模式日志bug、补充context列表规范化并清理未使用导入。
[2026-03-02 22:23:38] 进一步优化 Examples：新增 common_benchmark.py，统一分组/上下文规范化/错误结果/JSON-Parquet加载，6个运行脚本完成去重接入。
[2026-03-02 23:30:25] 继续去重 Examples：新增统一输出路径与结果写盘函数（build_output_path/save_results_json），6个运行脚本完成接入，移除重复 open/json.dump 代码。
[2026-03-03 00:57:44] 实现统一YAML参数管理：新增 configs/experiment.yaml 与 Examples/run_from_yaml.py（单配置驱动多框架命令分发），并更新运行.md 的统一运行说明。
[2026-03-03 01:03:46] 复查统一参数完整性：补充 common.neo4j 到 run_from_yaml 环境映射，增加 frameworks.<name>.subset 覆盖与子集合法性校验，更新 experiment.yaml/运行.md。
[2026-03-03 01:13:01] 新增 run.enforce_common 强一致模式，统一 subset/model/top_k；为 run_hipporag2.py 增加 --top_k 并映射 retrieval/linking/qa；更新 configs/experiment.yaml 与 运行.md。
[2026-03-03 09:33:44] 修复 run_from_yaml auto 框架选择：当 run.framework 指向 enabled=false 的框架时直接报错，避免误执行。
[2026-03-03 09:35:49] run_from_yaml dry-run 命令输出增加敏感参数脱敏（llm/embed api key），避免明文泄露。
[2026-03-03 09:36:45] 调整 configs/experiment.yaml: run.framework 从 lightrag 改为 all，匹配 lightrag.enabled=false，避免 auto 模式报错。
[2026-03-03 09:37:54] 按需求移除 run_from_yaml dry-run 的 API Key 脱敏逻辑，恢复命令明文打印。
[2026-03-03 09:44:00] 修复 ClearRAG 400：将 common.embed.name 从 BAAI/bge-large-en-v1.5 调整为 embedding-3（匹配智谱 embeddings 接口）。
[2026-03-03 09:47:03] 调整 clearrag 适配器 Linear 路径策略：默认使用 <working_dir>/linear（不再默认 ./data/linear），并在 checkpoint 存在但 linear 文件缺失时强制重建，避免查询阶段报 entity_ids.json 缺失。
[2026-03-03 09:53:24] 按外层统一方案处理 Linear 路径：回退 clearrag 内部改动；在 run_from_yaml 注入 LINEAR_DATA_DIR（默认 frameworks.clearrag.linear_data_dir 或 <base_dir>/linear），并更新 experiment.yaml/运行.md。
[2026-03-03 09:55:28] 对齐 ClearRAG 线性索引读取路径：common.neo4j.database 改为 first-test-db，frameworks.clearrag.linear_data_dir 改为 ./clearrag_workspace/Medical/data/linear；验证 entity_ids.json 路径存在。
[2026-03-03 10:33:03] 修复 LightRAG zhipu.py 合并冲突标记导致的 SyntaxError，合并装饰器参数并通过 AST 语法校验。
[2026-03-03 10:52:59] 增加统一路径防覆盖能力：run.run_id + paths 根目录 + {run_id}/{framework} 模板解析；run_from_yaml 自动下发 base_dir/output_dir；为 lightrag/fast-graphrag/hipporag2 新增 --output_dir。
[2026-03-03 11:00:32] 调整为索引复用模式：各框架 base_dir 固定、默认 skip_build=true；新增 fast-graphrag/hipporag2 的 --skip-build 支持；run_from_yaml 统一下发 skip_build。
[2026-03-03 17:20:37] 补充 LightRAG 对 medical_100 子集支持：run_from_yaml 扩展 lightrag 可用子集；run_lightrag 新增 medical_100 路径映射到 medical_questions_100_balanced.json，并更新 --subset 参数校验。
[2026-03-03 17:24:18] 统一数据集子集配置：新增 Examples/subset_registry.py 作为唯一子集与路径注册表；run_from_yaml 及 run_lightrag/run_clearrag/run_fast-graphrag/run_hipporag2/run_digimon 全部改为从该注册表读取 subset 支持与数据路径。
[2026-03-03 20:57:13] 调整为全框架统一子集支持：subset_registry 中 fast-graphrag/hipporag2/digimon 也改为支持 medical、medical_100、novel、hotpotqa。
[2026-03-04 00:22:24] 修复 run_clearrag 参数兼容：--passage-retrieval-mode 新增 none 选项（与 clearrag/config.py 检索模式一致）。
[2026-03-04 11:27:52] run_clearrag 适配器初始化显式指定 skill_registry_path=./clearrag/clearrag/skills，修复 skills 目录错位导致 load_skill 可用列表为空的问题。
[2026-03-04 16:53:17] 调整 sample 语义：移除 run_lightrag/run_clearrag/run_fast-graphrag/run_hipporag2/run_digimon 中按 args.sample 截断语料库数量的逻辑，sample 仅用于每个语料的问题采样。
[2026-03-04 16:59:13] 重设计采样参数：新增 corpus_sample（控制语料库数量），并将 sample 语义固定为仅控制每个语料库的问题数量；run_from_yaml 与 run_lightrag/run_clearrag/run_fast-graphrag/run_hipporag2/run_digimon 已统一透传与解析。
[2026-03-04 17:12:10] run_from_yaml 参数一致性增强：sample 与 corpus_sample 也按 enforce_common 合并规则解析（common 优先 / 可切换），避免框架局部配置破坏统一实验参数。
[2026-03-04 21:23:26] LightRAG 运行流程增强：新增 --corpus-concurrency（默认1）与 --index-only；run_from_yaml 已透传对应配置。run_lightrag 初始化显式设置 workspace=corpus_name，降低多语料并发下 pipeline 互相排队导致的提前查询问题。
[2026-03-05 22:12:08] 更新运行.md：新增 LinearRAG 框架状态、运行命令（完整/仅索引/仅查询/YAML 调度）与参数说明，补充 sample 与 corpus_sample 语义及 skip-build 索引依赖说明。
[2026-03-05 22:16:54] 更新运行.md：在 LinearRAG 章节新增依赖安装说明（linearrag requirements + SpaCy en_core_web_trf 模型）。
[2026-03-06 11:06:00] 修复 LinearRAG 结果 context 重复：在 Examples/run_linearrag.py 的 benchmark 输出阶段新增 normalize_context_list，去除 passage 编号前缀后按正文保序去重，不修改 LinearRAG 核心检索逻辑。
