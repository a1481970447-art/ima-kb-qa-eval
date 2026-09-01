"""D3 产出：RAG 复现 pipeline（手写最简链路，零框架依赖）。

对应 7 天计划 D3：解析 → 切块 → embedding → 向量检索 → 生成
模型走混元 TokenHub（llm_client.py），保持"腾讯叙事"一致。

设计要点（详见 D3_pipeline说明.md）：
- front-matter 物理隔离：D1 教训——元数据被索引进检索池造成污染（A-14），解析时直接剥离
- 切块贴近 ima 可观察行为：chunk 携带「文档·章节」标题前缀 + 每篇一个摘要 chunk，
  复现 D1 观察到的"高亮多为摘要/节标题而非答案段落"
- top-k 可配置（CLI -k / 环境变量 RAG_TOP_K）
- answer() 返回结构化结果（answer/retrieved/latency_*），直接作为 D4 评分与 D5 归因的输入

用法：
    python rag_pipeline.py build                # 建索引（增量：按文档 sha1 缓存向量）
    python rag_pipeline.py ask "ima 什么时候上线"  # 单条问答（端到端验收）
    python rag_pipeline.py ask -k 4 "..."       # top-k 可配置
    python rag_pipeline.py ask --json "..."     # 输出结构化 JSON（D4/D5 接口）
    python rag_pipeline.py smoke                # 从 eval_set 抽 4 条演示端到端
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from llm_client import chat, embed  # noqa: E402

CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "rag_index"
EVAL_SET = ROOT / "eval_set.json"

# ---------------- 可配置参数（环境变量可覆盖） ----------------
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))      # 单 chunk 字符数
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))  # 滑窗重叠
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "8"))          # 检索条数
EMBED_BATCH = 32

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

SYSTEM_PROMPT = (
    "你是知识库问答助手（复现 ima 知识库问答场景）。请严格遵守：\n"
    "1. 只依据【参考资料】回答，不得使用资料之外的知识编造。\n"
    "2. 若参考资料不足以回答问题，直接回答："
    "\"知识库中没有找到与该问题相关的可靠内容。\""
    "可补充一句库里实际有什么相关内容。\n"
    "3. 回答开头用【依据 Dxx】标注主要依据的文档编号（如【依据 D02】）。"
)


# ---------------- 1. 解析层 ----------------
def strip_front_matter(text: str) -> str:
    """剥离 YAML front-matter——D1 教训：内部元数据与正文必须物理隔离。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).strip()
    return text  # 没有闭合，原样返回


def parse_markdown(text: str) -> list[tuple[str, str]]:
    """按标题层级切节，返回 [(章节路径, 节正文)]。章节路径保留标题词，
    使标题字面匹配可命中——贴近 ima 高亮节标题的可观察行为。"""
    stack: list[tuple[int, str]] = []
    cur_title, cur_lines = "(开头)", []
    sections: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if any(l.strip() for l in cur_lines):
                sections.append((cur_title, "\n".join(cur_lines).strip()))
            level, title = len(m.group(1)), m.group(2).strip()
            stack = [s for s in stack if s[0] < level] + [(level, title)]
            cur_title = "·".join(t for _, t in stack)
            cur_lines = []
        else:
            cur_lines.append(line)
    if any(l.strip() for l in cur_lines):
        sections.append((cur_title, "\n".join(cur_lines).strip()))
    return sections


def parse_pdf(path: Path) -> list[tuple[str, str]]:
    """PDF 按页提取（pypdf），每页一节。英文论文双栏乱序属已知限制，
    与 D1 观察到的 ima 英文 PDF 高亮质量问题同源。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, 1):
        t = (page.extract_text() or "").strip()
        t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        if len(t) >= 30:
            sections.append((f"p.{i}", t))
    return sections


# ---------------- 2. 切块层 ----------------
def sliding_chunks(text: str, size: int, overlap: int) -> list[str]:
    """节内滑窗切块；末尾碎片并入前一块，避免产生无意义短 chunk。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    pieces = [text[i : i + size] for i in range(0, len(text), step)]
    if len(pieces) > 1 and len(pieces[-1]) < 60:
        pieces[-2] += " " + pieces.pop()
    return pieces


def build_doc_chunks(path: Path, doc_id: str) -> list[dict]:
    """单篇文档 → chunk 列表（含元数据）。

    每篇额外生成一个「摘要」chunk（正文开头 400 字），复现 ima
    检索高亮常为文档摘要的可观察行为。
    """
    if path.suffix.lower() == ".pdf":
        sections = parse_pdf(path)
        summary_src = sections[0][1] if sections else ""
        summary_section = "摘要(首页)"
    else:
        text = strip_front_matter(path.read_text(encoding="utf-8"))
        sections = parse_markdown(text)
        summary_src = "\n".join(b for _, b in sections)
        summary_section = "摘要(开头)"

    chunks: list[dict] = []
    if summary_src:
        chunks.append({"section": summary_section, "text": summary_src[:400]})
    for sec, body in sections:
        for piece in sliding_chunks(body, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"section": sec, "text": piece})

    chunks = [c for c in chunks if len(c["text"]) >= 30]
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{doc_id}#{i:03d}"
        c["doc_id"] = doc_id
        # 向量化文本携带「文档·章节」前缀：标题词可命中（贴近 ima 行为）
        c["embed_text"] = f"[{doc_id}·{c['section']}] {c['text']}"
    return chunks


# ---------------- 3. 索引层（增量缓存） ----------------
def load_corpus_docs() -> list[dict]:
    return json.loads(
        (CORPUS_DIR / "corpus_index.json").read_text(encoding="utf-8")
    )["documents"]


def cmd_build(force: bool = False) -> None:
    docs_meta = load_corpus_docs()
    INDEX_DIR.mkdir(exist_ok=True)
    index_path = INDEX_DIR / "index.json"

    old_docs, old_vecs = {}, {}
    if index_path.exists():
        old_docs = json.loads(index_path.read_text(encoding="utf-8")).get("docs", {})
        if (INDEX_DIR / "vectors.npz").exists():
            with np.load(INDEX_DIR / "vectors.npz") as npz:
                old_vecs = {k: npz[k] for k in npz.files}

    new_docs: dict[str, dict] = {}
    vec_entries: dict[str, np.ndarray] = {}
    pending: list[dict] = []  # 待向量的 chunk

    for d in docs_meta:
        doc_id, fname = d["doc_id"], d["file"]
        path = CORPUS_DIR / fname
        if not path.exists():
            print(f"  ! 缺文件，跳过 {doc_id}: {fname}")
            continue
        sha = hashlib.sha1(path.read_bytes()).hexdigest()
        if not force and old_docs.get(doc_id, {}).get("sha1") == sha:
            new_docs[doc_id] = old_docs[doc_id]
            vec_entries[doc_id] = old_vecs[doc_id]
            continue
        t0 = time.perf_counter()
        chunks = build_doc_chunks(path, doc_id)
        new_docs[doc_id] = {
            "file": fname,
            "sha1": sha,
            "n_chunks": len(chunks),
            "chunks": chunks,
        }
        pending.extend(chunks)
        print(f"  {doc_id} {fname} → {len(chunks)} chunks "
              f"({time.perf_counter() - t0:.1f}s 解析)")

    if pending:
        print(f"向量化 {len(pending)} 个新 chunk（批次 {EMBED_BATCH}）…")
        texts = [c["embed_text"] for c in pending]
        vecs: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            vecs.extend(embed(texts[i : i + EMBED_BATCH]))
            done = min(i + EMBED_BATCH, len(texts))
            print(f"  embedded {done}/{len(texts)}")
        by_doc: dict[str, list] = defaultdict(list)
        for c, v in zip(pending, vecs):
            by_doc[c["doc_id"]].append(v)
        for doc_id, vs in by_doc.items():
            vec_entries[doc_id] = np.asarray(vs, dtype=np.float32)
    else:
        print("所有文档均未变化，索引已缓存。")

    np.savez(INDEX_DIR / "vectors.npz", **vec_entries)
    index_path.write_text(
        json.dumps(
            {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "n_docs": len(new_docs),
                "n_chunks": sum(d["n_chunks"] for d in new_docs.values()),
                "docs": new_docs,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    total = sum(d["n_chunks"] for d in new_docs.values())
    print(f"索引完成：{len(new_docs)} 篇 / {total} chunks → {INDEX_DIR}/")


# ---------------- 4. 检索 + 生成（pipeline 主体） ----------------
class RagPipeline:
    def __init__(self) -> None:
        index_path = INDEX_DIR / "index.json"
        if not index_path.exists():
            sys.exit("索引不存在，先运行: python rag_pipeline.py build")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.chunks: list[dict] = []
        matrices = []
        with np.load(INDEX_DIR / "vectors.npz") as npz:
            for doc_id in sorted(index["docs"]):
                self.chunks.extend(index["docs"][doc_id]["chunks"])
                matrices.append(npz[doc_id])
        self.matrix = np.vstack(matrices).astype(np.float32)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.clip(norms, 1e-10, None)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        qv = np.asarray(embed([query])[0], dtype=np.float32)
        qv = qv / max(float(np.linalg.norm(qv)), 1e-10)
        sims = self.matrix @ qv
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "doc_id": self.chunks[i]["doc_id"],
                "section": self.chunks[i]["section"],
                "score": round(float(sims[i]), 4),
                "text": self.chunks[i]["text"],
            }
            for i in order
        ]

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict:
        """端到端单条问答。返回结构化记录（D4 评分 / D5 归因的直接输入）。"""
        t0 = time.perf_counter()
        retrieved = self.retrieve(question, top_k)
        t1 = time.perf_counter()
        context = "\n\n".join(
            f"[{i + 1}] {c['chunk_id']}（{c['section']}）\n{c['text']}"
            for i, c in enumerate(retrieved)
        )
        prompt = f"【参考资料】\n{context}\n\n【问题】{question}"
        reply = chat(prompt, system=SYSTEM_PROMPT, max_tokens=4096, temperature=0.2)
        t2 = time.perf_counter()
        return {
            "question": question,
            "answer": reply,
            "retrieved": retrieved,
            "top_k": top_k,
            "latency_retrieve_s": round(t1 - t0, 2),
            "latency_generate_s": round(t2 - t1, 2),
            "latency_total_s": round(t2 - t0, 2),
        }


# ---------------- CLI ----------------
def cmd_ask(question: str, top_k: int, show_context: bool, as_json: bool) -> None:
    pipe = RagPipeline()
    result = pipe.answer(question, top_k)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"问题：{question}（top-k={top_k}）")
    print("─" * 60)
    print("检索命中（按相似度降序）：")
    for c in result["retrieved"]:
        print(f"  {c['score']:.4f}  {c['chunk_id']}（{c['section']}）")
    if show_context:
        for c in result["retrieved"]:
            print(f"\n── {c['chunk_id']} ──\n{c['text'][:300]}")
    print("─" * 60)
    print(f"回答：\n{result['answer']}")
    print("─" * 60)
    print(f"延迟：检索 {result['latency_retrieve_s']}s ｜ 生成 "
          f"{result['latency_generate_s']}s ｜ 合计 {result['latency_total_s']}s")


def cmd_smoke() -> None:
    """从 eval_set 抽 4 条（事实/细节埋深/计算/拒答陷阱）跑端到端。"""
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))["eval_set"]
    picks = {e["id"]: e for e in eval_set}
    ids = ["F-01", "F-03", "R-10", "J-01"]
    pipe = RagPipeline()
    for qid in ids:
        item = picks[qid]
        print(f"\n{'═' * 60}\n[{qid}]（{item['type']}｜金标文档 {item['gold_doc_id']}）"
              f" {item['question']}")
        r = pipe.answer(item["question"])
        top_docs = [c["doc_id"] for c in r["retrieved"][:3]]
        hit = set(item["gold_doc_id"]) & set(top_docs) if item["gold_doc_id"] else set()
        flag = "✅ 金标进入 top-3" if hit else "❌ 金标未进 top-3" if item["gold_doc_id"] else "— 拒答题"
        print(f"top-3 检索：{top_docs} → {flag}")
        print(f"回答（前 240 字）：{r['answer'][:240]}")
        print(f"延迟：{r['latency_total_s']}s")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="RAG 复现 pipeline（D3）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="建索引（增量缓存）").add_argument(
        "--force", action="store_true", help="忽略缓存全量重建")
    p_ask = sub.add_parser("ask", help="单条问答")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p_ask.add_argument("--show-context", action="store_true")
    p_ask.add_argument("--json", action="store_true", dest="as_json")
    sub.add_parser("smoke", help="eval_set 抽测 4 条")
    args = ap.parse_args()

    if args.cmd == "build":
        cmd_build(force=args.force)
    elif args.cmd == "ask":
        cmd_ask(args.question, args.top_k, args.show_context, args.as_json)
    elif args.cmd == "smoke":
        cmd_smoke()


if __name__ == "__main__":
    main()
