import os
import streamlit as st
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# ページ基本設定 & セッション状態の初期化
# ---------------------------------------------------------
st.set_page_config(
    page_title="データ収集AI (高度版)",
    page_icon="🔍",
    layout="wide"
)

# 検索履歴の保持用セッション
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "current_report" not in st.session_state:
    st.session_state.current_report = None

st.title("🔍 データ収集AIエージェント (高度分析版)")
st.caption("ディベート・プレゼン用に最適化されたデータ収集、参照元表示、ファクトチェック機能を提供します。")

# ---------------------------------------------------------
# サイドバー（設定 & 検索履歴エリア）
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ API設定")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Google AI Studioで取得した無料APIキーを入力")
    
    st.markdown("---")
    st.header("📜 検索履歴")
    if st.session_state.search_history:
        for idx, item in enumerate(reversed(st.session_state.search_history)):
            btn_label = f"[{item['mode_label']}] {item['topic'][:12]}..."
            if st.button(btn_label, key=f"hist_{idx}"):
                st.session_state.current_report = item
    else:
        st.info("過去の検索履歴はありません")

# ---------------------------------------------------------
# データ収集 & 分析AIクラス
# ---------------------------------------------------------
class AdvancedDataCollectorAI:
    def __init__(self, google_api_key: str):
        # 最新の推奨モデル gemini-3.8-flash を使用
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=google_api_key,
            temperature=0.2
        )
        self.search_tool = DuckDuckGoSearchRun()

    def _search_web(self, query: str) -> str:
        try:
            return self.search_tool.run(query)
        except Exception as e:
            return f"検索エラー: {str(e)}"

    def collect_and_analyze(self, topic: str, mode: str, stance: str = None):
        status_text = st.empty()
        
        # 1. 検索キーワード生成
        status_text.info("🔍 立場に応じた検索キーワードを生成中...")
        search_queries = self._generate_search_queries(topic, mode, stance)
        
        # 2. Web検索と参照元の保持
        collected_raw_data = ""
        search_sources = []
        for i, q in enumerate(search_queries, 1):
            status_text.info(f"🔎 Webを検索中 ({i}/{len(search_queries)}): {q}")
            res = self._search_web(q)
            collected_raw_data += f"\n--- 検索クエリ: {q} の結果 ---\n{res}\n"
            search_sources.append({"query": q, "snippet": res})

        status_text.info("📝 収集データからレポートを作成＆ファクトチェックを実行中...")

        # 3. モード別分析結果の出力
        if mode == "debate":
            report = self._format_for_debate(topic, stance, collected_raw_data)
        else:
            report = self._format_for_presentation(topic, collected_raw_data)

        # 4. ファクトチェック & エビデンス検証
        fact_check = self._analyze_fact_check(topic, collected_raw_data)

        status_text.empty()
        return {
            "topic": topic,
            "mode_label": "ディベート" if mode == "debate" else "プレゼン",
            "stance": stance,
            "report": report,
            "fact_check": fact_check,
            "sources": search_sources
        }

    def _generate_search_queries(self, topic: str, mode: str, stance: str) -> list[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "テーマと立場に沿った信頼性の高いデータを集めるためのWeb検索キーワードを3つ作成してください。Pythonのリスト形式（['kw1', 'kw2', 'kw3']）の文字列だけで返してください。"),
            ("user", "テーマ/議題: {topic}\nモード: {mode}\n立場: {stance}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            response = chain.invoke({"topic": topic, "mode": mode, "stance": stance if stance else "なし"})
            queries = eval(response.strip().replace("```python", "").replace("```", ""))
            if isinstance(queries, list):
                return queries
        except:
            pass
        return [f"{topic} 統計 データ", f"{topic} {stance if stance else ''} 根拠", f"{topic} 最新事例"]

    def _format_for_debate(self, topic: str, stance: str, raw_data: str) -> str:
        # ★ f-string を削除し、LangChainの標準変数構文に修正
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたはディベート戦略専門のデータアナリストです。
            ユーザーの立場【{stance}】に強力な説得力を持たせるためのデータとロジックを整理してください。

            【出力フォーマット】
            ### 1. 指定立場側の主立論とキー数値
            ### 2. 立場を裏付ける客観的エビデンス・統計データ
            ### 3. 相手側（反対立場）から想定される攻撃・反論
            ### 4. 相手の反論を即座に論破するための切り返しデータ (キラーファクト)
            """),
            ("user", "議題: {topic}\n自分の立場: {stance}\n\n収集データ:\n{raw_data}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"topic": topic, "stance": stance, "raw_data": raw_data})

    def _format_for_presentation(self, topic: str, raw_data: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたはプレゼン資料作成専門のリサーチャーです。
            聴衆に刺さるインパクトのある数値や、スライドに引用しやすい明確なデータを抽出してください。

            【出力フォーマット】
            ### 1. エグゼクティブサマリー (3行まとめ)
            ### 2. プレゼンで使えるキーデータ・比較グラフ案
            ### 3. スライド構成案用データ
            ### 4. 想定Q&Aとデータ裏付け
            """),
            ("user", "テーマ: {topic}\n\n収集データ:\n{raw_data}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"topic": topic, "raw_data": raw_data})

    def _analyze_fact_check(self, topic: str, raw_data: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            収集されたデータについてファクトチェックと客観的信頼性評価を行ってください。

            【出力フォーマット】
            ### 🛡️ データ信頼性総合評価: [強 / 中 / 弱]
            ### 🟢 強み・信頼性の高い点
            ### 🔴 弱点・反論されやすいポイント（注意点）
            ### ⚠️ 注意すべきバイアスや数値の切り取り・前提条件
            """),
            ("user", "テーマ: {topic}\n\n収集データ:\n{raw_data}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"topic": topic, "raw_data": raw_data})

# ---------------------------------------------------------
# メイン画面 UI
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 入力・条件設定")
    mode_input = st.radio("目的（モード）", ["ディベート用", "プレゼン用"])
    
    stance_input = None
    if mode_input == "ディベート用":
        topic_input = st.text_input("ディベートの議題", placeholder="例: 日本は週休3日制を法的義務化すべきか")
        stance_input = st.radio("あなたの立場", ["賛成 (肯定派)", "反対 (否定派)", "両論併記・中立分析"])
    else:
        topic_input = st.text_input("欲しいデータ（テーマ）", placeholder="例: 日本の週休3日制導入率と生産性の変化")

    submit_button = st.button("🚀 データを収集・分析する", use_container_width=True)

with col2:
    st.subheader("📊 分析結果")
    
    if submit_button:
        if not api_key:
            st.error("左側のサイドバーに Gemini API キーを入力してください。")
        elif not topic_input:
            st.warning("議題またはテーマを入力してください。")
        else:
            mode_code = "debate" if mode_input == "ディベート用" else "presentation"
            try:
                ai = AdvancedDataCollectorAI(google_api_key=api_key)
                result_data = ai.collect_and_analyze(topic_input, mode_code, stance_input)
                
                # セッション履歴に追加
                st.session_state.search_history.append(result_data)
                st.session_state.current_report = result_data
                st.success("データ収集およびファクトチェックが完了しました！")
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

    # レポート表示エリア（タブUI）
    if st.session_state.current_report:
        report_data = st.session_state.current_report
        
        st.markdown(f"#### 議題/テーマ: {report_data['topic']}")
        if report_data.get("stance"):
            st.info(f"指定立場: **{report_data['stance']}**")

        tab1, tab2, tab3 = st.tabs(["📄 整理レポート", "🛡️ ファクトチェック分析", "🔗 参照元・検索ソース"])
        
        with tab1:
            st.markdown(report_data["report"])
            st.download_button(
                label="📥 レポートをダウンロード (.txt)",
                data=report_data["report"],
                file_name=f"{report_data['topic']}_report.txt",
                mime="text/plain"
            )

        with tab2:
            st.markdown(report_data["fact_check"])

        with tab3:
            st.caption("AIが検索・抽出に利用したWebデータスニペット")
            for idx, src in enumerate(report_data["sources"], 1):
                with st.expander(f"ソース {idx}: クエリ「{src['query']}」"):
                    st.write(src["snippet"])
