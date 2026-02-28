"""VLM inference pipeline for solar panel analysis.

Accepts images + text prompts and generates structured markdown reports
using Qwen3.5-VL for facility analysis, panel placement, and ROI estimation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    import torch
    from PIL import Image

logger = logging.getLogger(__name__)
console = Console()


# --- Prompt Templates ---

PROMPT_PANEL_PLACEMENT = """\
あなたは太陽光パネル設置の専門コンサルタントです。
以下の画像を分析し、太陽光パネルの最適な設置場所を提案してください。

## 分析対象
- 画像1: 日射量ヒートマップ（赤=高日射量、青=低日射量）
- 画像2: 施設の写真/3Dモデル（提供されている場合）

## 追加データ
{context_data}

## 出力形式（Markdown）
以下の項目を含むレポートを作成してください：
1. **施設概要** - 建物の特徴と屋根の状態
2. **推奨設置エリア** - 優先順位付きの設置候補場所
3. **設置面積と容量** - 各エリアの推定面積とkW容量
4. **注意事項** - 影、障害物、構造的な制約
5. **総合評価** - 設置の推奨度（5段階）とコメント
"""

PROMPT_ENERGY_COST = """\
あなたはエネルギーコスト削減の専門アドバイザーです。
以下の施設データと画像を分析し、電力コスト削減の提案を行ってください。

## 分析データ
{context_data}

## 出力形式（Markdown）
1. **現状分析** - 現在のエネルギー消費パターン
2. **太陽光発電による削減効果** - 年間発電量と削減額の見積もり
3. **投資回収計画** - 初期投資、年間削減額、回収期間
4. **追加の削減施策** - 太陽光以外のコスト削減提案
5. **5年間のコスト予測** - 導入前後の比較表
"""

PROMPT_ROI_ANALYSIS = """\
あなたは太陽光発電投資の財務アナリストです。
以下のシミュレーション結果と画像を基に、詳細なROI分析レポートを作成してください。

## シミュレーション結果
{context_data}

## 出力形式（Markdown）
1. **投資概要** - 設備容量、初期投資額、設置面積
2. **収益予測** - 年間発電量、売電/自家消費の内訳
3. **財務指標** - 回収期間、NPV、IRR
4. **リスク分析** - パネル劣化、電力価格変動、天候リスク
5. **感度分析** - 楽観/基準/悲観シナリオ
6. **推奨事項** - 投資判断と優先順位
"""

PROMPT_GENERAL_QA = """\
あなたは工場の太陽光発電設備に関する専門家です。
以下の画像と情報に基づいて、ユーザーの質問に回答してください。

## 施設情報
{context_data}

## ユーザーの質問
{user_question}

Markdownフォーマットで詳細に回答してください。
"""

PROMPT_SEASONAL_OPTIMIZATION = """\
あなたは太陽光発電の季節別最適化の専門家です。
以下のサンパス図とデータを分析し、季節ごとの最適化提案を行ってください。

## 分析データ
{context_data}

## 出力形式（Markdown）
1. **年間日射パターン** - 季節ごとの日射量の特徴
2. **最適パネル角度** - 季節別の推奨チルト角
3. **発電量予測** - 月別・季節別の発電量見積もり
4. **季節別運用提案** - 蓄電池活用や需要調整の提案
5. **年間スケジュール** - メンテナンスと最適化のカレンダー
"""

PROMPT_TEMPLATES: dict[str, str] = {
    "panel_placement": PROMPT_PANEL_PLACEMENT,
    "energy_cost": PROMPT_ENERGY_COST,
    "roi_analysis": PROMPT_ROI_ANALYSIS,
    "general_qa": PROMPT_GENERAL_QA,
    "seasonal_optimization": PROMPT_SEASONAL_OPTIMIZATION,
}


# --- Data Classes ---


@dataclass
class InferenceRequest:
    """Input for VLM inference."""

    images: list[str | Path | Image.Image]
    prompt_template: str = "general_qa"
    context_data: str = ""
    user_question: str = ""
    custom_prompt: str | None = None
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class InferenceResult:
    """Structured output from VLM inference."""

    text: str
    prompt_template: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    vram_peak_mb: float
    images_count: int
    model_id: str = ""
    metadata: dict = field(default_factory=dict)


# --- Pipeline ---


class VLMPipeline:
    """VLM inference pipeline for solar panel analysis."""

    def __init__(
        self,
        model=None,
        processor=None,
        model_config: ModelConfig | None = None,
    ):
        self.model = model
        self.processor = processor
        self.model_config = model_config or ModelConfig()
        self._loaded = model is not None and processor is not None

    def load(self) -> None:
        """Load model and processor if not already loaded."""
        if self._loaded:
            console.print("[yellow]Model already loaded, skipping.")
            return
        self.model, self.processor = load_model(self.model_config)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call pipeline.load() first.")

    def _load_image(self, img: str | Path | Image.Image) -> Image.Image:
        """Load image from path or return existing PIL Image."""
        if isinstance(img, Image.Image):
            return img
        path = Path(img)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def _build_prompt(self, request: InferenceRequest) -> str:
        """Build the full prompt from template and request data."""
        if request.custom_prompt:
            return request.custom_prompt

        template = PROMPT_TEMPLATES.get(request.prompt_template)
        if template is None:
            raise ValueError(
                f"Unknown prompt template: {request.prompt_template}. "
                f"Available: {list(PROMPT_TEMPLATES.keys())}"
            )

        return template.format(
            context_data=request.context_data or "（データなし）",
            user_question=request.user_question or "",
        )

    def _build_messages(
        self,
        images: list[Image.Image],
        prompt: str,
    ) -> list[dict]:
        """Build ChatML messages for Qwen3.5-VL."""
        content: list[dict] = []

        for img in images:
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "あなたはExaSenseの太陽光発電分析AIアシスタントです。"
                            "工場や施設の太陽光パネル設置に関する専門的な分析を行います。"
                            "回答は常にMarkdownフォーマットで構造化してください。"
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on images + prompt.

        Args:
            request: Inference request with images and prompt.

        Returns:
            Structured inference result.
        """
        self._ensure_loaded()

        images = [self._load_image(img) for img in request.images]
        prompt = self._build_prompt(request)
        messages = self._build_messages(images, prompt)

        console.print(f"[blue]Running inference: template={request.prompt_template}")
        console.print(f"  Images: {len(images)}, max_tokens: {request.max_new_tokens}")

        vram_before = get_vram_usage_mb()
        t0 = time.perf_counter()

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.do_sample else 1.0,
                top_p=request.top_p if request.do_sample else 1.0,
                do_sample=request.do_sample,
            )

        generated_ids = output_ids[0, input_tokens:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        output_tokens = len(generated_ids)

        elapsed = time.perf_counter() - t0
        vram_peak = get_vram_usage_mb()

        console.print(f"[green]Inference complete: {elapsed:.1f}s")
        console.print(f"  Tokens: {input_tokens} in / {output_tokens} out")
        console.print(f"  Speed: {output_tokens / elapsed:.1f} tokens/s")

        logger.info(
            "Inference: %.1fs, %d in / %d out tokens, %.1f tok/s",
            elapsed,
            input_tokens,
            output_tokens,
            output_tokens / elapsed,
        )

        return InferenceResult(
            text=output_text,
            prompt_template=request.prompt_template,
            latency_seconds=round(elapsed, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            vram_peak_mb=round(vram_peak, 0),
            images_count=len(images),
            model_id=self.model_config.model_id,
        )

    def infer_stream(
        self,
        request: InferenceRequest,
    ) -> Generator[str, None, InferenceResult]:
        """Run streaming inference, yielding tokens as they are generated.

        Yields:
            Generated text chunks.

        Returns:
            Final InferenceResult (accessible via generator .value after StopIteration).
        """
        self._ensure_loaded()

        from transformers import TextIteratorStreamer
        from threading import Thread

        images = [self._load_image(img) for img in request.images]
        prompt = self._build_prompt(request)
        messages = self._build_messages(images, prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_tokens = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature if request.do_sample else 1.0,
            "top_p": request.top_p if request.do_sample else 1.0,
            "do_sample": request.do_sample,
            "streamer": streamer,
        }

        t0 = time.perf_counter()
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_text = ""
        output_tokens = 0
        for chunk in streamer:
            full_text += chunk
            output_tokens += 1
            yield chunk

        thread.join()
        elapsed = time.perf_counter() - t0
        vram_peak = get_vram_usage_mb()

        console.print(f"\n[green]Stream complete: {elapsed:.1f}s, {output_tokens} tokens")

        return InferenceResult(
            text=full_text,
            prompt_template=request.prompt_template,
            latency_seconds=round(elapsed, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            vram_peak_mb=round(vram_peak, 0),
            images_count=len(images),
            model_id=self.model_config.model_id,
        )

    def analyze_panel_placement(
        self,
        heatmap_image: str | Path | Image.Image,
        facility_image: str | Path | Image.Image | None = None,
        context_data: str = "",
    ) -> InferenceResult:
        """Analyze optimal solar panel placement."""
        images = [heatmap_image]
        if facility_image is not None:
            images.append(facility_image)

        return self.infer(
            InferenceRequest(
                images=images,
                prompt_template="panel_placement",
                context_data=context_data,
            )
        )

    def analyze_energy_cost(
        self,
        images: list[str | Path | Image.Image],
        context_data: str = "",
    ) -> InferenceResult:
        """Analyze energy cost reduction opportunities."""
        return self.infer(
            InferenceRequest(
                images=images,
                prompt_template="energy_cost",
                context_data=context_data,
            )
        )

    def analyze_roi(
        self,
        images: list[str | Path | Image.Image],
        context_data: str = "",
    ) -> InferenceResult:
        """Generate detailed ROI analysis report."""
        return self.infer(
            InferenceRequest(
                images=images,
                prompt_template="roi_analysis",
                context_data=context_data,
            )
        )

    def ask(
        self,
        images: list[str | Path | Image.Image],
        question: str,
        context_data: str = "",
    ) -> InferenceResult:
        """General Q&A about the facility."""
        return self.infer(
            InferenceRequest(
                images=images,
                prompt_template="general_qa",
                context_data=context_data,
                user_question=question,
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VLM inference on images")
    parser.add_argument("images", nargs="+", help="Image file paths")
    parser.add_argument(
        "--template",
        default="general_qa",
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Prompt template to use",
    )
    parser.add_argument("--question", default="", help="User question (for general_qa)")
    parser.add_argument("--context", default="", help="Additional context data")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--output", type=str, default=None, help="Save result to file")
    args = parser.parse_args()

    config = ModelConfig(quantize_4bit=args.quantize)
    pipeline = VLMPipeline(model_config=config)
    pipeline.load()

    request = InferenceRequest(
        images=args.images,
        prompt_template=args.template,
        context_data=args.context,
        user_question=args.question,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.stream:
        console.print("[bold blue]Streaming output:\n")
        gen = pipeline.infer_stream(request)
        try:
            for chunk in gen:
                console.print(chunk, end="")
        except StopIteration as e:
            result = e.value
        console.print()
    else:
        result = pipeline.infer(request)

    console.print("\n[bold]--- Result ---")
    console.print(result.text)
    console.print(f"\n[dim]Latency: {result.latency_seconds}s")
    console.print(f"[dim]Tokens: {result.input_tokens} in / {result.output_tokens} out")

    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
        console.print(f"[green]Saved to {args.output}")
