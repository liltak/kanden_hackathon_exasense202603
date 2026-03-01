"""Training data formatter for Qwen2.5-VL fine-tuning.

Converts simulation results + images into ChatML format JSONL
compatible with Unsloth training pipeline.
"""

import base64
import json
import logging
import random
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

SYSTEM_MESSAGE = (
    "あなたはExaSenseの太陽光発電分析AIアシスタントです。"
    "工場や施設の太陽光パネル設置に関する専門的な分析を行います。"
    "回答は常にMarkdownフォーマットで構造化してください。"
)


# --- Prompt Variations for Data Augmentation ---

PANEL_PLACEMENT_PROMPTS = [
    "この日射量ヒートマップを分析し、太陽光パネルの最適な設置場所を提案してください。",
    "画像の日射量分布を基に、ソーラーパネルの設置優先エリアをランキングしてください。",
    "ヒートマップの分析結果から、パネル設置に最も適した屋根エリアを特定してください。",
    "この施設の日射量データを見て、太陽光パネルの配置計画を作成してください。",
    "日射量マップを基にした太陽光パネル設置の最適化レポートを作成してください。",
]

FACILITY_ANALYSIS_PROMPTS = [
    "この3Dモデルのスクリーンショットを分析し、施設の太陽光発電ポテンシャルを評価してください。",
    "施設の3D画像から、屋根の形状と太陽光パネル設置の適合性を評価してください。",
    "この建物の3Dモデルを見て、太陽光発電設備の設置可能エリアと制約を分析してください。",
    "3Dモデル画像を基に、この工場の太陽光発電導入に関する総合分析レポートを作成してください。",
]

SEASONAL_OPTIMIZATION_PROMPTS = [
    "このサンパス図を分析し、季節ごとの太陽光パネル発電量の最適化提案を行ってください。",
    "太陽の軌跡データを基に、年間を通じた発電効率の最適化戦略を提案してください。",
    "サンパス図と日射データから、季節変動を考慮したパネル角度の最適化レポートを作成してください。",
]

ROI_ANALYSIS_PROMPTS = [
    "以下のシミュレーション結果を基に、太陽光パネル投資のROI分析レポートを作成してください。",
    "シミュレーションデータから、太陽光発電設備の投資回収計画を詳細に分析してください。",
    "この発電シミュレーション結果について、財務的な観点から投資判断のレポートを作成してください。",
]


# --- Data Classes ---


@dataclass
class TrainingSample:
    """A single training sample in ChatML format."""

    messages: list[dict]
    image_paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "images": self.image_paths,
            "metadata": self.metadata,
        }

    def validate(self) -> list[str]:
        """Validate sample format. Returns list of error messages."""
        errors = []
        if not self.messages:
            errors.append("messages list is empty")
            return errors

        roles = [m.get("role") for m in self.messages]

        if roles[0] not in ("system", "user"):
            errors.append(f"First message role must be 'system' or 'user', got '{roles[0]}'")

        if "assistant" not in roles:
            errors.append("No assistant message found")

        for i, msg in enumerate(self.messages):
            if "role" not in msg:
                errors.append(f"Message {i} missing 'role'")
            if "content" not in msg:
                errors.append(f"Message {i} missing 'content'")

        for path in self.image_paths:
            if not Path(path).exists():
                errors.append(f"Image not found: {path}")

        return errors


@dataclass
class DatasetStats:
    """Statistics for a formatted dataset."""

    total_samples: int
    samples_by_type: dict[str, int]
    avg_assistant_length: float
    total_images: int
    validation_errors: int


# --- Image Utilities ---


def image_to_base64(image_path: str | Path) -> str:
    """Convert image file to base64 string."""
    img = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_image_content(image_path: str | Path) -> dict:
    """Build image content block for ChatML message."""
    return {
        "type": "image",
        "image": f"file://{Path(image_path).resolve()}",
    }


# --- Sample Generators ---


def generate_panel_placement_sample(
    heatmap_image: str | Path,
    simulation_data: dict,
    expected_report: str,
    facility_image: str | Path | None = None,
    augment: bool = False,
) -> list[TrainingSample]:
    """Generate training samples for panel placement analysis.

    Args:
        heatmap_image: Path to irradiance heatmap image.
        simulation_data: JSON-serializable simulation results.
        expected_report: Expected markdown report (ground truth).
        facility_image: Optional facility photo/3D model screenshot.
        augment: Generate multiple prompt variations.

    Returns:
        List of TrainingSample objects.
    """
    prompts = PANEL_PLACEMENT_PROMPTS if augment else PANEL_PLACEMENT_PROMPTS[:1]
    context = json.dumps(simulation_data, ensure_ascii=False, indent=2)
    samples = []

    for prompt_text in prompts:
        user_content = []
        image_paths = []

        user_content.append(build_image_content(heatmap_image))
        image_paths.append(str(heatmap_image))

        if facility_image is not None:
            user_content.append(build_image_content(facility_image))
            image_paths.append(str(facility_image))

        user_content.append({
            "type": "text",
            "text": f"{prompt_text}\n\n## シミュレーションデータ\n```json\n{context}\n```",
        })

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": expected_report},
        ]

        samples.append(TrainingSample(
            messages=messages,
            image_paths=image_paths,
            metadata={"type": "panel_placement", "augmented": augment},
        ))

    return samples


def generate_facility_analysis_sample(
    model_screenshot: str | Path,
    facility_data: dict,
    expected_report: str,
    augment: bool = False,
) -> list[TrainingSample]:
    """Generate training samples for facility analysis.

    Args:
        model_screenshot: Path to 3D model screenshot.
        facility_data: Facility metadata (area, building info, etc.).
        expected_report: Expected markdown report.
        augment: Generate multiple prompt variations.

    Returns:
        List of TrainingSample objects.
    """
    prompts = FACILITY_ANALYSIS_PROMPTS if augment else FACILITY_ANALYSIS_PROMPTS[:1]
    context = json.dumps(facility_data, ensure_ascii=False, indent=2)
    samples = []

    for prompt_text in prompts:
        user_content = [
            build_image_content(model_screenshot),
            {
                "type": "text",
                "text": f"{prompt_text}\n\n## 施設データ\n```json\n{context}\n```",
            },
        ]

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": expected_report},
        ]

        samples.append(TrainingSample(
            messages=messages,
            image_paths=[str(model_screenshot)],
            metadata={"type": "facility_analysis", "augmented": augment},
        ))

    return samples


def generate_seasonal_optimization_sample(
    sunpath_image: str | Path,
    solar_data: dict,
    expected_report: str,
    augment: bool = False,
) -> list[TrainingSample]:
    """Generate training samples for seasonal optimization analysis.

    Args:
        sunpath_image: Path to sun path diagram image.
        solar_data: Solar position and irradiance data.
        expected_report: Expected markdown report.
        augment: Generate multiple prompt variations.

    Returns:
        List of TrainingSample objects.
    """
    prompts = SEASONAL_OPTIMIZATION_PROMPTS if augment else SEASONAL_OPTIMIZATION_PROMPTS[:1]
    context = json.dumps(solar_data, ensure_ascii=False, indent=2)
    samples = []

    for prompt_text in prompts:
        user_content = [
            build_image_content(sunpath_image),
            {
                "type": "text",
                "text": f"{prompt_text}\n\n## 太陽軌跡データ\n```json\n{context}\n```",
            },
        ]

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": expected_report},
        ]

        samples.append(TrainingSample(
            messages=messages,
            image_paths=[str(sunpath_image)],
            metadata={"type": "seasonal_optimization", "augmented": augment},
        ))

    return samples


def generate_roi_analysis_sample(
    images: list[str | Path],
    roi_data: dict,
    expected_report: str,
    augment: bool = False,
) -> list[TrainingSample]:
    """Generate training samples for ROI analysis.

    Args:
        images: Paths to relevant images (heatmap, facility, etc.).
        roi_data: ROI calculation results.
        expected_report: Expected markdown report.
        augment: Generate multiple prompt variations.

    Returns:
        List of TrainingSample objects.
    """
    prompts = ROI_ANALYSIS_PROMPTS if augment else ROI_ANALYSIS_PROMPTS[:1]
    context = json.dumps(roi_data, ensure_ascii=False, indent=2)
    samples = []

    for prompt_text in prompts:
        user_content = []
        image_paths = []

        for img in images:
            user_content.append(build_image_content(img))
            image_paths.append(str(img))

        user_content.append({
            "type": "text",
            "text": f"{prompt_text}\n\n## ROIデータ\n```json\n{context}\n```",
        })

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": expected_report},
        ]

        samples.append(TrainingSample(
            messages=messages,
            image_paths=image_paths,
            metadata={"type": "roi_analysis", "augmented": augment},
        ))

    return samples


# --- Dataset I/O ---


def save_dataset_jsonl(
    samples: list[TrainingSample],
    output_path: str | Path,
    shuffle: bool = True,
    seed: int = 42,
) -> DatasetStats:
    """Save training samples to JSONL format.

    Args:
        samples: List of training samples.
        output_path: Output JSONL file path.
        shuffle: Shuffle samples before saving.
        seed: Random seed for shuffling.

    Returns:
        DatasetStats with summary information.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validation_errors = 0
    valid_samples = []
    for sample in samples:
        errors = sample.validate()
        if errors:
            validation_errors += 1
            logger.warning("Invalid sample skipped: %s", errors)
        else:
            valid_samples.append(sample)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(valid_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    # Compute stats
    samples_by_type: dict[str, int] = {}
    total_assistant_len = 0
    total_images = 0
    for s in valid_samples:
        sample_type = s.metadata.get("type", "unknown")
        samples_by_type[sample_type] = samples_by_type.get(sample_type, 0) + 1
        total_images += len(s.image_paths)
        for msg in s.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                total_assistant_len += len(content) if isinstance(content, str) else 0

    avg_len = total_assistant_len / len(valid_samples) if valid_samples else 0

    stats = DatasetStats(
        total_samples=len(valid_samples),
        samples_by_type=samples_by_type,
        avg_assistant_length=round(avg_len, 0),
        total_images=total_images,
        validation_errors=validation_errors,
    )

    _print_stats(stats, output_path)
    return stats


def _print_stats(stats: DatasetStats, output_path: Path) -> None:
    """Print dataset statistics."""
    table = Table(title=f"Dataset: {output_path.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total samples", str(stats.total_samples))
    table.add_row("Total images", str(stats.total_images))
    table.add_row("Avg assistant length", f"{stats.avg_assistant_length:.0f} chars")
    table.add_row("Validation errors", str(stats.validation_errors))
    for sample_type, count in sorted(stats.samples_by_type.items()):
        table.add_row(f"  {sample_type}", str(count))
    console.print(table)


def load_dataset_jsonl(path: str | Path) -> list[TrainingSample]:
    """Load training samples from JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of TrainingSample objects.
    """
    path = Path(path)
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(TrainingSample(
                    messages=data["messages"],
                    image_paths=data.get("images", []),
                    metadata=data.get("metadata", {}),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Error at line %d: %s", line_num, e)

    console.print(f"[blue]Loaded {len(samples)} samples from {path}")
    return samples


def split_dataset(
    samples: list[TrainingSample],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    """Split samples into train and eval sets.

    Args:
        samples: All training samples.
        eval_ratio: Fraction for evaluation.
        seed: Random seed.

    Returns:
        Tuple of (train_samples, eval_samples).
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
    train = shuffled[:split_idx]
    eval_ = shuffled[split_idx:]

    console.print(f"[blue]Split: {len(train)} train / {len(eval_)} eval")
    return train, eval_


# --- Batch Generation from Simulation Results ---


def generate_from_simulation_results(
    results_dir: str | Path,
    output_dir: str | Path,
    augment: bool = True,
    eval_ratio: float = 0.1,
) -> tuple[Path, Path | None]:
    """Generate training data from ExaSense simulation output directory.

    Expects the results directory to contain:
    - irradiance_results.json
    - irradiance_heatmap.html (or .png screenshot)
    - sun_path.html (or .png screenshot)

    Args:
        results_dir: Path to simulation results.
        output_dir: Path to save JSONL files.
        augment: Enable prompt augmentation.
        eval_ratio: Eval split ratio.

    Returns:
        Tuple of (train_jsonl_path, eval_jsonl_path).
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Generating training data from {results_dir}")

    irradiance_path = results_dir / "irradiance_results.json"
    if not irradiance_path.exists():
        console.print(f"[red]irradiance_results.json not found in {results_dir}")
        raise FileNotFoundError(f"Missing: {irradiance_path}")

    irradiance_data = json.loads(irradiance_path.read_text(encoding="utf-8"))

    heatmap_images = list(results_dir.glob("irradiance_heatmap*.png"))
    sunpath_images = list(results_dir.glob("sun_path*.png"))

    all_samples: list[TrainingSample] = []

    if heatmap_images:
        report = _generate_panel_placement_report(irradiance_data)
        for img in heatmap_images:
            samples = generate_panel_placement_sample(
                heatmap_image=img,
                simulation_data=irradiance_data,
                expected_report=report,
                augment=augment,
            )
            all_samples.extend(samples)
        console.print(f"  [green]Panel placement: {len(heatmap_images)} images")

    if sunpath_images:
        report = _generate_seasonal_report(irradiance_data)
        solar_summary = {"irradiance_summary": _summarize_irradiance(irradiance_data)}
        for img in sunpath_images:
            samples = generate_seasonal_optimization_sample(
                sunpath_image=img,
                solar_data=solar_summary,
                expected_report=report,
                augment=augment,
            )
            all_samples.extend(samples)
        console.print(f"  [green]Seasonal optimization: {len(sunpath_images)} images")

    if heatmap_images:
        roi_data = _compute_roi_summary(irradiance_data)
        report = _generate_roi_report(roi_data)
        samples = generate_roi_analysis_sample(
            images=heatmap_images[:1],
            roi_data=roi_data,
            expected_report=report,
            augment=augment,
        )
        all_samples.extend(samples)
        console.print("  [green]ROI analysis: generated")

    if not all_samples:
        console.print("[yellow]No training samples generated. Check your simulation results.")
        return output_dir / "train.jsonl", None

    train_samples, eval_samples = split_dataset(all_samples, eval_ratio=eval_ratio)

    train_path = output_dir / "train.jsonl"
    save_dataset_jsonl(train_samples, train_path, shuffle=True)

    eval_path = None
    if eval_samples:
        eval_path = output_dir / "eval.jsonl"
        save_dataset_jsonl(eval_samples, eval_path, shuffle=False)

    return train_path, eval_path


def _summarize_irradiance(data: list[dict]) -> dict:
    """Create a summary of irradiance data."""
    if not data:
        return {}
    irr_values = [d["annual_irradiance_kwh_m2"] for d in data]
    return {
        "num_faces": len(data),
        "max_irradiance_kwh_m2": round(max(irr_values), 1),
        "min_irradiance_kwh_m2": round(min(irr_values), 1),
        "mean_irradiance_kwh_m2": round(sum(irr_values) / len(irr_values), 1),
        "total_area_m2": round(sum(d["area_m2"] for d in data), 1),
    }


def _compute_roi_summary(data: list[dict]) -> dict:
    """Compute basic ROI summary from irradiance data."""
    panel_efficiency = 0.20
    cost_per_kw = 250_000
    electricity_price = 30

    suitable = [d for d in data if d["annual_irradiance_kwh_m2"] > 800]
    total_area = sum(d["area_m2"] for d in suitable) * 0.7
    capacity_kw = total_area * 200 / 1000
    annual_gen = sum(
        d["annual_irradiance_kwh_m2"] * d["area_m2"] * 0.7 * panel_efficiency
        for d in suitable
    )

    return {
        "suitable_faces": len(suitable),
        "total_area_m2": round(total_area, 1),
        "capacity_kw": round(capacity_kw, 2),
        "annual_generation_kwh": round(annual_gen, 0),
        "installation_cost_jpy": round(capacity_kw * cost_per_kw),
        "annual_savings_jpy": round(annual_gen * electricity_price),
        "payback_years": round(
            (capacity_kw * cost_per_kw) / (annual_gen * electricity_price), 1
        )
        if annual_gen > 0
        else None,
    }


def _generate_panel_placement_report(data: list[dict]) -> str:
    """Generate a template panel placement report from simulation data."""
    summary = _summarize_irradiance(data)
    suitable = [d for d in data if d["annual_irradiance_kwh_m2"] > 800]
    suitable.sort(key=lambda x: x["annual_irradiance_kwh_m2"], reverse=True)

    lines = [
        "# 太陽光パネル設置場所分析レポート",
        "",
        "## 1. 施設概要",
        f"- 分析対象面数: {summary['num_faces']}面",
        f"- 総屋根面積: {summary['total_area_m2']} m²",
        f"- 平均日射量: {summary['mean_irradiance_kwh_m2']} kWh/m²/年",
        "",
        "## 2. 推奨設置エリア",
        f"- パネル設置適地: {len(suitable)}面",
    ]

    for i, face in enumerate(suitable[:5], 1):
        lines.append(
            f"  {i}. Face #{face['face_id']}: "
            f"{face['annual_irradiance_kwh_m2']:.0f} kWh/m²/年, "
            f"面積 {face['area_m2']:.1f} m²"
        )

    lines.extend([
        "",
        "## 3. 設置面積と容量",
        f"- 総設置可能面積: {sum(d['area_m2'] for d in suitable) * 0.7:.1f} m²（利用率70%）",
        f"- 推定設備容量: {sum(d['area_m2'] for d in suitable) * 0.7 * 200 / 1000:.1f} kW",
        "",
        "## 4. 注意事項",
        "- 日射量800 kWh/m²/年未満のエリアはパネル設置非推奨",
        "- 構造強度の確認が必要",
        "- 周辺建物による季節的な影の影響を考慮",
        "",
        "## 5. 総合評価",
        "- 推奨度: ★★★★☆（4/5）",
        "- 十分な日射量が確保できるエリアが複数確認されました",
    ])

    return "\n".join(lines)


def _generate_seasonal_report(data: list[dict]) -> str:
    """Generate a template seasonal optimization report."""
    return "\n".join([
        "# 季節別発電最適化レポート",
        "",
        "## 1. 年間日射パターン",
        "- 夏季（6-8月）: 日射量が最大。長い日照時間により高い発電量が期待できます",
        "- 冬季（12-2月）: 日射量は夏の約60%。太陽高度が低いため傾斜角の最適化が重要",
        "- 春秋（3-5月, 9-11月）: 安定した日射量。年間平均に近い発電量",
        "",
        "## 2. 最適パネル角度",
        "- 年間最適チルト角: 約30°（大阪の緯度に基づく）",
        "- 夏季推奨: 15-20°（太陽高度が高い）",
        "- 冬季推奨: 40-45°（太陽高度が低い）",
        "",
        "## 3. 発電量予測",
        "- 年間総発電量は施設全体のシミュレーション結果に基づきます",
        "- 月別変動係数: 夏季1.2-1.3倍、冬季0.6-0.7倍（年間平均比）",
        "",
        "## 4. 季節別運用提案",
        "- 夏季: 余剰電力の売電またはEV充電に活用",
        "- 冬季: 蓄電池との組み合わせでピークカット",
        "- 梅雨時期: 蓄電池からの放電を計画的に実施",
        "",
        "## 5. 年間スケジュール",
        "- 3月/9月: パネル角度調整（可動式の場合）",
        "- 5月/11月: パネル洗浄・点検",
        "- 8月: ピーク発電時の設備点検",
    ])


def _generate_roi_report(roi_data: dict) -> str:
    """Generate a template ROI analysis report."""
    return "\n".join([
        "# ROI分析レポート",
        "",
        "## 1. 投資概要",
        f"- 設備容量: {roi_data.get('capacity_kw', 'N/A')} kW",
        f"- 設置面積: {roi_data.get('total_area_m2', 'N/A')} m²",
        f"- 初期投資額: ¥{roi_data.get('installation_cost_jpy', 0):,.0f}",
        "",
        "## 2. 収益予測",
        f"- 年間発電量: {roi_data.get('annual_generation_kwh', 0):,.0f} kWh",
        f"- 年間削減額: ¥{roi_data.get('annual_savings_jpy', 0):,.0f}",
        "",
        "## 3. 財務指標",
        f"- 投資回収期間: {roi_data.get('payback_years', 'N/A')}年",
        "- 25年NPV: 詳細計算が必要",
        "",
        "## 4. リスク分析",
        "- パネル劣化: 年間0.5%の出力低下を想定",
        "- 電力価格変動: 年間2%の上昇を基準シナリオとして採用",
        "- 天候リスク: 過去10年の気象データに基づく変動を考慮",
        "",
        "## 5. 推奨事項",
        "- 投資回収期間が10年以内であれば、積極的な導入を推奨",
        "- 補助金・税制優遇の活用で初期投資を削減可能",
    ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format training data for VLM fine-tuning")
    parser.add_argument(
        "--results-dir",
        default="data/simulation_results",
        help="Simulation results directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/vlm_training",
        help="Output directory for JSONL files",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable prompt augmentation")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Eval split ratio")
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Validate an existing JSONL file",
    )
    args = parser.parse_args()

    if args.validate_only:
        console.print(f"[bold blue]Validating {args.validate_only}")
        samples = load_dataset_jsonl(args.validate_only)
        errors = 0
        for i, s in enumerate(samples):
            errs = s.validate()
            if errs:
                errors += 1
                console.print(f"[red]Sample {i}: {errs}")
        console.print(f"\n[bold]Validation: {len(samples)} samples, {errors} errors")
    else:
        train_path, eval_path = generate_from_simulation_results(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            augment=not args.no_augment,
            eval_ratio=args.eval_ratio,
        )
        console.print(f"\n[bold green]Training data saved:")
        console.print(f"  Train: {train_path}")
        if eval_path:
            console.print(f"  Eval: {eval_path}")
