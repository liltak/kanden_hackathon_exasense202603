"""Celery task for VLM analysis pipeline (Phase 4).

Runs on GPU workers: Qwen3.5-VL inference for facility analysis.
Requires CUDA-enabled environment with transformers + vllm.
"""

from __future__ import annotations

import logging
import time

from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="exasense.run_vlm_analysis")
def run_vlm_analysis_task(
    self,
    image_paths: list[str],
    mesh_path: str | None = None,
    simulation_task_id: str | None = None,
    prompt: str | None = None,
) -> dict:
    """Run VLM analysis on facility images.

    Args:
        image_paths: List of image file paths to analyze.
        mesh_path: Optional path to 3D mesh for spatial context.
        simulation_task_id: Optional task ID to fetch simulation results.
        prompt: Custom analysis prompt. Uses default if None.

    Returns:
        Dict with status, analysis text, recommendations, elapsed_seconds.
    """
    t0 = time.time()

    self.update_state(
        state="PROGRESS",
        meta={"step": "loading_model", "progress": 0.1, "message": "Loading VLM model..."},
    )

    try:
        from ..vlm.inference import VLMInference

        vlm = VLMInference()

        self.update_state(
            state="PROGRESS",
            meta={"step": "analyzing", "progress": 0.3, "message": "Analyzing images..."},
        )

        # Build context from simulation results if available
        context = ""
        if simulation_task_id:
            context = _get_simulation_context(simulation_task_id)

        if prompt is None:
            prompt = (
                "この工場施設の画像を分析し、太陽光パネルの最適な設置場所を特定してください。"
                "屋根の形状、方角、遮蔽物の有無、構造的な制約を考慮して、"
                "具体的な設置提案を日本語で記述してください。"
            )

        if context:
            prompt = f"{prompt}\n\n【シミュレーション結果】\n{context}"

        analysis = vlm.analyze(
            image_paths=image_paths,
            prompt=prompt,
        )

        self.update_state(
            state="PROGRESS",
            meta={"step": "generating_report", "progress": 0.8, "message": "Generating report..."},
        )

        # Extract recommendations
        recommendations = vlm.extract_recommendations(analysis)

        elapsed = time.time() - t0

        return {
            "status": "complete",
            "analysis": analysis,
            "recommendations": recommendations,
            "n_images": len(image_paths),
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        logger.exception("VLM analysis failed")
        return {"status": "failed", "error": str(e), "elapsed_seconds": time.time() - t0}


def _get_simulation_context(task_id: str) -> str:
    """Fetch simulation results for VLM context."""
    try:
        import os

        import asyncpg

        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return ""

        # Sync query for Celery worker context
        import asyncio

        async def _fetch():
            # Convert async URL to sync format for direct query
            conn_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            conn = await asyncpg.connect(conn_url)
            try:
                row = await conn.fetchrow(
                    "SELECT roi_data FROM simulations WHERE task_id = $1",
                    task_id,
                )
                return row
            finally:
                await conn.close()

        row = asyncio.run(_fetch())
        if row and row["roi_data"]:
            import json

            roi = json.loads(row["roi_data"]) if isinstance(row["roi_data"], str) else row["roi_data"]
            return (
                f"設置可能容量: {roi['total_capacity_kw']:.1f}kW, "
                f"年間発電量: {roi['total_annual_generation_kwh']:,.0f}kWh, "
                f"投資回収: {roi['overall_payback_years']:.1f}年"
            )
    except Exception:
        pass
    return ""
