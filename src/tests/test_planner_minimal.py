from __future__ import annotations

from src.agents.planner import plan_candidates
from src.tools.registry import build_registry
from src.utils.parsing import parse_first_message


FIRST_MESSAGE = """
سؤال: «مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده.»

پلاگین‌ها:
- partDeltaPluginHttpBasedAtlasReadByKey
  توضیحات: فراخوانی HTTP برای خواندن رکوردهای اطلس بر اساس کلید یا namespace.
  ورودی‌ها: namespace (اجباری)، project (اختیاری)، headers (اختیاری)، throwError (اختیاری)
  خروجی: data.result[*].keys شامل آرایه‌ای از کلیدها.
- membasedAtlasKeyStreamAggregator
  توضیحات: جمع مقادیر عددی انتهایی کلیدهایی که با name داده شده شروع می‌شوند یا index مشخصی دارند.
  ورودی‌ها: name یا index (حداقل یکی)، جریان ورودی از HttpBasedAtlasReadByKey.
  خروجی: مقدار عددی مجموع.
"""


def test_persian_price_sum_plan():
    parsed = parse_first_message(FIRST_MESSAGE)
    registry = build_registry(parsed["plugin_docs"])
    result = plan_candidates(parsed["question"], registry, k=3)

    assert result["chosen"] == 0
    top_plan = result["candidates"][result["chosen"]].plan
    assert top_plan == ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"]
    assert len(result["candidates"]) == 3
