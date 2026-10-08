"""
Quick check that the app works with Claude before a full run. Costs a few cents.

Three checks:
  1. Plain agent call           (key, model name and LiteLLM connection work)
  2. Structured output          (Sentiment Analyst returns a valid SentimentResult)
  3. Full manager run           (tool calls, human approval and the final report work together)

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python smoke_test.py
"""
import asyncio
import os
import traceback

from agents import Agent, Runner

import review_agents as ra


async def auto_approve(tool_name: str, args: dict):
    # Stand-in for the human reviewer: approve every request
    print(f"    approved: {tool_name}")
    return "approve", None


async def check_plain_call() -> str:
    agent = Agent(name="Ping", instructions="Reply with the single word: pong", model=ra.make_model(ra.MODEL))
    result = await Runner.run(agent, "ping")
    return f"reply = {result.final_output!r}"


async def check_structured_output(ctx: ra.AppContext) -> str:
    # Pick the first 1-star review so the analyst has a complaint to find
    review = next(r for r in ctx.reviews.values() if int(r["Review Rating"]) == 1)
    text = f"Title: {review['Review Title']}\nRating: {review['Review Rating']}\n{review['Review Content']}"
    result = await Runner.run(ra.sentiment_analyst, text)
    out = ra.parse_output(result.final_output, ra.SentimentResult)
    assert isinstance(out, ra.SentimentResult), f"could not parse JSON from: {str(out)[:200]}"
    return f"{review['Review Id']}: sentiment={out.overall_sentiment}, complaints={[c.issue_category for c in out.complaints]}"


async def check_full_run(ctx: ra.AppContext) -> str:
    review_id = next(rid for rid, r in ctx.reviews.items() if int(r["Review Rating"]) == 1)
    rec = await ra.process_review(review_id, ctx, auto_approve)
    ra.print_report(rec)
    assert isinstance(rec.final_output, ra.ReviewReport), "manager did not return a ReviewReport"
    return f"{review_id}: tickets={len(ctx.tickets)}, replies={len(ctx.replies)}, guardrail blocks={len(ctx.guardrail_blocks)}"


async def check_ranking(ctx: ra.AppContext) -> str:
    ids = list(ctx.reviews)[:8]
    rec = await ra.run_with_approvals("Which product issues should we fix first? Rank the issues in these reviews: " + ", ".join(ids),
                                      ctx, auto_approve)
    ra.print_report(rec)
    assert isinstance(rec.final_output, ra.PriorityReport), "manager did not return a PriorityReport"
    assert [t["tool"] for t in rec.trace].count("rank_issues") == 1, "rank_issues should be called exactly once"
    return f"top issue = {rec.final_output.ranking[0].issue_category if rec.final_output.ranking else 'none'}"


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")
    ctx = ra.AppContext(reviews=ra.load_reviews())
    print(f"Model: {ra.MODEL}\n")
    results = {}
    for name, check in [("1 plain call", lambda: check_plain_call()),
                        ("2 structured output", lambda: check_structured_output(ctx)),
                        ("3 full manager run", lambda: check_full_run(ctx)),
                        ("4 priority ranking", lambda: check_ranking(ctx))]:
        try:
            print(f"[{name}]")
            print("    PASS -", await check())
            results[name] = True
        except Exception:
            print("    FAIL")
            traceback.print_exc()
            results[name] = False
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {n}" for n, ok in results.items()))


if __name__ == "__main__":
    asyncio.run(main())
