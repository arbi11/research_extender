"""Level-3 construction of official examples from their article text.

An article build is a data-only assembly of reusable components (see
``components``), driven by the parameters recorded for its case, each with the
article passage it comes from, plus official library entries and FEMM defaults
where the article relies on them. It never reads the example's native model
file. The resulting typed model is compiled by ``reproduce.compile_model``, so
article builds, rebuilds and new designs share one compiler.
"""
from .components import assemble


def build(case):
    article = case.get("article") or {}
    if article.get("status") == "not_attempted":
        return {"status": "not_attempted", "reason": article["reason"]}
    if "assembly" not in article:
        raise ValueError("No article assembly is registered for case " + case["id"])
    parameters = {name: item["value"] for name, item in article["parameters"].items()}
    model, used = assemble(article["assembly"], parameters)
    return {"status": "built", "model": model, "parameters": article["parameters"], "components": used,
            "assumptions": article.get("assumptions", [])}
