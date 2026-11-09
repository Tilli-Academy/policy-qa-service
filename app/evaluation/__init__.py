"""Offline/opt-in RAG evaluation package (Milestone 6).

This package is a *consumer* of the application's public APIs
(:class:`PolicyRetriever`, :func:`answer_question`) and is never imported by the
runtime application (``app.main``/routes/services), so evaluation stays fully
separate from serving.

Public surface:
    - ``dataset``: load/validate the labeled evaluation set.
    - ``metrics``: pure retrieval/answer metric functions.
    - ``retrieval_eval``: offline retrieval quality (no Groq).
    - ``answer_eval``: answer quality (one Groq call per answerable item; opt-in).
"""
