from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
import csv
import io

from thread_store import load_thread
from artifact_generator import generate_evidence_table, generate_annotated_bib, generate_synthesis_memo

router = APIRouter(prefix="/api", tags=["export"])


@router.get("/export/{format}")
def export_artifact(format: str, artifact_type: str, thread_id: str):
    """Export artifact as Markdown, CSV, or PDF."""
    thread = load_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if artifact_type == "evidence-table":
        data = generate_evidence_table(thread)
        rows = data.get("rows", [])
        if format == "md":
            return Response(content=data.get("markdown", ""), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="evidence_table_{thread_id[:8]}.md"'})
        if format == "csv":
            return _evidence_table_csv(rows, thread_id)
        if format == "pdf":
            return _to_pdf(data.get("markdown", ""), f"evidence_table_{thread_id[:8]}.pdf")

    if artifact_type == "annotated-bib":
        data = generate_annotated_bib(thread)
        entries = data.get("entries", [])
        if format == "md":
            return Response(content=data.get("markdown", ""), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="annotated_bib_{thread_id[:8]}.md"'})
        if format == "csv":
            return _annotated_bib_csv(entries, thread_id)
        if format == "pdf":
            return _to_pdf(data.get("markdown", ""), f"annotated_bib_{thread_id[:8]}.pdf")

    if artifact_type == "synthesis-memo":
        data = generate_synthesis_memo(thread)
        content = data.get("markdown", data.get("content", ""))
        if format == "md":
            return Response(content=content, media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="synthesis_memo_{thread_id[:8]}.md"'})
        if format == "csv":
            raise HTTPException(status_code=400, detail="Synthesis memo is not tabular; use MD or PDF")
        if format == "pdf":
            return _to_pdf(content, f"synthesis_memo_{thread_id[:8]}.pdf")

    raise HTTPException(status_code=400, detail=f"Unknown format or artifact_type: {format}, {artifact_type}")


def _evidence_table_csv(rows: list, thread_id: str) -> Response:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Claim", "Evidence snippet", "source_id", "chunk_id", "Confidence", "Notes"])
    for r in rows:
        w.writerow([
            r.get("claim", ""),
            r.get("evidence_snippet", ""),
            r.get("source_id", ""),
            r.get("chunk_id", ""),
            r.get("confidence", ""),
            r.get("notes", ""),
        ])
    return Response(content=out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="evidence_table_{thread_id[:8]}.csv"'})


def _annotated_bib_csv(entries: list, thread_id: str) -> Response:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["source_id", "title", "claim", "method", "limitations", "why_it_matters"])
    for e in entries:
        w.writerow([
            e.get("source_id", ""),
            e.get("title", ""),
            e.get("claim", ""),
            e.get("method", ""),
            e.get("limitations", ""),
            e.get("why_it_matters", ""),
        ])
    return Response(content=out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="annotated_bib_{thread_id[:8]}.csv"'})


def _to_pdf(markdown_content: str, filename: str) -> Response:
    from markdown import markdown

    html_content = markdown(markdown_content, extensions=["tables"])
    html_doc = f"<html><body style='font-family:sans-serif;margin:1em;'>{html_content}</body></html>"

    # Try WeasyPrint first (best quality, requires system libs: brew install pango on macOS)
    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html_doc).write_pdf()
    except (ImportError, OSError):
        # Fallback: fpdf2 (pure Python, no system deps) when WeasyPrint unavailable
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.write_html(html_doc)
            pdf_bytes = bytes(pdf.output())
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF export requires weasyprint (brew install pango) or fpdf2. Run: pip install fpdf2",
            )

    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})

