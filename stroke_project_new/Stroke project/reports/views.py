from django.shortcuts import get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.http import FileResponse, Http404
from django.conf import settings

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer
)
from reportlab.lib.units import inch

import os

from detection.models import StrokeDetectionResult
from .models import DiagnosticReport


class GenerateReportView(LoginRequiredMixin, View):
    """Generate and download PDF diagnostic report"""

    def get(self, request, pk):
        # 🔐 Get detection result (only owner's scan)
        result = get_object_or_404(
            StrokeDetectionResult,
            pk=pk,
            ct_scan__uploaded_by=request.user
        )

        # ✅ Always ensure report exists
        try:
            report = result.diagnosticreport
        except DiagnosticReport.DoesNotExist:
            report = self.create_pdf_report(result)

        # 🛑 Final safety check
        if not report or not report.report_file:
            raise Http404("Diagnostic report could not be generated")

        return FileResponse(
            open(report.report_file.path, 'rb'),
            as_attachment=True,
            filename=f"stroke_report_{result.ct_scan.patient_id}.pdf"
        )

    def create_pdf_report(self, result):
        """Create PDF report and save DB record"""

        # 📁 Ensure media/reports folder exists
        reports_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        # 📄 File paths
        filename = f"reports/diagnostic_report_{result.id}.pdf"
        filepath = os.path.join(settings.MEDIA_ROOT, filename)

        # 🧾 Create PDF
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # 🏷 Title
        elements.append(
            Paragraph("Stroke Detection Diagnostic Report", styles["Title"])
        )
        elements.append(Spacer(1, 0.3 * inch))

        # 👤 Patient Information
        patient_data = [
            ["Patient Information", ""],
            ["Patient ID:", result.ct_scan.patient_id],
            ["Patient Name:", result.ct_scan.patient_name],
            ["Age:", str(result.ct_scan.patient_age)],
            ["Gender:", result.ct_scan.patient_gender],
            ["Scan Date:", result.ct_scan.upload_date.strftime("%Y-%m-%d %H:%M")],
        ]

        patient_table = Table(patient_data, colWidths=[2 * inch, 4 * inch])
        patient_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ]))

        elements.append(patient_table)
        elements.append(Spacer(1, 0.3 * inch))

        # 🧠 Detection Results
        # Prefer an `accuracy` attribute if present; otherwise use confidence_score
        accuracy_value = getattr(result, 'accuracy', None)
        if accuracy_value is None:
            accuracy_value = getattr(result, 'confidence_score', 0.0)

        result_data = [
            ["Detection Results", ""],
            ["Prediction:", result.prediction.upper()],
            ["Accuracy:", f"{float(accuracy_value):.2f}%"],
            ["Processing Time:", f"{result.processing_time:.2f} seconds"],
            ["Model Version:", result.model_version],
            ["Prediction Date:", result.prediction_date.strftime("%Y-%m-%d %H:%M")],
        ]

        result_table = Table(result_data, colWidths=[2 * inch, 4 * inch])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ]))

        elements.append(result_table)

        # 🏗 Build PDF
        doc.build(elements)

        # 💾 Save DB record (OneToOne safe)
        report = DiagnosticReport.objects.create(
            detection_result=result,
            report_file=filename
        )

        return report
