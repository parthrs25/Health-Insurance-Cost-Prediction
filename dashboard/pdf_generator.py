import os
from fpdf import FPDF
from datetime import datetime

class PatientReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.set_text_color(31, 78, 121)
        self.cell(0, 10, "Medical Cost & Health Risk Assessment Report", ln=True, align="C")
        self.set_font("Arial", "I", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by ML & Quantile Regression", ln=True, align="C")
        self.line(10, 25, 200, 25)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_bytes(patient: dict, pred_cost: float, low_bound: float, high_bound: float) -> bytes:
    pdf = PatientReportPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. Patient Profile Summary", ln=True)

    pdf.set_font("Arial", "", 10)
    pdf.cell(95, 6, f"Age: {patient['age']} years", ln=False)
    pdf.cell(95, 6, f"Sex: {patient['sex'].capitalize()}", ln=True)
    pdf.cell(95, 6, f"BMI: {patient['bmi']} kg/m^2", ln=False)
    pdf.cell(95, 6, f"Children: {patient['children']}", ln=True)
    pdf.cell(95, 6, f"Smoking Status: {patient['smoker'].upper()}", ln=False)
    pdf.cell(95, 6, f"Region: {patient['region'].capitalize()}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Predicted Annual Insurance Expenditures", ln=True)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(39, 174, 96)
    pdf.cell(0, 8, f"Median Estimate: ${pred_cost:,.2f} USD", ln=True)

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"Confidence Range (10th - 90th percentile): ${low_bound:,.2f} - ${high_bound:,.2f} USD", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Risk Assessment & Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)

    if patient["smoker"] == "yes":
        pdf.multi_cell(0, 6, "- Smoking is the single primary multiplier for medical insurance costs. Quitting smoking can reduce annual charges by up to 65%.")
    else:
        pdf.multi_cell(0, 6, "- Non-smoker profile maintains a significantly lower baseline risk tier.")

    if patient["bmi"] >= 30:
        pdf.multi_cell(0, 6, "- BMI is in the obese category (>= 30). Combined with smoking status, this creates exponential expenditure increase.")

    return bytes(pdf.output())
