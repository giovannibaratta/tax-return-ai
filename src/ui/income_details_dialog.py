"""Detailed inspector dialog for approved tax income ledger records."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictTaxIncomeRecord,
)


def _card_frame() -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setStyleSheet(
        "QFrame { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px; }"
    )
    return frame


def _section_header(title: str) -> QLabel:
    lbl = QLabel(title)
    f = QFont()
    f.setBold(True)
    f.setPointSize(11)
    lbl.setFont(f)
    lbl.setStyleSheet("color: #2c3e50; margin-bottom: 4px;")
    return lbl


class IncomeRecordDetailsDialog(QDialog):
    """Dialog displaying complete details for an approved TaxIncomeRecord."""

    def __init__(self, record: StrictTaxIncomeRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.record: StrictTaxIncomeRecord = record

        self.setWindowTitle(
            f"📑 Approved Tax Income Record #{record.id or 0} ({record.tax_year} - {record.jurisdiction.capitalize()})"
        )
        self.resize(780, 620)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        payload = self.record.payload

        layout.addWidget(self._build_banner_card(payload))

        if isinstance(payload, IrishEmploymentDetailSummaryPayload):
            layout.addWidget(self._build_employment_card(payload))
            layout.addWidget(self._build_financial_card(payload))
            layout.addWidget(self._build_prsi_card(payload))

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def _build_banner_card(self, payload: object) -> QFrame:
        banner = _card_frame()
        b_layout = QGridLayout(banner)
        b_layout.setContentsMargins(12, 12, 12, 12)
        b_layout.setSpacing(8)

        created_str = self.record.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.record.created_at else "N/A"
        emp_name = payload.employer_name if isinstance(payload, IrishEmploymentDetailSummaryPayload) else "N/A"

        b_layout.addWidget(QLabel("<b>Ledger Record ID:</b>"), 0, 0)
        b_layout.addWidget(QLabel(f"#{self.record.id or 0}"), 0, 1)
        b_layout.addWidget(QLabel("<b>Tax Year:</b>"), 0, 2)
        b_layout.addWidget(QLabel(str(self.record.tax_year)), 0, 3)

        b_layout.addWidget(QLabel("<b>Jurisdiction:</b>"), 1, 0)
        b_layout.addWidget(QLabel(self.record.jurisdiction.capitalize()), 1, 1)
        b_layout.addWidget(QLabel("<b>Income Type:</b>"), 1, 2)
        b_layout.addWidget(QLabel(self.record.income_type), 1, 3)

        b_layout.addWidget(QLabel("<b>Employer / Payer:</b>"), 2, 0)
        b_layout.addWidget(QLabel(f"<b>{emp_name}</b>"), 2, 1)
        b_layout.addWidget(QLabel("<b>Date Approved:</b>"), 2, 2)
        b_layout.addWidget(QLabel(created_str), 2, 3)
        return banner

    @staticmethod
    def _build_employment_card(payload: IrishEmploymentDetailSummaryPayload) -> QFrame:
        emp_card = _card_frame()
        e_layout = QGridLayout(emp_card)
        e_layout.setContentsMargins(12, 8, 12, 8)
        e_layout.setSpacing(8)

        start_str = payload.start_date.strftime("%Y-%m-%d") if payload.start_date else "N/A"
        end_str = payload.end_date.strftime("%Y-%m-%d") if payload.end_date else "Ongoing"

        e_layout.addWidget(_section_header("🏢 Employment / Job Details"), 0, 0, 1, 4)
        e_layout.addWidget(QLabel("<b>Employer Reg No (ERN):</b>"), 1, 0)
        e_layout.addWidget(QLabel(payload.employer_registration_number or "N/A"), 1, 1)
        e_layout.addWidget(QLabel("<b>Employment ID:</b>"), 1, 2)
        e_layout.addWidget(QLabel(payload.employment_id or "N/A"), 1, 3)

        e_layout.addWidget(QLabel("<b>Start Date:</b>"), 2, 0)
        e_layout.addWidget(QLabel(start_str), 2, 1)
        e_layout.addWidget(QLabel("<b>End Date:</b>"), 2, 2)
        e_layout.addWidget(QLabel(end_str), 2, 3)
        return emp_card

    @staticmethod
    def _build_financial_card(payload: IrishEmploymentDetailSummaryPayload) -> QFrame:
        fin_card = _card_frame()
        f_layout = QGridLayout(fin_card)
        f_layout.setContentsMargins(12, 8, 12, 8)
        f_layout.setSpacing(8)

        f_layout.addWidget(_section_header("💰 Tax & Deductions Breakdown"), 0, 0, 1, 4)

        f_layout.addWidget(QLabel("<b>Gross Pay:</b>"), 1, 0)
        f_layout.addWidget(QLabel(f"€{payload.gross_pay_eur:,.2f}"), 1, 1)

        tax_pay_str = f"€{payload.pay_for_income_tax_eur:,.2f}" if payload.pay_for_income_tax_eur is not None else "N/A"
        f_layout.addWidget(QLabel("<b>Pay for Income Tax:</b>"), 1, 2)
        f_layout.addWidget(QLabel(tax_pay_str), 1, 3)

        f_layout.addWidget(QLabel("<b>Income Tax / PAYE Paid:</b>"), 2, 0)
        f_layout.addWidget(QLabel(f"<font color='#c0392b'><b>€{payload.income_tax_paid_eur:,.2f}</b></font>"), 2, 1)

        bik_str = f"€{payload.taxable_benefits_eur:,.2f}" if payload.taxable_benefits_eur is not None else "N/A"
        f_layout.addWidget(QLabel("<b>Taxable Benefits (BIK):</b>"), 2, 2)
        f_layout.addWidget(QLabel(bik_str), 2, 3)

        usc_pay_str = f"€{payload.pay_for_usc_eur:,.2f}" if payload.pay_for_usc_eur is not None else "N/A"
        f_layout.addWidget(QLabel("<b>Pay for USC:</b>"), 3, 0)
        f_layout.addWidget(QLabel(usc_pay_str), 3, 1)

        f_layout.addWidget(QLabel("<b>USC Paid:</b>"), 3, 2)
        f_layout.addWidget(QLabel(f"<font color='#d35400'><b>€{payload.usc_paid_eur:,.2f}</b></font>"), 3, 3)

        f_layout.addWidget(QLabel("<b>Employee PRSI Paid:</b>"), 4, 0)
        f_layout.addWidget(QLabel(f"<font color='#2980b9'><b>€{payload.prsi_paid_eur:,.2f}</b></font>"), 4, 1)

        empr_prsi_str = (
            f"€{payload.employer_prsi_paid_eur:,.2f}" if payload.employer_prsi_paid_eur is not None else "N/A"
        )
        f_layout.addWidget(QLabel("<b>Employer PRSI Paid:</b>"), 4, 2)
        f_layout.addWidget(QLabel(empr_prsi_str), 4, 3)

        lpt_str = f"€{payload.lpt_deducted_eur:,.2f}" if payload.lpt_deducted_eur is not None else "N/A"
        f_layout.addWidget(QLabel("<b>LPT Deducted:</b>"), 5, 0)
        f_layout.addWidget(QLabel(lpt_str), 5, 1)
        return fin_card

    @staticmethod
    def _build_prsi_card(payload: IrishEmploymentDetailSummaryPayload) -> QFrame:
        prsi_card = _card_frame()
        p_layout = QVBoxLayout(prsi_card)
        p_layout.setContentsMargins(12, 8, 12, 8)
        p_layout.setSpacing(6)

        p_layout.addWidget(_section_header("🏥 PRSI Contribution Classes"))

        if payload.prsi_classes:
            prsi_table = QTableWidget()
            prsi_table.setColumnCount(2)
            prsi_table.setHorizontalHeaderLabels(["PRSI Class", "Number of Insurable Weeks"])
            prsi_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            prsi_table.setRowCount(len(payload.prsi_classes))
            for r_idx, c_entry in enumerate(payload.prsi_classes):
                item_class = QTableWidgetItem(c_entry.prsi_class)
                item_class.setFlags(Qt.ItemFlag.ItemIsEnabled)
                item_weeks = QTableWidgetItem(f"{c_entry.insurable_weeks} weeks")
                item_weeks.setFlags(Qt.ItemFlag.ItemIsEnabled)
                prsi_table.setItem(r_idx, 0, item_class)
                prsi_table.setItem(r_idx, 1, item_weeks)
            prsi_table.setFixedHeight(90)
            p_layout.addWidget(prsi_table)
        else:
            p_summary = (
                f"Primary Class: <b>{payload.prsi_class or 'N/A'}</b> | Weeks: <b>{payload.prsi_weeks or 'N/A'}</b>"
            )
            p_layout.addWidget(QLabel(p_summary))
        return prsi_card
