"""Side-by-side 3-Voter Diff & Merge Dialog for Income Documents."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import cast

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QWidget,
)

from backend.db_manager import DatabaseManager
from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictStagedTaxIncomeRecord,
)
from src.ui.voter_diff_dialog import GenericVoterDiffDialog, VoterDiffField

IncomeFieldGetter = Callable[[IrishEmploymentDetailSummaryPayload], str]


class IncomeVoterDiffDialog(GenericVoterDiffDialog):
    """Side-by-side 3-Voter Diff and Merge Dialog for Tax Income Records.

    Inherits table layout, 3-voter comparison, and color highlighting from GenericVoterDiffDialog.
    Provides domain-specific payload parsing and persistence for Irish EDS documents.
    """

    def __init__(
        self,
        staged_record: StrictStagedTaxIncomeRecord,
        db: DatabaseManager,
        parent: QWidget | None = None,
    ) -> None:
        self.staged_record: StrictStagedTaxIncomeRecord = staged_record
        self.db: DatabaseManager = db

        doc_name = staged_record.source_file_name or f"Record #{staged_record.id}"
        header_text = (
            f"<b>Source File:</b> {staged_record.source_file_name or 'N/A'} | "
            f"<b>Jurisdiction:</b> {staged_record.jurisdiction.upper()} | "
            f"<b>Tax Year:</b> {staged_record.tax_year} | "
            f"<b>Status:</b> <font color='#e67e22'>{staged_record.verification_status.upper()}</font>"
        )
        if staged_record.discrepancies:
            disc_html = "<br/>".join(f"• {d}" for d in staged_record.discrepancies)
            header_text += f"<br/><font color='#c0392b'><b>Discrepancies:</b><br/>{disc_html}</font>"

        fields = self._build_income_fields(staged_record)

        super().__init__(
            title=f"🏛️ Income Voter Consensus Diff - {doc_name}",
            header_html=header_text,
            fields=fields,
            accept_button_text="✅ Approve & Promote to Ledger",
            parent=parent,
        )

        # Add "Save Staged Only" button to the action bar
        self.btn_save_staged = QPushButton("💾 Save Staged Only")
        self.btn_save_staged.clicked.connect(self._save_staged_only)
        btn_layout = self.findChild(QHBoxLayout)
        if btn_layout:
            # Insert before the accept button (which is 2nd from right)
            btn_layout.insertWidget(btn_layout.count() - 2, self.btn_save_staged)

    @classmethod
    def _build_income_fields(cls, staged_record: StrictStagedTaxIncomeRecord) -> list[VoterDiffField]:
        """Build VoterDiffField list from staged record and candidate voter payloads."""
        specs: list[tuple[str, str, IncomeFieldGetter, bool]] = [
            ("tax_year", "Tax Year", lambda p: str(p.tax_year), False),
            ("employer_name", "Employer Name", lambda p: str(p.employer_name), False),
            (
                "employer_registration_number",
                "Employer Reg No (ERN)",
                lambda p: str(p.employer_registration_number or ""),
                False,
            ),
            ("employment_id", "Employment ID", lambda p: str(p.employment_id or ""), False),
            ("start_date", "Start Date", lambda p: p.start_date.isoformat() if p.start_date else "", False),
            ("end_date", "End Date", lambda p: p.end_date.isoformat() if p.end_date else "", False),
            ("gross_pay_eur", "Gross Pay (EUR)", lambda p: str(p.gross_pay_eur), True),
            ("pay_for_income_tax_eur", "Pay for Income Tax (EUR)", lambda p: str(p.pay_for_income_tax_eur or ""), True),
            ("income_tax_paid_eur", "Income Tax / PAYE (EUR)", lambda p: str(p.income_tax_paid_eur), True),
            ("taxable_benefits_eur", "Taxable Benefits (EUR)", lambda p: str(p.taxable_benefits_eur or ""), True),
            ("pay_for_usc_eur", "Pay for USC (EUR)", lambda p: str(p.pay_for_usc_eur or ""), True),
            ("usc_paid_eur", "USC (EUR)", lambda p: str(p.usc_paid_eur), True),
            ("prsi_paid_eur", "Employee PRSI (EUR)", lambda p: str(p.prsi_paid_eur), True),
            ("employer_prsi_paid_eur", "Employer PRSI (EUR)", lambda p: str(p.employer_prsi_paid_eur or ""), True),
            (
                "prsi_classes",
                "PRSI Classes (e.g. A1:12, M:0)",
                lambda p: (
                    ", ".join(f"{c.prsi_class}:{c.insurable_weeks}" for c in p.prsi_classes) if p.prsi_classes else ""
                ),
                False,
            ),
            ("prsi_class", "Primary PRSI Class", lambda p: str(p.prsi_class or ""), False),
            ("prsi_weeks", "Total PRSI Weeks", lambda p: str(p.prsi_weeks or ""), False),
            ("lpt_deducted_eur", "LPT Deducted (EUR)", lambda p: str(p.lpt_deducted_eur or ""), True),
        ]

        curr_payload = (
            cast(IrishEmploymentDetailSummaryPayload, staged_record.payload)
            if staged_record.payload is not None
            else None
        )
        voters = [cast(IrishEmploymentDetailSummaryPayload, v) for v in (staged_record.voter_outputs or [])]

        v1 = voters[0] if len(voters) >= 1 else None
        v2 = voters[1] if len(voters) >= 2 else None  # noqa: PLR2004
        v3 = voters[2] if len(voters) >= 3 else None  # noqa: PLR2004

        result: list[VoterDiffField] = []
        for key, label, getter, is_num in specs:
            result.append(
                VoterDiffField(
                    key=key,
                    label=label,
                    val_v1=getter(v1).strip() if v1 is not None else "",
                    val_v2=getter(v2).strip() if v2 is not None else "",
                    val_v3=getter(v3).strip() if v3 is not None else "",
                    initial_merged_val=getter(curr_payload).strip() if curr_payload is not None else "",
                    is_numeric=is_num,
                )
            )
        return result

    def extract_merged_payload(self) -> IrishEmploymentDetailSummaryPayload:
        """Extract and validate merged payload from table inputs."""
        merged = self.get_merged_values()
        data: dict[str, object] = {
            "income_type": "irish_employment_detail_summary",
        }
        prsi_classes_list: list[dict[str, object]] = []

        for key, text_val in merged.items():
            if key in (
                "gross_pay_eur",
                "pay_for_income_tax_eur",
                "income_tax_paid_eur",
                "taxable_benefits_eur",
                "pay_for_usc_eur",
                "usc_paid_eur",
                "prsi_paid_eur",
                "employer_prsi_paid_eur",
                "lpt_deducted_eur",
            ):
                data[key] = Decimal(text_val.replace(",", ".")) if text_val else None
            elif key in ("tax_year", "prsi_weeks"):
                data[key] = int(text_val) if text_val else None
            elif key in ("start_date", "end_date"):
                data[key] = datetime.fromisoformat(text_val) if text_val else None
            elif key == "prsi_classes":
                if text_val:
                    for part in text_val.split(","):
                        sub = part.strip().split(":")
                        if len(sub) == 2:  # noqa: PLR2004
                            prsi_classes_list.append(
                                {"prsi_class": sub[0].strip(), "insurable_weeks": int(sub[1].strip())}
                            )
            else:
                data[key] = text_val if text_val else None

        # Validate mandatory fields
        required_checks: list[tuple[str, str]] = [
            ("tax_year", "Tax Year"),
            ("employer_name", "Employer Name"),
            ("gross_pay_eur", "Gross Pay (EUR)"),
            ("income_tax_paid_eur", "Income Tax / PAYE (EUR)"),
            ("usc_paid_eur", "USC (EUR)"),
            ("prsi_paid_eur", "Employee PRSI (EUR)"),
        ]
        for key, label in required_checks:
            if data.get(key) is None:
                raise ValueError(f"Field '{label}' cannot be empty. Please pick a voter value or enter an amount.")

        if prsi_classes_list:
            data["prsi_classes"] = prsi_classes_list

        return IrishEmploymentDetailSummaryPayload.model_validate(data)

    def _on_accept_clicked(self) -> None:
        """Approve merged record and promote to approved ledger."""
        try:
            # Sync merged values from table
            super()._on_accept_clicked()

            assert self.staged_record.id is not None
            merged_payload = self.extract_merged_payload()
            self.staged_record.payload = merged_payload
            self.staged_record.tax_year = merged_payload.tax_year
            self.db.update_staged_tax_income_record(self.staged_record)

            app_id = self.db.approve_staged_tax_income_record(self.staged_record.id)
            QMessageBox.information(
                self,
                "Success",
                f"Staged record #{self.staged_record.id} approved and saved as Ledger Record #{app_id}!",
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save and approve record: {e}")

    def _approve_merged_record(self) -> None:
        """Compatibility alias for approve action."""
        self._on_accept_clicked()

    def _save_staged_only(self) -> None:
        """Save merged values into the staged record without promoting to ledger."""
        try:
            # Sync merged values from table
            self._merged_values = {}
            for row, spec in enumerate(self.fields):
                it = self.table.item(row, 4)
                self._merged_values[spec.key] = it.text().strip() if it else ""

            merged_payload = self.extract_merged_payload()
            self.staged_record.payload = merged_payload
            self.staged_record.tax_year = merged_payload.tax_year
            self.staged_record.verification_status = "pending_approval"
            self.db.update_staged_tax_income_record(self.staged_record)
            QMessageBox.information(self, "Success", f"Staged record #{self.staged_record.id} updated.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save changes: {e}")
