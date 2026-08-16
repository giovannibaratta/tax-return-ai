from pydantic import BaseModel
from sqlmodel import Session

from backend.db_manager import DatabaseManager
from backend.db_models import FinancialRecord
from backend.ingestion.parser import BasePDFParser, ParsedPage
from backend.llm.pydantic_ai_runner import PydanticAIRunner, T


def insert_financial_record(db_manager: DatabaseManager, record: FinancialRecord) -> FinancialRecord:
    """Insert a transaction record using SQLModel (used for testing)."""
    with Session(db_manager.engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        return record


class DummyMockPydanticRunner(PydanticAIRunner):
    """Mock runner returning validated structured instances or strings directly."""

    def __init__(self, output_obj: BaseModel | str) -> None:
        self._output_obj = output_obj

    @property
    def model_name(self) -> str:
        return "DummyMock"

    def complete(self, prompt: str, system_instruction: str = "") -> str:
        if isinstance(self._output_obj, str):
            return self._output_obj
        return self._output_obj.model_dump_json()

    def complete_structured(
        self,
        prompt: str,
        schema_cls: type[T],
        system_instruction: str = "",
    ) -> T:
        if isinstance(self._output_obj, str):
            return schema_cls.model_validate_json(self._output_obj)
        if isinstance(self._output_obj, schema_cls):
            return self._output_obj
        return schema_cls.model_validate(self._output_obj.model_dump())


class DummyMockPDFParser(BasePDFParser):
    """Mock PDF Parser returning static text pages."""

    @classmethod
    def parse_pdf(cls, file_path: str, force_parsing: bool = False, **kwargs: object) -> list[ParsedPage]:
        return [
            ParsedPage(
                page_number=1,
                combined_content=(
                    "Revenue Employment Detail Summary 2025\n"
                    "Employer: Acme Tech Ltd (ERN: 9876543A)\n"
                    "Gross Pay: €85,000.00\n"
                    "Income Tax: €22,000.00\n"
                    "USC: €3,500.00\n"
                    "PRSI: €3,400.00\n"
                    "Weeks: 52"
                ),
            )
        ]
