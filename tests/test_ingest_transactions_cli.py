from unittest.mock import MagicMock, patch

from backend.ingestion.ingest_transactions import main


def test_ingest_transactions_file_flag(tmp_path):
    """Test specifying a single PDF file via --file flag."""
    # Given: A dummy PDF file inside a temporary directory structure
    pdf_dir = tmp_path / "raw_sources" / "records" / "ireland" / "degiro"
    pdf_dir.mkdir(parents=True)
    pdf_file = pdf_dir / "custom_statement.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")
    db_file = tmp_path / "test.db"

    # When: Running main with --file pointing to single PDF
    test_args = [
        "ingest_transactions.py",
        "ingest",
        "--file",
        str(pdf_file),
        "--mode",
        "mock",
        "--test",
        "--db",
        str(db_file),
    ]

    with patch("sys.argv", test_args):
        with patch("backend.ingestion.ingest_transactions.TransactionPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.ingest_records_document.return_value = ("approved", [])
            mock_pipeline_cls.return_value = mock_pipeline

            main()

            # Then: Document should be ingested with correct path, jurisdiction, and provider
            mock_pipeline.ingest_records_document.assert_called_once()
            called_doc = mock_pipeline.ingest_records_document.call_args.kwargs["doc"]
            assert called_doc.file_path == str(pdf_file)
            assert called_doc.account_country == "ireland"
            assert called_doc.provider == "degiro"


def test_ingest_transactions_folder_flag(tmp_path):
    """Test specifying a folder via --folder flag."""
    # Given: A directory structure with two PDF files
    sub_dir = tmp_path / "reports"
    sub_dir.mkdir()
    pdf1 = sub_dir / "doc1.pdf"
    pdf2 = sub_dir / "doc2.pdf"
    pdf1.write_bytes(b"%PDF-1.4 content 1")
    pdf2.write_bytes(b"%PDF-1.4 content 2")
    db_file = tmp_path / "test.db"

    # When: Running main with --folder flag
    test_args = [
        "ingest_transactions.py",
        "ingest",
        "--folder",
        str(sub_dir),
        "--mode",
        "mock",
        "--test",
        "--db",
        str(db_file),
    ]

    with patch("sys.argv", test_args):
        with patch("backend.ingestion.ingest_transactions.TransactionPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.ingest_records_document.return_value = ("approved", [])
            mock_pipeline_cls.return_value = mock_pipeline

            main()

            # Then: Both PDF documents should be processed
            assert mock_pipeline.ingest_records_document.call_count == 2


def test_ingest_transactions_explicit_subcommands(tmp_path):
    """Test explicit 'ingest', 'list', and 'delete' subcommands."""
    db_file = tmp_path / "test.db"

    # 1. Test explicit 'list' subcommand
    list_args = [
        "ingest_transactions.py",
        "list",
        "--db",
        str(db_file),
    ]
    with patch("sys.argv", list_args):
        with patch("backend.ingestion.ingest_transactions._handle_list") as mock_list:
            main()
            mock_list.assert_called_once()

    # 2. Test explicit 'delete' subcommand with --all
    delete_args = [
        "ingest_transactions.py",
        "delete",
        "--all",
        "--db",
        str(db_file),
    ]
    with patch("sys.argv", delete_args):
        with patch("backend.ingestion.ingest_transactions._handle_delete_all") as mock_delete_all:
            main()
            mock_delete_all.assert_called_once()
