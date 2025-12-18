"""Behavioral tests for output tables module.

Tests verify the public interface behavior:
- Table creation with various column options
- Model row addition with different configurations
- Edge cases: empty models, missing values

Tests focus on the structure and content of tables, not Rich rendering details.
"""

from local_ai.models.schema import ModelSearchResult, Quantization
from local_ai.output.tables import (
    add_model_row,
    create_local_models_table,
    create_model_table,
    create_search_results_table,
)


class TestCreateModelTable:
    """Verify create_model_table produces correctly structured tables."""

    def test_creates_table_with_default_columns(self) -> None:
        """Should create table with Model, Downloads, and Likes columns by default."""
        table = create_model_table("Test Table")

        # Extract column names
        column_headers = [col.header for col in table.columns]
        assert "Model" in column_headers
        assert "Downloads" in column_headers
        assert "Likes" in column_headers
        # Author should NOT be present by default (show_full_id=True)
        assert "Author" not in column_headers

    def test_creates_table_with_author_column_when_not_full_id(self) -> None:
        """Should add Author column when show_full_id is False."""
        table = create_model_table("Test Table", show_full_id=False)

        column_headers = [col.header for col in table.columns]
        assert "Model" in column_headers
        assert "Author" in column_headers

    def test_creates_table_with_quant_column_when_enabled(self) -> None:
        """Should add Quant column when show_quant is True."""
        table = create_model_table("Test Table", show_quant=True)

        column_headers = [col.header for col in table.columns]
        assert "Quant" in column_headers

    def test_creates_table_without_downloads_when_disabled(self) -> None:
        """Should omit Downloads column when show_downloads is False."""
        table = create_model_table("Test Table", show_downloads=False)

        column_headers = [col.header for col in table.columns]
        assert "Downloads" not in column_headers

    def test_creates_table_without_likes_when_disabled(self) -> None:
        """Should omit Likes column when show_likes is False."""
        table = create_model_table("Test Table", show_likes=False)

        column_headers = [col.header for col in table.columns]
        assert "Likes" not in column_headers


class TestAddModelRow:
    """Verify add_model_row adds correct data to tables."""

    def test_adds_row_with_full_id(self) -> None:
        """Should add model ID when show_full_id is True."""
        table = create_model_table("Test")
        model = ModelSearchResult(
            id="mlx-community/test-model-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
        )

        add_model_row(table, model)

        # Verify row was added (check row_count)
        assert table.row_count == 1

    def test_adds_row_with_name_and_author_when_not_full_id(self) -> None:
        """Should add separate name and author columns when show_full_id is False."""
        table = create_model_table("Test", show_full_id=False)
        model = ModelSearchResult(
            id="mlx-community/test-model-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
        )

        add_model_row(table, model, show_full_id=False)

        assert table.row_count == 1

    def test_adds_row_with_quantization(self) -> None:
        """Should include quantization value when show_quant is True."""
        table = create_model_table("Test", show_quant=True)
        model = ModelSearchResult(
            id="mlx-community/test-model-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
        )

        add_model_row(table, model, show_quant=True)

        assert table.row_count == 1


class TestCreateSearchResultsTable:
    """Verify create_search_results_table creates populated tables."""

    def test_creates_empty_table_with_no_models(self) -> None:
        """Should create table with zero rows when given empty model list."""
        table = create_search_results_table("Empty Results", models=[])

        assert table.row_count == 0

    def test_creates_table_with_all_models(self) -> None:
        """Should add a row for each model in the list."""
        models = [
            ModelSearchResult(
                id="mlx-community/model-1-4bit",
                author="mlx-community",
                downloads=1000,
                likes=10,
            ),
            ModelSearchResult(
                id="mlx-community/model-2-8bit",
                author="mlx-community",
                downloads=2000,
                likes=20,
            ),
        ]

        table = create_search_results_table("Search Results", models=models)

        assert table.row_count == 2

    def test_creates_table_with_quantization_column(self) -> None:
        """Should include Quant column when show_quant is True."""
        models = [
            ModelSearchResult(
                id="mlx-community/model-4bit",
                author="mlx-community",
                downloads=100,
                likes=5,
            ),
        ]

        table = create_search_results_table("Results", models=models, show_quant=True)

        column_headers = [col.header for col in table.columns]
        assert "Quant" in column_headers

    def test_creates_table_with_author_column(self) -> None:
        """Should include Author column when show_full_id is False."""
        models = [
            ModelSearchResult(
                id="other-org/model-fp16",
                author="other-org",
                downloads=500,
                likes=25,
            ),
        ]

        table = create_search_results_table("Results", models=models, show_full_id=False)

        column_headers = [col.header for col in table.columns]
        assert "Author" in column_headers


class TestCreateLocalModelsTable:
    """Verify create_local_models_table creates correct structure."""

    def test_creates_table_with_size_column(self) -> None:
        """Should include Size column for local models."""
        models = [
            ModelSearchResult(
                id="mlx-community/local-model-4bit",
                author="mlx-community",
                downloads=0,
                likes=0,
                size_bytes=4_000_000_000,
            ),
        ]

        table = create_local_models_table(models)

        column_headers = [col.header for col in table.columns]
        assert "Size" in column_headers
        assert "Quant" in column_headers
        assert table.row_count == 1

    def test_creates_empty_table_for_no_local_models(self) -> None:
        """Should create empty table when no local models."""
        table = create_local_models_table(models=[])

        assert table.row_count == 0

    def test_uses_custom_title(self) -> None:
        """Should use provided title for the table."""
        table = create_local_models_table(models=[], title="My Custom Title")

        assert table.title == "My Custom Title"

    def test_handles_model_with_none_size(self) -> None:
        """Should handle models with no size information gracefully."""
        models = [
            ModelSearchResult(
                id="mlx-community/unknown-size",
                author="mlx-community",
                downloads=0,
                likes=0,
                size_bytes=None,
            ),
        ]

        table = create_local_models_table(models)

        # Should still create table without error
        assert table.row_count == 1
