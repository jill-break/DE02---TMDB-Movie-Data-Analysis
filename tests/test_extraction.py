import pytest
from unittest.mock import MagicMock, patch
from src.extraction.fetch_data import TMDBDataFetcher

@pytest.fixture
def fetcher_instance():
    return TMDBDataFetcher()

@patch('src.extraction.fetch_data.requests.Session.get')
def test_fetch_single_movie_success(mock_get, fetcher_instance):
    """Test successful API fetch with a mocked response."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'id': 123, 'title': 'Mock Movie'}
    mock_get.return_value = mock_response

    result = fetcher_instance.fetch_single_movie(123)

    assert result['title'] == 'Mock Movie'
    assert mock_get.called